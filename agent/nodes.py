import json
import time

from langchain_core.messages import HumanMessage, SystemMessage

import agent.prompts as prompts
from agent.grader import GradeDocument
from agent.state import (
    AnswerPlan,
    ChunkEvidence,
    LearningState,
    TutorConfig,
    TutorState,
)


def update_tracking(state: TutorState, model):
    """
    Infers and updates the student's current pedagogical learning state
    from the recent conversation context.
    """

    print("\n========== START TRACKING ==========")

    start_time = time.perf_counter()

    # -------------------------
    # Conversation context
    # -------------------------
    messages = state["messages"]

    conversation = []

    for i, msg in enumerate(messages):
        role = "student" if isinstance(msg, HumanMessage) else "tutor"

        conversation.append({
            "turn": i,
            "role": role,
            "content": msg.content
        })

    conversation_text = json.dumps(conversation, indent=2, ensure_ascii=False)

    # -------------------------
    # Previous state
    # -------------------------
    previous_state = state.get("learning_state")

    previous_state_text = (
        previous_state.model_dump_json(indent=2)
        if previous_state
        else "None"
    )

    # -------------------------
    # Build prompt
    # -------------------------

    prompt = f"""
{prompts.TRACKING_PROMPT}

---

Previous learning state:
{previous_state_text}

---

Recent conversation:
{conversation_text}
"""

    # -------------------------
    # Structured output
    # -------------------------
    structured_llm = model.with_structured_output(LearningState)

    new_learning_state = structured_llm.invoke(prompt)

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"[TRACKING NODE] Execution time: {execution_time:.2f} seconds")

    return {
        "learning_state": new_learning_state,
    }

def plan_instruction(state: TutorState, config: TutorConfig, model):
    """
    Determines the pedagogical response strategy for the current interaction.
    The node does NOT generate the final answer.
    It only produces a structured instructional plan that downstream
    nodes will execute.
    """

    print("\n========== START PLANNING ==========")

    start_time = time.perf_counter()

    learning_state = state["learning_state"]

    planning_context = f"""
Current learning state:
{learning_state.model_dump_json(indent=2)}
"""

    prompt = f"""
{prompts.PLANNING_PROMPT}

---

{planning_context}

Recent conversation:
{state["messages"][-6:]}
"""

    structured_model = model.with_structured_output(AnswerPlan)

    answer_plan = structured_model.invoke(prompt)

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"[PLANNING NODE] Execution time: {execution_time:.2f} seconds")

    return {
        "answer_plan": answer_plan,
    }

def retrieve_documents(state: TutorState, retriever):
    """
    Retrieves relevant instructional documents based on the current learning state
    and the student's question. Implements adaptive query construction to boost
    retrieval performance.
    """

    print("\n========== START DOCUMENT RETRIEVAL ==========")

    start_time = time.perf_counter()

    learning_state = state["learning_state"]
    answer_plan = state["answer_plan"]

    question = next(
        msg.content
        for msg in reversed(state["messages"])
        if isinstance(msg, HumanMessage)
    )

    topic = learning_state.topic or ""
    subtopic = learning_state.subtopic or ""
    intent = learning_state.intent

    strategy = answer_plan.strategy

    # -------------------------
    # Adaptive query building
    # -------------------------

    query_parts = []

    # 1. question
    query_parts.append(question)

    # 2. topic and subtopic
    if topic:
        query_parts.append(topic)

    if subtopic:
        query_parts.append(subtopic)

    # 3. intent-aware boost
    if intent in ["debug_confusion", "learn"]:
        query_parts.append("explanation conceptual intuition")

    elif intent == "practice":
        query_parts.append("exercises examples practice problems")

    elif intent == "solve_problem":
        query_parts.append("step by step solution reasoning")

    elif intent == "exam_prep":
        query_parts.append("summary key points review")
    
    # 4. strategy-aware boost
    if strategy == "guided_teaching":
        query_parts.append("intuitive explanation examples")

    elif strategy == "exercise_first":
        query_parts.append("practice questions exercises")

    elif strategy == "step_by_step":
        query_parts.append("step by step breakdown")

    elif strategy == "hint_only":
        query_parts.append("minimal guidance hints")

    query = " ".join(query_parts)

    docs = retriever.invoke(query)

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"[DOCUMENT RETRIEVAL] Execution time: {execution_time:.2f} seconds")

    return {
        "retrieved_docs": docs,
    }

def grade_documents(state: TutorState, config: TutorConfig, model):
    """
    Filters retrieved documents by semantic relevance
    before answer generation.
    """

    print("\n========== START DOCUMENT GRADING ==========")

    start_time = time.perf_counter()

    question = next(
        (
            msg.content
            for msg in reversed(state["messages"])
            if isinstance(msg, HumanMessage)
        ),
        None
    )

    if not question:
        raise ValueError(
            "No HumanMessage found during document grading."
        )

    retrieved_docs = state.get("retrieved_docs", [])

    if not retrieved_docs:
        return {
            "retrieved_docs": [],
        }

    learning_state = state["learning_state"]

    grader = model.with_structured_output(GradeDocument)

    system = SystemMessage(
        content=prompts.GRADE_PROMPT.format(
            domain=config.subject
        )
    )

    scored_docs = []

    for doc in retrieved_docs:

        result: GradeDocument = grader.invoke([
            system,
            HumanMessage(
                content=f"""
Student question:
{question}

Learning state:
{learning_state.model_dump_json(indent=2)}

Retrieved chunk:
{doc.page_content}
""" 
            )
        ])

        scored_docs.append(
            (
                doc,
                result.relevance_score,
                result.reason
            )
        )

    # ordenar por score
    scored_docs.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # threshold
    filtered_docs = [
        doc
        for doc, score, _
        in scored_docs
        if score >= 0.5
    ]

    # fallback
    if not filtered_docs:
        filtered_docs = [
            doc
            for doc, _, _
            in scored_docs[:3]
        ]

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"[DOCUMENT GRADING] Execution time: {execution_time:.2f} seconds")

    return {
        "retrieved_docs": filtered_docs,
    }
 
def extract_evidence(state: TutorState, config: TutorConfig, model):

    print("\n========== START EVIDENCE EXTRACTION ==========")

    start_time = time.perf_counter()

    retrieved_docs = state.get("retrieved_docs", [])

    if not retrieved_docs:
        return {
            "evidence": []
        }

    extractor = model.with_structured_output(
        ChunkEvidence
    )

    system = SystemMessage(
        content=prompts.EVIDENCE_PROMPT
    )

    evidence = []

    for i, doc in enumerate(retrieved_docs, start=1):

        metadata = doc.metadata # type: ignore

        result = extractor.invoke([
            system,
            HumanMessage(
                content=f"""
Reference ID:
DOC_{i}

Metadata:

{metadata}

Chunk:

{doc.page_content}
"""
            )
        ])

        result.doc_id = f"DOC_{i}"

        evidence.append(result)

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"[EVIDENCE EXTRACTION] Execution time: {execution_time:.2f} seconds")

    return {
        "evidence": evidence,
    }

def generate_answer(state: TutorState, config: TutorConfig, model):
    """
    Generates the final answer to the student's question based on the
    current learning state, the instructional plan, and the retrieved
    documents.
    """

    print("\n========== START ANSWER GENERATION ==========")

    start_time = time.perf_counter()

    question = next(
        msg.content
        for msg in reversed(state["messages"])
        if isinstance(msg, HumanMessage)
    )

    learning_state = state["learning_state"]
    answer_plan = state["answer_plan"]

    planning_rationale = getattr(
        answer_plan,
        "rationale",
        "",
    )

    context = ""

    for ev in state["evidence"]:

        context += f"""
=======================================

SOURCE
{ev.doc_id}

CITATION
{ev.citation}

EVIDENCE
{chr(10).join("- "+e for e in ev.evidence)}
"""

    system_prompt = SystemMessage(
        content=prompts.GENERATE_PROMPT.format(
            domain=config.subject,
            question=question,
            learning_state=learning_state.model_dump_json(
                indent=2
            ),
            answer_plan=answer_plan.model_dump_json(
                indent=2
            ),
            planning_rationale=planning_rationale,
            context=context,
            answer_language=config.answer_language,
            max_sentences=config.max_sentences,
        )
    )

    conversation_window = state["messages"][-8:]

    response = model.invoke(
        [system_prompt]
        + conversation_window
    )

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"[ANSWER GENERATION] Execution time: {execution_time:.2f} seconds")

    return {
        "messages": [response]
    }