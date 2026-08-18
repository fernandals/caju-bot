import json
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

import agent.prompts as prompts
from agent.grader import GradeDocument
from agent.state import (
    AnswerPlan,
    ChunkEvidence,
    LearningState,
    TeachingState,
    TutorConfig,
    TutorState,
)
from agent.teaching import advance_teaching_state
from rag.citation import format_citation
from rag.models import ChunkMetadata


def build_conversation_transcript(messages) -> str:
    """
    Renders messages as a clean {turn, role, content} transcript for LLM
    prompts, instead of interpolating raw BaseMessage objects (which
    stringify with noisy internals like ids/additional_kwargs/
    response_metadata and waste tokens).
    """

    conversation = [
        {
            "turn": i,
            "role": "student" if isinstance(msg, HumanMessage) else "tutor",
            "content": msg.content,
        }
        for i, msg in enumerate(messages)
    ]

    return json.dumps(conversation, indent=2, ensure_ascii=False)


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
    conversation_text = build_conversation_transcript(state["messages"])

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

    proposed_teaching_state = advance_teaching_state(
        state.get("teaching_state") or TeachingState(),
        learning_state,
    )

    planning_context = f"""
Current learning state:
{learning_state.model_dump_json(indent=2)}

Proposed teaching stage (already decided by the system; see STRATEGY
RULES above for how to use it):
{proposed_teaching_state.model_dump_json(indent=2)}
"""

    prompt = f"""
{prompts.PLANNING_PROMPT}

---

{planning_context}

Recent conversation:
{build_conversation_transcript(state["messages"][-6:])}
"""

    structured_model = model.with_structured_output(AnswerPlan)

    answer_plan = structured_model.invoke(prompt)

    # The only planning-time override of the deterministic pacing: the LLM
    # judged the student explicitly wants directness even though the
    # proposed stage was "guided". Keep topic_anchor/stage intact so a
    # guided arc can resume next turn if the student re-engages. Gated on
    # config.allow_direct_answers: when a course wants guidance enforced,
    # this specific escape hatch is disabled (the frustration/exam_prep
    # escape valves in advance_teaching_state still apply regardless, since
    # those are about the student's wellbeing, not about skipping effort).
    teaching_state = proposed_teaching_state

    if (
        answer_plan.strategy == "direct_answer"
        and teaching_state.mode == "guided"
        and config.allow_direct_answers
    ):
        teaching_state = teaching_state.model_copy(update={"mode": "direct"})

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"[PLANNING NODE] Execution time: {execution_time:.2f} seconds")

    return {
        "answer_plan": answer_plan,
        "teaching_state": teaching_state,
    }

def retrieve_documents(state: TutorState, retriever):
    """
    Retrieves relevant instructional documents based on the current learning
    state and the student's question.
    """

    print("\n========== START DOCUMENT RETRIEVAL ==========")

    start_time = time.perf_counter()

    learning_state = state["learning_state"]

    question = next(
        msg.content
        for msg in reversed(state["messages"])
        if isinstance(msg, HumanMessage)
    )

    # Augment with topic/subtopic, which are genuinely part of what's being
    # asked about. Previously this also injected intent/strategy-derived
    # phrases (e.g. "exercises examples practice problems") into the
    # embedding query as a relevance "boost" - removed: there was no
    # evidence it improved retrieval, and stuffing unrelated keywords into
    # a semantic search query risks diluting it away from the actual
    # question rather than sharpening it.
    query_parts = [question]

    if learning_state.topic:
        query_parts.append(learning_state.topic)

    if learning_state.subtopic:
        query_parts.append(learning_state.subtopic)

    query = " ".join(query_parts) # type: ignore

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

    grading_inputs = [
        [
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
        ]
        for doc in retrieved_docs
    ]

    results: list[GradeDocument] = grader.batch(grading_inputs)

    scored_docs = list(zip(retrieved_docs, results))

    # ordenar por score
    scored_docs.sort(
        key=lambda x: x[1].relevance_score,
        reverse=True
    )

    # threshold
    filtered_docs = [
        doc
        for doc, result
        in scored_docs
        if result.relevance_score >= 0.5
    ]

    # fallback: only when the top scores are at least weakly relevant.
    # Below this floor, the chunks are clearly irrelevant and shouldn't be
    # forced into the answer just to have "something" to cite.
    if not filtered_docs:
        filtered_docs = [
            doc
            for doc, result
            in scored_docs[:3]
            if result.relevance_score >= 0.2
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

    extraction_inputs = [
        [
            system,
            HumanMessage(
                content=f"""
Chunk:

{doc.page_content}
"""
            )
        ]
        for doc in retrieved_docs
    ]

    results: list[ChunkEvidence] = extractor.batch(extraction_inputs)

    evidence = []

    for i, (doc, result) in enumerate(zip(retrieved_docs, results), start=1):

        metadata = ChunkMetadata.model_validate(doc.metadata) # type: ignore

        result.doc_id = f"DOC_{i}"
        result.citation = format_citation(metadata)

        evidence.append(result)

        print(f"[EVIDENCE EXTRACTION] Processed DOC_{i} / {len(retrieved_docs)}")
        print(f"[EVIDENCE EXTRACTION] Result: {result.model_dump_json(indent=2)}")
        print(f"[EVIDENCE EXTRACTION] Retrieved document content: {doc.page_content}")
        print(f"[EVIDENCE EXTRACTION] Retrieved document metadata: {metadata}")

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
    teaching_state = state.get("teaching_state") or TeachingState()

    planning_rationale = getattr(
        answer_plan,
        "rationale",
        "",
    )

    teaching_instructions = (
        prompts.DIRECT_MODE_INSTRUCTIONS
        if teaching_state.mode == "direct"
        else prompts.TEACHING_STAGE_INSTRUCTIONS[teaching_state.stage]
    )

    context = ""
    citations_by_doc_id = {}

    for ev in state["evidence"]:

        citations_by_doc_id[ev.doc_id] = ev.citation

        context += f"""
=======================================

SOURCE
{ev.doc_id}

CITE_AS
[[CITE:{ev.doc_id}]]

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
            teaching_instructions=teaching_instructions,
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

    if isinstance(response.content, str):
        response = response.model_copy(
            update={
                "content": substitute_citation_markers(
                    response.content,
                    citations_by_doc_id,
                )
            }
        )

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"[ANSWER GENERATION] Execution time: {execution_time:.2f} seconds")

    return {
        "messages": [response]
    }

CITATION_MARKER_REGEX = re.compile(r"\[\[CITE:(DOC_\d+)\]\]")

def substitute_citation_markers(
    text: str,
    citations_by_doc_id: dict[str, str],
) -> str:
    """
    Replaces opaque [[CITE:DOC_n]] markers left by the generation model
    with the real, deterministically-built citation for that source.

    The model is deliberately never shown the human-readable citation text,
    so it can't accidentally translate, paraphrase, or otherwise alter it
    while writing the answer.
    """

    def replace(match: re.Match) -> str:
        return citations_by_doc_id.get(match.group(1), "")

    return CITATION_MARKER_REGEX.sub(replace, text)