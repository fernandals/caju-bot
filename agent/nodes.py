import agent.prompts as prompts
from agent.state import AnswerPlan, LearningState, TutorState, TutorConfig
from langchain_core.messages import HumanMessage, SystemMessage
from agent.grader import GradeDocument

def update_tracking(state: TutorState, model):
    """
    Infers and updates the student's current pedagogical learning state
    from the recent conversation context.
    """

    print("-------> Updating tracking information...")

    # -------------------------
    # Recent conversation window
    # -------------------------

    recent_messages = state["messages"][-6:]

    conversation = []

    for msg in recent_messages:
        role = "Student"

        if not isinstance(msg, HumanMessage):
            role = "Tutor"

        conversation.append(f"{role}: {msg.content}")

    conversation_text = "\n".join(conversation)

    # -------------------------
    # Previous learning state
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

Previous learning state:
{previous_state_text}

Recent conversation:
{conversation_text}
"""

    # -------------------------
    # Structured extraction
    # -------------------------

    structured_llm = model.with_structured_output(LearningState)

    new_learning_state = structured_llm.invoke(prompt)

    # -------------------------
    # Return updated state
    # -------------------------

    return {
        "messages": state["messages"],
        "learning_state": new_learning_state,
    }

def plan_instruction(state: TutorState, config: TutorConfig, model):
    """
    Determines the pedagogical response strategy for the current interaction.
    The node does NOT generate the final answer.
    It only produces a structured instructional plan that downstream
    nodes will execute.
    """
    print("-------> Planning instructional strategy...")

    learning_state = state["learning_state"]

    system_prompt = prompts.PLANNING_PROMPT.format(
        domain=config.subject,
        course_level=config.course_level,
    )

    planning_context = f"""
Current learning state:
{learning_state.model_dump_json(indent=2)}

Course configuration:
- Subject: {config.subject}
- Course level: {config.course_level}
- Maximum sentences: {config.max_sentences}
- Answer language: {config.answer_language}
"""

    structured_model = model.with_structured_output(AnswerPlan)

    answer_plan = structured_model.invoke(
        [
            SystemMessage(content=system_prompt),
            SystemMessage(content=planning_context),
        ] + state["messages"][-6:]
    )

    return {
        "messages": state["messages"],
        "learning_state": learning_state,
        "answer_plan": answer_plan,
    }

def retrieve_documents(state: TutorState, retriever):
    print("-------> Retrieving documents...")

    learning_state = state["learning_state"]
    answer_plan = state["answer_plan"]

    topic = learning_state.topic or ""

    concepts = answer_plan.concepts_to_cover

    query_parts = [
        topic,
        *concepts,
    ]

    query = "\n".join(query_parts)

    docs = retriever.invoke(query)

    return {
        "messages": state["messages"],
        "learning_state": learning_state,
        "answer_plan": answer_plan,
        "retrieved_docs": docs,
    }

def generate_answer(state: TutorState, config: TutorConfig, model):
    """
    Generates the final pedagogical response based on:
    - current learning state
    - instructional plan
    - retrieved documents (if available)
    """

    print("-------> Generating answer...")

    # -------------------------
    # Latest student question
    # -------------------------

    question = next(
        msg.content
        for msg in reversed(state["messages"])
        if isinstance(msg, HumanMessage)
    )

    # -------------------------
    # Retrieved context
    # -------------------------

    retrieved_docs = state.get("retrieved_docs", [])

    print(f"{len(retrieved_docs)} documents retrieved for generation.")

    context_blocks = []

    for i, doc in enumerate(retrieved_docs, 1):
        block = f"""
[DOCUMENT {i}]

CONTENT:
{doc.page_content}

METADATA:
{doc.metadata}
"""
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    # -------------------------
    # Learning state
    # -------------------------

    learning_state = state["learning_state"]

    # -------------------------
    # Instructional plan
    # -------------------------

    answer_plan = state["answer_plan"]

    # -------------------------
    # Prompt
    # -------------------------

    system_prompt = SystemMessage(
        content=prompts.GENERATE_PROMPT.format(
            domain=config.subject,
            question=question,
            context=context,
            learning_state=learning_state.model_dump_json(indent=2),
            answer_plan=answer_plan.model_dump_json(indent=2),
            max_sentences=config.max_sentences,
            answer_language=config.answer_language,
        )
    )

    # -------------------------
    # Generation
    # -------------------------

    response = model.invoke(
        [system_prompt] + state["messages"]
    )

    return {
        "messages": [response],
        "learning_state": learning_state,
        "answer_plan": answer_plan,
        "retrieved_docs": retrieved_docs,
    }

def grade_documents(state: TutorState, config: TutorConfig, model):
    """
    Filters retrieved documents by semantic relevance
    before answer generation.
    """

    print("-------> Grading retrieved documents for relevance...")

    # -------------------------
    # Latest user question
    # -------------------------

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
    
    # -------------------------
    # Retrieved docs
    # -------------------------

    retrieved_docs = state.get("retrieved_docs", [])

    print(
        f"{len(retrieved_docs)} chunks retrieved for grading."
    )

    if not retrieved_docs:
        return {
            "retrieved_docs": [],
        }

    # -------------------------
    # Structured grader
    # -------------------------

    grader = model.with_structured_output(GradeDocument)

    system = SystemMessage(
        content=prompts.GRADE_PROMPT.format(
            domain=config.subject
        )
    )

     # -------------------------
    # Grade each chunk
    # -------------------------

    relevant_docs = []

    for doc in retrieved_docs:

        result: GradeDocument = grader.invoke([
            system,
            HumanMessage(
                content=f"""
Student question:
{question}

Retrieved chunk:
{doc.page_content}
"""
            )
        ])

        if result.relevant:
            relevant_docs.append(doc)

    # -------------------------
    # Fallback
    # -------------------------

    filtered_docs = (
        relevant_docs
        if relevant_docs
        else retrieved_docs
    )

    print(
        f"{len(filtered_docs)} chunks kept after grading."
    )

    # -------------------------
    # Return updated state
    # -------------------------

    return {
        "messages": state["messages"],
        "learning_state": state["learning_state"],
        "answer_plan": state["answer_plan"],
        "retrieved_docs": filtered_docs,
    }
