import agent.prompts as prompts
from agent.state import LearningState, TutorState, TutorConfig
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from student_model.updater import update_student_profile, update_conversation_topic
from agent.grader import GradeDocument

def decide(state: TutorState, config: TutorConfig, model):
    """Decides whether to call the retrieval tool or generate direct answer
    based on the current conversation state."""
    print("-------> Deciding next action...")

    system_prompt = prompts.DECIDE_PROMPT.format(domain=config.subject)

    response = (
        model.invoke(
            [SystemMessage(system_prompt)] + state["messages"]
        )
    )

    return {"messages": [response], "student_profile": state["student_profile"]}

def generate_answer(state: TutorState, config: TutorConfig, model):
    """Generates an answer based on the current conversation state and student profile."""
    print("-------> Generating answer...")

    # utlima mensagem do usuário
    question = next(
        msg.content for msg in reversed(state["messages"])
        if isinstance(msg, HumanMessage)
    )
    # ultima resposta do retriever
    context = next(
        msg.content for msg in reversed(state["messages"])
        if isinstance(msg, ToolMessage)
    ) 
    
    # ------ geração --------
    system_prompt = SystemMessage(
        content=prompts.GENERATE_PROMPT.format(
            domain=config.subject,
            question=question,
            context=context
        )
    )

    response = model.invoke(
        [system_prompt] + state["messages"]
    )

    final_response = response

    return {"messages": [final_response], "student_profile": state["student_profile"]}

def grade_documents(state: TutorState, config: TutorConfig, model):
    """Filters retrieved chunks by relevance before generation."""
    print("-------> Grading retrieved documents for relevance...")

    question = next(
        (msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        None
    )
    if not question:
        raise ValueError("No HumanMessage found during document grading.")

    tool_message = next(
        (msg for msg in reversed(state["messages"]) if isinstance(msg, ToolMessage)),
        None
    )
    chunks = [c.strip() for c in tool_message.content.split("---CHUNK---") if c.strip()]
    print(len(chunks), "chunks retrieved for grading.")

    grader = model.with_structured_output(GradeDocument)

    system = SystemMessage(content=(prompts.GRADE_PROMPT.format(domain=config.subject)))

    relevant_chunks = []
    for chunk in chunks:
        result: GradeDocument = grader.invoke([
            system,
            HumanMessage(content=f"Question: {question}\n\nChunk: {chunk}")
        ])
        if result.relevant:
            relevant_chunks.append(chunk)

    # if nothing passed, keep all chunks as fallback to avoid empty context
    filtered = relevant_chunks if relevant_chunks else chunks

    # rebuild messages replacing old ToolMessages with filtered ones
    non_tool_messages = [msg for msg in state["messages"] if not isinstance(msg, ToolMessage)]
    return {"messages": non_tool_messages + filtered}

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
