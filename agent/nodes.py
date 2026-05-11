import agent.prompts as prompts
from agent.state import TutorState, TutorConfig
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

def update_tracking(state: TutorState):
    """Updates the student profile and topic based on the conversation history."""
    print("-------> Updating tracking information...")

    last_user_msg = next(
        msg.content for msg in reversed(state["messages"])
        if isinstance(msg, HumanMessage)
    )

    profile = update_student_profile(state.get("student_profile"), last_user_msg)
    #topic = update_conversation_topic(state.get("current_topic"), last_user_msg)

    return {"messages": state["messages"], "student_profile": profile, "current_topic": None}