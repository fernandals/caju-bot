from agent.nodes import generate_answer, plan_instruction, update_tracking, grade_documents, retrieve_documents
from agent.state import TutorState, TutorConfig
from langgraph.graph import StateGraph, START, END
from functools import partial

def route_after_planning(state: TutorState):
    if state["answer_plan"].needs_retrieval:
        return "retrieve"

    return "generate_answer"

def build_graph(config: TutorConfig, retriever, models) -> StateGraph[TutorState]:
    graph = StateGraph(TutorState)

    graph.add_node("tracking", partial(update_tracking, model=models.tracking_llm))
    graph.add_node("planning", partial(plan_instruction, config=config, model=models.planning_llm))
    graph.add_node("retrieve", partial(retrieve_documents, retriever=retriever))
    graph.add_node("grade_documents", partial(grade_documents, config=config, model=models.grading_llm))
    graph.add_node("generate_answer", partial(generate_answer, config=config, model=models.generation_llm))

    graph.add_edge(START, "tracking")
    graph.add_edge("tracking", "planning")

    graph.add_conditional_edges(
        "planning",
        route_after_planning,
        {
            "retrieve": "retrieve",
            "generate_answer": "generate_answer",
        },
    )

    graph.add_edge("retrieve", "grade_documents")     
    graph.add_edge("grade_documents", "generate_answer") 
    graph.add_edge("generate_answer", END)

    graph = graph.compile()

    print("--> Graph Visualization:")
    print(graph.get_graph().draw_ascii())

    return graph # type: ignore