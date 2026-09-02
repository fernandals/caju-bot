import logging
from functools import partial

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes import (
    assess_documents,
    generate_answer,
    plan_instruction,
    retrieve_documents,
    update_tracking,
)
from agent.state import TutorConfig, TutorState

logger = logging.getLogger(__name__)


def route_after_planning(state: TutorState):
    if state["answer_plan"].needs_retrieval:
        return "retrieve"

    return "generate_answer"

def build_graph(config: TutorConfig, retriever, models) -> CompiledStateGraph:
    graph = StateGraph(TutorState)

    graph.add_node("tracking", partial(update_tracking, model=models.tracking_llm))
    graph.add_node("planning", partial(plan_instruction, config=config, model=models.planning_llm))
    graph.add_node("retrieve", partial(retrieve_documents, retriever=retriever))
    graph.add_node("assess_documents", partial(assess_documents, config=config, model=models.grading_llm))
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

    graph.add_edge("retrieve", "assess_documents")
    graph.add_edge("assess_documents", "generate_answer")
    graph.add_edge("generate_answer", END)

    compiled = graph.compile()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Tutor graph:\n%s", compiled.get_graph().draw_ascii())

    return compiled
