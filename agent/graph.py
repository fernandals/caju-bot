from agent.nodes import generate_answer, decide, update_tracking
from agent.state import TutorState, TutorConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from functools import partial
from agent.nodes import generate_answer, decide, update_tracking, grade_documents  

def build_graph(config: TutorConfig, retrieve_tool, response_model) -> StateGraph[TutorState]:
    graph = StateGraph(TutorState)

    graph.add_node("tracking", update_tracking)
    graph.add_node("decide", partial(decide, config=config, model=response_model))
    graph.add_node("retrieve", ToolNode([retrieve_tool]))
    graph.add_node("grade_documents", partial(grade_documents, config=config, model=response_model))
    graph.add_node("generate_answer", partial(generate_answer, config=config, model=response_model))

    graph.add_edge(START, "tracking")
    graph.add_edge("tracking", "decide")

    graph.add_conditional_edges(
        "decide",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )

    graph.add_edge("retrieve", "grade_documents")     
    graph.add_edge("grade_documents", "generate_answer") 
    graph.add_edge("generate_answer", END)

    graph = graph.compile()

    print("--> Graph Visualization:")
    print(graph.get_graph().draw_ascii())

    return graph