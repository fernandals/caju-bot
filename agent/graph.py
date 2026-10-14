import logging
import re
from functools import partial

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes import (
    assess_documents,
    generate_answer,
    greet,
    latest_student_question,
    list_topics,
    plan_instruction,
    retrieve_documents,
    update_tracking,
)
from agent.state import TutorConfig, TutorState

logger = logging.getLogger(__name__)


# A turn goes to the cheap `greet` node only if it is confidently nothing
# but a greeting / acknowledgement / sign-off AND tracking pulled no topic
# from it. Anything else - including a short real request tracking failed
# to parse - falls through to planning, where a wasted turn is worse than
# an extra one. Matched against the whole (stripped) message.
_SMALLTALK_RE = re.compile(
    r"^(oi+|ol[áa]+|e a[íi]|ea[íi]|al[ôo]|hey+|hi+|hello+|yo|"
    r"bom dia|boa tarde|boa noite|tudo bem|tudo certo|tudo joia|como vai|"
    r"beleza|blz|de boa|"
    r"obrigad[oa]|obg|valeu|vlw|agradec[ei]d[oa]|"
    r"tchau|at[ée] mais|at[ée] logo|falou|"
    r"ok|okay|okey|entendi|entendido|certo|show|legal|bacana|massa|"
    r"thanks|thank you|thx|bye)"
    r"[\s!.,;:)+_\-]*$",
    re.IGNORECASE,
)


# "What can I study?" - the TOPICS starter button's own message, or the
# same question typed. Answered from the course outline (list_topics),
# not by the teaching pipeline: guided teaching would answer with a
# question back. Kept narrow - a topic-specific "o que é X?" must not
# match - and anchored to the whole message.
_TOPICS_RE = re.compile(
    r"(?:e\s+)?"
    r"(?:"
    # "quais (são os) assuntos/tópicos/conteúdos (da disciplina) (posso estudar com você)"
    r"(?:quais|que)\s+(?:s[ãa]o\s+)?(?:os\s+)?"
    r"(?:assuntos|t[óo]picos|temas|conte[úu]dos)"
    r"(?:\s+(?:da|de|do|dessa|desta)\s+(?:disciplina|mat[ée]ria|curso|cadeira))?"
    r"(?:\s+(?:que\s+)?(?:eu\s+)?(?:posso|d[áa]\s+p(?:ra|ara)|consigo|podemos|vamos)\s+"
    r"(?:estudar|ver|aprender|revisar))?"
    r"|"
    # "o que (eu) posso estudar"
    r"o\s+que\s+(?:eu\s+)?(?:posso|d[áa]\s+p(?:ra|ara)|consigo|podemos)\s+(?:estudar|aprender)"
    r")"
    r"(?:\s+(?:com\s+voc[êe]|aqui))?"
    r"(?:\s+(?:nessa|nesta|na)\s+(?:disciplina|mat[ée]ria|cadeira))?"
    r"[\s?!.]*",
    re.IGNORECASE,
)


def is_topics_request(text: str | None) -> bool:
    """Whole message asks which topics the course covers."""
    return bool(text) and bool(_TOPICS_RE.fullmatch(text.strip()))


def is_smalltalk(text: str | None) -> bool:
    """Whole message is only a greeting / acknowledgement / sign-off."""
    return bool(text) and bool(_SMALLTALK_RE.match(text.strip()))


def route_start(state: TutorState, config: TutorConfig):
    """
    "What can I study?" is answered straight from the course outline -
    before tracking, so it neither costs a tracking call nor leaves a
    vague "the whole course" topic in the learning state. Without an
    outline it goes through the normal pipeline.
    """
    if config.course_outline and is_topics_request(
        latest_student_question(state["messages"])
    ):
        return "list_topics"

    return "tracking"


def route_after_tracking(state: TutorState):
    """
    Skip the full teaching pipeline for a turn that is only a greeting or
    small talk. Anything that set a topic / open question / difficulty goes
    to planning; so does anything that isn't a confident small-talk match.
    """
    learning_state = state["learning_state"]

    if (
        learning_state.topic
        or learning_state.open_question
        or learning_state.current_difficulty
    ):
        return "planning"

    if is_smalltalk(latest_student_question(state["messages"])):
        return "greet"

    return "planning"


def route_after_planning(state: TutorState):
    if state["answer_plan"].needs_retrieval:
        return "retrieve"

    return "generate_answer"

def build_graph(config: TutorConfig, retriever, models) -> CompiledStateGraph:
    graph = StateGraph(TutorState)

    graph.add_node("tracking", partial(update_tracking, model=models.tracking_llm))
    graph.add_node("greet", partial(greet, config=config, model=models.generation_llm))
    graph.add_node("list_topics", partial(list_topics, config=config))
    graph.add_node("planning", partial(plan_instruction, config=config, model=models.planning_llm))
    graph.add_node("retrieve", partial(retrieve_documents, retriever=retriever, config=config))
    graph.add_node("assess_documents", partial(assess_documents, config=config, model=models.grading_llm))
    graph.add_node(
        "generate_answer",
        partial(
            generate_answer,
            config=config,
            model=models.generation_llm,
            review_model=models.review_llm,
        ),
    )

    graph.add_conditional_edges(
        START,
        partial(route_start, config=config),
        {
            "list_topics": "list_topics",
            "tracking": "tracking",
        },
    )

    graph.add_edge("list_topics", END)

    graph.add_conditional_edges(
        "tracking",
        route_after_tracking,
        {
            "greet": "greet",
            "planning": "planning",
        },
    )

    graph.add_edge("greet", END)

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
