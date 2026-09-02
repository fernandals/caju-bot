import logging
import os

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

import agent.prompts as prompts
from agent.graph import build_graph
from agent.models import ModelRegistry
from agent.state import TutorConfig
from rag.knowledge_base import load_knowledge_base

logger = logging.getLogger(__name__)

def load_pipeline(discipline_name: str):
    """
    Load the tutoring pipeline for a given discipline.

    Args:
        discipline_name (str): The name of the discipline.

    Returns:
        tuple: (compiled graph, system prompt, tutor config)
    """
    logger.info("Loading pipeline for discipline: %s", discipline_name)

    # --------------------------------------------------
    # Load retriever
    # --------------------------------------------------
    # load_knowledge_base raises ValueError with a clear message if the
    # discipline has no persisted KB - let it propagate.
    retriever = load_knowledge_base(discipline_name)

    # --------------------------------------------------
    # Build Configuration
    # --------------------------------------------------
    config = TutorConfig(
        subject=discipline_name,
        course_level=os.getenv("COURSE_LEVEL", "beginner"),
        answer_language=os.getenv("ANSWER_LANGUAGE", "Português"),
        allow_direct_answers=os.getenv("ALLOW_DIRECT_ANSWERS", "True").lower() == "true"
    )

    # --------------------------------------------------
    # System Prompt
    # --------------------------------------------------
    tutor_prompt = SystemMessage(
        content=prompts.SYSTEM_PROMPT.format(
            domain=config.subject,
            max_sentences=config.max_sentences,
            course_level=config.course_level,
            answer_language=config.answer_language
        )
    )
    
    # --------------------------------------------------
    # Model Configuration
    # --------------------------------------------------
    model_temperature = float(os.getenv("MODEL_TEMPERATURE", 0))

    # The graph runs synchronously inside a worker thread (see
    # chainlit_app._run_graph_sync), so a call with no timeout would block
    # that student's whole turn indefinitely on a network stall. Bound it
    # and let a couple of retries absorb transient failures.
    request_timeout = float(os.getenv("MODEL_TIMEOUT", 30))
    max_retries = int(os.getenv("MODEL_MAX_RETRIES", 2))

    model_names = {
        "generation": os.getenv("GENERATION_MODEL", "gpt-4o-mini"),
        "tracking": os.getenv("TRACKING_MODEL", "gpt-4.1-nano"),
        "planning": os.getenv("PLANNING_MODEL", "gpt-4.1-mini"),
        "grading": os.getenv("GRADING_MODEL", "gpt-4.1-nano")
    }

    logger.info("Using models: %s", model_names)

    def _chat(model_name: str) -> ChatOpenAI:
        return ChatOpenAI(
            model=model_name,
            temperature=model_temperature,
            timeout=request_timeout,
            max_retries=max_retries,
        )

    models = ModelRegistry(
        generation_llm=_chat(model_names["generation"]),
        tracking_llm=_chat(model_names["tracking"]),
        planning_llm=_chat(model_names["planning"]),
        grading_llm=_chat(model_names["grading"]),
    )

    # --------------------------------------------------
    # Build Graph
    # --------------------------------------------------
    graph = build_graph(config, retriever, models)

    logger.info("Pipeline loaded successfully for discipline: %s", discipline_name)

    return graph, tutor_prompt, config