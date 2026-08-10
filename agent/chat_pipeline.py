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
    try:
        retriever = load_knowledge_base(discipline_name)
    except ValueError as e:
        logger.error("Failed to load knowledge base: %s", e)
        raise

    # --------------------------------------------------
    # Build Configuration
    # --------------------------------------------------
    config = TutorConfig(
        subject=discipline_name,
        course_level=os.getenv("COURSE_LEVEL", "beginner"),
        answer_language=os.getenv("ANSWER_LANGUAGE", "Português"),
        allow_direct_answers=os.getenv("ALLOW_DIRECT_ANSWERS", "False").lower() == "true"
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
    model_names = {
        "generation": os.getenv("GENERATION_MODEL", "gpt-4o-mini"),
        "tracking": os.getenv("TRACKING_MODEL", "gpt-4.1-nano"),
        "planning": os.getenv("PLANNING_MODEL", "gpt-4.1-mini"),
        "grading": os.getenv("GRADING_MODEL", "gpt-4.1-nano")
    }

    logger.info("Using models: %s", model_names)

    models = ModelRegistry(
        generation_llm=ChatOpenAI(
            model=model_names["generation"],
            temperature=model_temperature
        ),

        tracking_llm=ChatOpenAI(
            model=model_names["tracking"],
            temperature=model_temperature
        ),

        planning_llm=ChatOpenAI(
            model=model_names["planning"],
            temperature=model_temperature
        ),

        grading_llm=ChatOpenAI(
            model=model_names["grading"],
            temperature=model_temperature
        )
    )

    # --------------------------------------------------
    # Build Graph
    # --------------------------------------------------
    graph = build_graph(config, retriever, models)

    logger.info("Pipeline loaded successfully for discipline: %s", discipline_name)

    return graph, tutor_prompt, config