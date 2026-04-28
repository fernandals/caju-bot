from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
import os

from rag.knowledge_base import load_knowledge_base
from agent.tools import build_retrieve_tool
from agent.graph import build_graph
import agent.prompts as prompts
from agent.state import TutorConfig


def load_pipeline(discipline_name: str):
    retriever = load_knowledge_base(discipline_name)
    retrieve_tool = build_retrieve_tool(retriever)

    config = TutorConfig(subject=discipline_name)

    tutor_prompt = SystemMessage(content=prompts.SYSTEM_PROMPT.format(
        domain=config.subject,
        max_sentences=config.max_sentences,
        course_level=config.course_level,
        answer_language=config.answer_language
    ))

    response_model = init_chat_model(
        os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=float(os.getenv("MODEL_TEMPERATURE", 0))
    ).bind_tools([retrieve_tool])

    graph = build_graph(config, retrieve_tool, response_model)

    return graph, tutor_prompt, config