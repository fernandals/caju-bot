"""
Short conversation titles for the chat-history sidebar.

Chainlit names a thread after the raw first message, which gives titles
like "olá!" or a whole run-on sentence. Once per thread - after the first
real (non-greeting) turn - a small model turns the student's question
into a 3-6 word title. Best effort: any failure keeps Chainlit's default.
"""

import functools
import logging
import os
import re

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

MAX_TITLE_CHARS = 60

TITLE_PROMPT = """
Crie um título curto para uma conversa de estudo, a partir da primeira
pergunta do aluno abaixo.

Regras:
- 3 a 6 palavras, em {language}.
- Nomeie o ASSUNTO (ex.: "Estilo Pipe-Filter", "Cliente-Servidor:
  escalabilidade", "Revisão para a prova"), não a ação do aluno
  ("Aluno pergunta sobre...").
- Sem aspas, sem ponto final, sem emoji.
- Responda só com o título.

Assunto identificado pelo sistema (pode ajudar): {topic}

Pergunta do aluno:
{question}
"""


@functools.lru_cache(maxsize=1)
def _title_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("TITLE_MODEL", "gpt-4.1-nano"),
        temperature=0,
        timeout=float(os.getenv("MODEL_TIMEOUT", 30)),
        max_retries=int(os.getenv("MODEL_MAX_RETRIES", 2)),
    )


def clean_title(raw: str | None) -> str | None:
    """Normalizes a model-written title; None if nothing usable is left."""
    if not raw:
        return None

    title = raw.strip().splitlines()[0] if raw.strip() else ""
    title = re.sub(r"^(t[íi]tulo|title)\s*:\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).strip(" .!\t\"'“”‘’`*#")

    if not title:
        return None

    if len(title) > MAX_TITLE_CHARS:
        cut = title[:MAX_TITLE_CHARS].rsplit(" ", 1)[0]
        title = cut.rstrip(" ,;:-") + "…"

    return title


def generate_title(question: str, topic: str | None, language: str = "Português") -> str | None:
    """Blocking LLM call - run it off the event loop. Never raises."""
    try:
        raw = _title_model().invoke(
            TITLE_PROMPT.format(
                language=language,
                topic=topic or "(não identificado)",
                question=question.strip()[:1000],
            )
        ).content
    except Exception:  # noqa: BLE001 - a title is cosmetic
        logger.warning("Title generation failed", exc_info=True)
        return None

    return clean_title(raw if isinstance(raw, str) else str(raw))
