"""
Session-boundary student profiler.

Runs once per conversation (not per turn): given the student's stored
StudentProfile and the transcript of their most recent conversation, it
returns an updated profile. Wired in via the on_chat_start catch-up in
chainlit_app.py - see utils/student_profile.py for persistence.
"""

import logging
import os
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI

import agent.prompts as prompts
from agent.state import StudentProfile

logger = logging.getLogger(__name__)


def build_profiler_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("PROFILER_MODEL", "gpt-4.1-mini"),
        temperature=float(os.getenv("MODEL_TEMPERATURE", 0)),
        timeout=float(os.getenv("MODEL_TIMEOUT", 30)),
        max_retries=int(os.getenv("MODEL_MAX_RETRIES", 2)),
    )


def _confidence_for(sessions_observed: int) -> float:
    # Grow steadily, cap below 1.0 - the profile is always a bit provisional.
    return round(min(0.9, 0.25 * sessions_observed), 2)


def profile_student(
    previous: StudentProfile,
    transcript: str,
    model=None,
) -> StudentProfile:
    """
    Merge one conversation transcript into `previous`. Returns `previous`
    unchanged on any failure - profiling must never break a session.

    sessions_observed / confidence / last_updated are set here, not by the
    model.
    """
    model = model or build_profiler_model()

    try:
        structured = model.with_structured_output(StudentProfile)

        updated = structured.invoke(
            f"{prompts.PROFILER_PROMPT}\n\n"
            "---\n\n"
            f"Current profile:\n{previous.model_dump_json(indent=2)}\n\n"
            "---\n\n"
            f"Most recent conversation:\n{transcript}\n"
        )

        if updated is None:
            logger.warning("Profiler returned no structured output; keeping previous")
            return previous

        updated.sessions_observed = previous.sessions_observed + 1
        updated.confidence = _confidence_for(updated.sessions_observed)
        updated.last_updated = datetime.now(timezone.utc).isoformat()

        return updated
    except Exception:  # noqa: BLE001 - never break a session over profiling
        logger.warning("Profiler failed; keeping previous profile", exc_info=True)
        return previous
