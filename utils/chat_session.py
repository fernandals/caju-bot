"""
Pure helpers for the student chat session (chainlit_app.py), kept here so
they can be unit-tested without importing the app (which loads the
knowledge base at import time).
"""

import json
import logging

from agent.graph import is_smalltalk
from agent.state import LearningState, StudentProfile, TeachingState
from utils.quick_replies import ACTION_NAME, STYLE_ACTION_NAME

logger = logging.getLogger(__name__)

# Key of the tutor's state in the Chainlit thread metadata (see
# chainlit_app._persist_turn_metadata).
TUTOR_STATE_KEY = "tutor_state"

# Placeholder thread name set when a conversation starts with the style
# question click (chainlit_app.on_style_choice), until the generated title.
NEW_THREAD_NAME = "Nova conversa"

# Same, for a conversation whose first turn only listed the course topics
# (the TOPICS button or the same question typed - chainlit_app.on_message).
TOPICS_THREAD_NAME = "Conteúdos da disciplina"


def truncate_for_edit(messages: list, message_id: str) -> bool:
    """
    Chainlit's edit_message re-sends an EARLIER message under its original
    id (it has already removed the later messages from the UI and the
    database). Cut `messages` back to just before it, in place, so the
    edited question replaces the old one instead of being appended after
    the exchange it was meant to replace. Returns True if this was an edit.
    """
    for index, msg in enumerate(messages):
        if msg.id == message_id:
            del messages[index:]
            return True
    return False


def needs_title(name: str | None) -> bool:
    """
    Chainlit names a thread after its first interaction: the raw first
    message, or the action name if it began with a button click (a quick
    reply, or the first-conversation style question). Those
    (and a bare greeting) are worth replacing with a generated title.
    """
    return (
        not name
        or name in (ACTION_NAME, STYLE_ACTION_NAME, NEW_THREAD_NAME, TOPICS_THREAD_NAME)
        or is_smalltalk(name)
    )


def tutor_state_snapshot(final_state) -> dict | None:
    """
    JSON-able copy of what the tutor knows about where this conversation
    stands - learning state and teaching arc - to restore on resume.
    """
    learning_state = final_state.get("learning_state")
    teaching_state = final_state.get("teaching_state")

    if learning_state is None or teaching_state is None:
        return None

    return {
        "learning_state": learning_state.model_dump(mode="json"),
        "teaching_state": teaching_state.model_dump(mode="json"),
    }


def restore_tutor_state(metadata) -> dict:
    """
    TutorState keys to resume a conversation where it stopped, from the
    thread metadata ({} if nothing usable was saved - older threads).

    Frustration is reset: a resume is a new sitting, and yesterday's
    impatience must not trigger an early full explanation today.
    """
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except ValueError:
            return {}

    saved = (metadata or {}).get(TUTOR_STATE_KEY) if isinstance(metadata, dict) else None

    if not isinstance(saved, dict):
        return {}

    try:
        learning_state = LearningState.model_validate(saved["learning_state"])
        teaching_state = TeachingState.model_validate(saved["teaching_state"])
    except Exception:  # noqa: BLE001 - schema drift: resume fresh instead
        logger.warning("Saved tutor state unreadable; resuming fresh", exc_info=True)
        return {}

    return {
        "learning_state": learning_state.model_copy(update={"frustration_level": 0.0}),
        "teaching_state": teaching_state,
    }


def should_ask_style(profile: StudentProfile) -> bool:
    """Ask how they like explanations only in a student's first conversations,
    until they answer or skip."""
    return (
        profile.sessions_observed == 0
        and profile.stated_style == "unknown"
        and not profile.style_asked
    )
