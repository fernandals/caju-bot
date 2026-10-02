"""
Pure helpers for the student chat session (chainlit_app.py), kept here so
they can be unit-tested without importing the app (which loads the
knowledge base at import time).
"""

from agent.graph import is_smalltalk
from utils.quick_replies import ACTION_NAME


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
    message, or the action name if it began with a button click. Those
    (and a bare greeting) are worth replacing with a generated title.
    """
    return not name or name == ACTION_NAME or is_smalltalk(name)
