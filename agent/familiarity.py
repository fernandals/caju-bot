"""
Is the topic being discussed one the student's profile already knows
about? Deterministic, no LLM call.

StudentProfile.solid_topics / shaky_topics are free text written by the
profiler, and LearningState.topic / subtopic are free text written by the
tracker - two models, two vocabularies. Real data mixes languages and
spellings for the same thing ("client-server", "cliente-servidor",
"Client-Server architecture", "Estilo Client-Server") and pads topics with
the course name ("Software Architecture | pipe-filter style"). So topics
are compared as sets of CORE words:

- lowercase, accents stripped, split on anything not a letter/digit;
- drop stopwords (pt/en), words generic to any course ("style",
  "basics", ...) and words of the course name itself;
- two words match if equal, if they share a long enough prefix
  (cliente/client, servidor/server, filtro/filter) or are close spellings
  (arquitetura/architecture).

A profile topic matches when all its core words are found in the current
topic, or the other way round. Topics left with no core words ("software
architecture basics" in a Software Architecture course) never match - too
generic to say anything.

Known limit: words that share a prefix but not a meaning can collide
(service/server); unrelated translations never match (blackboard/quadro).
"""

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Literal

from agent.state import LearningState, StudentProfile

Familiarity = Literal["solid", "shaky", "new"]

_STOPWORDS = {
    # pt
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "e", "em", "na",
    "no", "nas", "nos", "um", "uma", "para", "com", "entre", "por", "sobre",
    # en
    "the", "of", "and", "in", "on", "for", "to", "with", "between", "an",
    "vs", "x",
}

_GENERIC = {
    "style", "styles", "estilo", "estilos", "basics", "basic", "basico",
    "basicos", "introduction", "introducao", "intro", "overview", "visao",
    "geral", "concept", "concepts", "conceito", "conceitos", "fundamentals",
    "fundamentos", "definition", "definicao", "topic", "topico",
}

_MIN_PREFIX = 4
_MIN_RATIO = 0.7


def _words(text: str | None) -> list[str]:
    if not text:
        return []
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]


def _same_word(a: str, b: str) -> bool:
    if a == b:
        return True

    shorter = min(len(a), len(b))
    if shorter < _MIN_PREFIX:
        return False

    prefix = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        prefix += 1

    if prefix >= _MIN_PREFIX and prefix >= 0.6 * shorter:
        return True

    return shorter >= 6 and SequenceMatcher(None, a, b).ratio() >= _MIN_RATIO


def _contains(words: set[str], other: set[str]) -> bool:
    """Every word of `words` has a match in `other`."""
    return all(any(_same_word(w, o) for o in other) for w in words)


def core_words(text: str | None, subject: str = "") -> set[str]:
    subject_words = set(_words(subject))
    return {
        w
        for w in _words(text)
        if w not in _STOPWORDS
        and w not in _GENERIC
        and not w.isdigit()
        and not any(_same_word(w, s) for s in subject_words)
    }


def shares_word(a: set[str], b: set[str]) -> bool:
    """At least one core word in common - looser than topics_match, for
    matching a free-text note ("client-server: thinks the server starts
    the communication") against the current topic."""
    return any(_same_word(x, y) for x in a for y in b)


def topics_match(a: set[str], b: set[str]) -> bool:
    return bool(a) and bool(b) and (_contains(a, b) or _contains(b, a))


def topic_familiarity(
    learning_state: LearningState,
    profile: StudentProfile | None,
    subject: str = "",
) -> Familiarity:
    """
    "shaky" / "solid" if the current topic matches one of the profile's
    lists, else "new". Shaky wins if it matches both - erring towards more
    support. The confidence gate is the caller's (agent/teaching.py).
    """
    if profile is None:
        return "new"

    current = core_words(
        f"{learning_state.topic or ''} {learning_state.subtopic or ''}", subject
    )

    if not current:
        return "new"

    for familiarity, topics in (("shaky", profile.shaky_topics), ("solid", profile.solid_topics)):
        if any(topics_match(current, core_words(t, subject)) for t in topics):
            return familiarity  # type: ignore[return-value]

    return "new"
