"""
What the profile knows about the topic being discussed RIGHT NOW, picked
in code (no LLM call) and handed to the generator in the end-of-context
turn reminder (agent/nodes.generate_answer) - the spot the model follows
most reliably - instead of buried in the long system prompt.

- misconceptions the student voiced before on this topic -> check whether
  the belief is still there;
- what moved them forward before on this topic -> prefer that again;
- on explanation turns, an analogy bridge from what they know outside the
  course (prior_knowledge) - the one exception to "examples and analogies
  only from the material", allowed only as an explicitly framed analogy,
  never as a fact about the subject.

Notes are matched to the topic by shared core words (agent/familiarity.py).
Nothing matches -> nothing is added.
"""

from agent.familiarity import core_words, shares_word
from agent.plan_profile import is_explanation_turn
from agent.state import LearningState, StudentProfile, TeachingState
from agent.teaching import profile_confident

MAX_NOTES_PER_KIND = 2


def _about_topic(notes: list[str], topic_words: set[str], subject: str) -> list[str]:
    return [n for n in notes if shares_word(core_words(n, subject), topic_words)][:MAX_NOTES_PER_KIND]


def select_profile_notes(
    profile: StudentProfile | None,
    learning_state: LearningState,
    teaching_state: TeachingState,
    subject: str = "",
) -> tuple[list[str], list[str]]:
    """(notes for the generator, levers for the study log)."""
    if not profile_confident(profile):
        return [], []

    topic_words = core_words(
        f"{learning_state.topic or ''} {learning_state.subtopic or ''}", subject
    )

    notes: list[str] = []
    levers: list[str] = []

    if topic_words:
        for belief in _about_topic(profile.misconceptions, topic_words, subject):  # type: ignore[union-attr]
            notes.append(
                f'Earlier this student believed: "{belief}". Shape this '
                "reply so it shows whether that belief is still there, and "
                "correct it (grounded in the material) if it is."
            )
        if any(n.startswith("Earlier") for n in notes):
            levers.append("known_misconception")

        clicked = _about_topic(profile.what_clicked, topic_words, subject)  # type: ignore[union-attr]
        for worked in clicked:
            notes.append(
                f'What helped this student before on this topic: "{worked}". '
                "Prefer that kind of explanation or question again if it "
                "fits this turn (facts and examples still only from the "
                "material)."
            )
        if clicked:
            levers.append("what_clicked")

    if is_explanation_turn(teaching_state) and profile.prior_knowledge:  # type: ignore[union-attr]
        known = "; ".join(profile.prior_knowledge[:3])  # type: ignore[union-attr]
        notes.append(
            "ANALOGY EXCEPTION (overrides the grounding rule for analogies "
            "only): you may add ONE short analogy to something this student "
            f"already knows ({known}), framed explicitly as an analogy "
            '("é parecido com...", "pense em..."), with no citation marker '
            "on it. It must never be presented as a fact about the subject, "
            "and every factual claim still comes from the retrieved "
            "evidence. Skip it if no natural analogy fits."
        )
        levers.append("prior_knowledge_analogy")

    return notes, levers
