"""
The teaching arc (agent/teaching.py): when the tutor guides with a
question, when it explains, and how the profile and the session bend
that. Deterministic - this decides what a student experiences most
directly, whether they get guided or just told.
"""

import pytest

from agent.nodes import resolve_teaching_instructions
from agent.prompts import ATTEMPT_REVIEW_NOTE, DIRECT_MODE_INSTRUCTIONS
from agent.state import LearningState, StudentProfile, TeachingState
from agent.teaching import (
    GUIDED_TURN_CAP,
    GUIDED_TURN_CAP_SHAKY,
    MIN_GUIDED_ATTEMPTS,
    advance_teaching_state,
    continue_exercise,
    guided_turn_cap,
    is_course_review,
)

ANCHOR = ("loops", "for-loop")
CONFIDENT = dict(confidence=0.8)
SENSITIVE = StudentProfile(frustration_tendency="high", **CONFIDENT)
POOR_GUIDING = StudentProfile(responds_to_guiding_questions="poorly", **CONFIDENT)


def ts(stage="check", turns=1, **kwargs) -> TeachingState:
    return TeachingState(**{"topic_anchor": ANCHOR, "stage": stage, "turns_in_stage": turns, **kwargs})


def ls(**kwargs) -> LearningState:
    return LearningState(**{"topic": ANCHOR[0], "subtopic": ANCHOR[1], **kwargs})


# (previous state, what tracking saw, profile) -> (mode, stage, turns_in_stage)
TRANSITIONS = {
    # A new topic or subtopic restarts the arc with a guiding question.
    "new topic": (ts("deepen", 3), ls(topic="recursion", subtopic=None), None, ("guided", "introduce", 0)),
    "new subtopic": (ts("check", 2), ls(subtopic="while-loop"), None, ("guided", "introduce", 0)),
    "first turn ever": (TeachingState(), ls(), None, ("guided", "introduce", 0)),
    # Escape valves: frustration or exam prep go straight to the explanation.
    "frustrated": (ts(), ls(frustration_level=0.9), None, ("direct", "deepen", None)),
    "just below threshold": (ts(), ls(frustration_level=0.59), None, ("guided", None, None)),
    "sensitive student escapes earlier": (ts(), ls(frustration_level=0.5), SENSITIVE, ("direct", "deepen", None)),
    "unconfident profile ignored": (
        ts(), ls(frustration_level=0.5), StudentProfile(frustration_tendency="high", confidence=0.1), ("guided", None, None)
    ),
    "exam prep": (ts("introduce", 0), ls(intent="exam_prep"), None, ("direct", "deepen", None)),
    "exam prep on a new topic": (ts(), ls(subtopic="while-loop", intent="exam_prep"), None, ("direct", "deepen", None)),
    "frustration carried to a new topic": (ts(), ls(topic="recursion", frustration_level=0.8), None, ("direct", None, None)),
    "practice stays guided": (ts("introduce", 0), ls(intent="practice"), None, ("guided", None, None)),
    # After pointing at the material ("refer").
    "refer, then mastered": (ts("refer", 0), ls(learning_progress="mastered"), None, ("guided", "wrap_up", None)),
    "refer, then engaged": (ts("refer", 0), ls(learning_progress="improving"), None, (None, "check", None)),
    "refer, then stable": (ts("refer", 0), ls(learning_progress="stable"), None, (None, "deepen", None)),
    "refer, then stuck": (ts("refer", 0), ls(learning_progress="stuck"), None, (None, "deepen", None)),
    # The guided loop.
    "right on the easy first question: one more": (ts("introduce", 0), ls(learning_progress="mastered"), None, (None, "check", 1)),
    "right after a nudge: wrap up": (ts(), ls(learning_progress="mastered"), None, (None, "wrap_up", None)),
    "stuck on the first reply: one more try": (ts("introduce", 0), ls(learning_progress="stuck"), None, (None, "check", 1)),
    "stuck after a nudge: explain": (ts(), ls(learning_progress="stuck"), None, (None, "deepen", None)),
    "still engaged: keep asking": (ts(), ls(learning_progress="stable"), None, (None, "check", 2)),
    "turn cap reached: explain": (ts("check", GUIDED_TURN_CAP - 1), ls(learning_progress="stable"), None, (None, "deepen", None)),
    "explained: wrap up": (ts("deepen", 0), ls(learning_progress="stable"), None, ("guided", "wrap_up", None)),
    "wrapped up, still engaged: new cycle": (ts("wrap_up", 0), ls(learning_progress="stable"), None, (None, "introduce", 0)),
}


@pytest.mark.parametrize("previous, learning, profile, expected", TRANSITIONS.values(), ids=TRANSITIONS.keys())
def test_transitions(previous, learning, profile, expected):
    result = advance_teaching_state(previous, learning, profile)

    for field, value in zip(("mode", "stage", "turns_in_stage"), expected):
        if value is not None:
            assert getattr(result, field) == value


def replies_until_explained(profile=None, familiarity="new", progress="improving", **start):
    """Stages after each student reply, from the introduce turn on."""
    state = ts("introduce", 0, **start)
    stages = []
    for _ in range(8):
        state = advance_teaching_state(state, ls(learning_progress=progress), profile, familiarity)
        stages.append(state.stage)
        if state.stage != "check":
            break
    return stages


@pytest.mark.parametrize(
    "profile, familiarity, replies",
    [
        (None, "new", GUIDED_TURN_CAP),
        (StudentProfile(**CONFIDENT), "shaky", GUIDED_TURN_CAP_SHAKY),
        (POOR_GUIDING, "new", MIN_GUIDED_ATTEMPTS),
        # Below the confidence floor the profile changes nothing.
        (StudentProfile(responds_to_guiding_questions="poorly", confidence=0.2), "shaky", GUIDED_TURN_CAP),
    ],
)
def test_profile_sets_how_long_the_tutor_guides(profile, familiarity, replies):
    stages = replies_until_explained(profile, familiarity)
    assert len(stages) == replies and stages[-1] == "deepen"


def test_guiding_never_ends_before_the_minimum():
    assert guided_turn_cap(POOR_GUIDING, "shaky") == MIN_GUIDED_ATTEMPTS == 2
    assert replies_until_explained(POOR_GUIDING, "shaky", progress="stuck") == ["check", "deepen"]


def test_solid_topic_accepts_mastery_on_the_first_reply():
    assert replies_until_explained(StudentProfile(**CONFIDENT), "solid", "mastered") == ["wrap_up"]
    assert replies_until_explained(StudentProfile(**CONFIDENT), "new", "mastered") == ["check", "wrap_up"]


def test_session_counts_survive_topic_changes_and_shorten_a_struggling_session():
    stuck = advance_teaching_state(ts(), ls(learning_progress="stuck"))
    assert (stuck.stage, stuck.session_arcs_stuck) == ("deepen", 1)
    assert advance_teaching_state(stuck, ls(topic="other")).session_arcs_stuck == 1

    mastered = advance_teaching_state(ts(), ls(learning_progress="mastered"))
    assert mastered.session_arcs_mastered == 1

    # Two stuck arcs, none mastered: the rest of the session guides briefly
    # - unless in-session adaptation is off, or guiding did work once.
    def replies(adapt=True, **counts):
        state, n = ts("introduce", 0, **counts), 0
        while True:
            state = advance_teaching_state(state, ls(learning_progress="improving"), None, adapt_in_session=adapt)
            n += 1
            if state.stage != "check":
                return n

    assert replies(session_arcs_stuck=2) == MIN_GUIDED_ATTEMPTS
    assert replies(session_arcs_stuck=2, adapt=False) == GUIDED_TURN_CAP
    assert replies(session_arcs_stuck=2, session_arcs_mastered=1) == GUIDED_TURN_CAP


# ---------------- course review ----------------


@pytest.mark.parametrize(
    "learning, review",
    [
        (ls(intent="exam_prep", topic=None), True),
        (ls(intent="exam_prep", topic="Árvores AVL"), False),
        (ls(intent="review", topic=None), False),
        (None, False),
    ],
)
def test_course_review_is_exam_prep_without_a_topic(learning, review):
    assert is_course_review(learning) is review


def test_course_review_instructions():
    outline = ["Complexidade assintótica", "Árvores AVL"]
    review = resolve_teaching_instructions(ts(mode="direct"), "direct_answer", course_review=True, course_outline=outline)

    assert review.startswith("COURSE REVIEW")
    assert "1. Complexidade assintótica\n2. Árvores AVL" in review
    # No outline: the usual direct explanation.
    assert resolve_teaching_instructions(ts(mode="direct"), "direct_answer", course_review=True) == DIRECT_MODE_INSTRUCTIONS


# ---------------- exercise attempts ----------------


@pytest.mark.parametrize(
    "previous_anchor, proposed, stage",
    [
        # "me passa um exercício" names no topic; the attempt does.
        ((None, None), ts("introduce", 0, topic_anchor=("árvores", "rotações")), "check"),
        (("árvores", None), ts("introduce", 0, topic_anchor=("árvores", "AVL")), "check"),
        # A different topic is still a topic change (but may be the attempt).
        (("árvores", None), ts("introduce", 0, topic_anchor=("heap", None)), "introduce"),
        (ANCHOR, ts("deepen", 0, mode="direct"), "deepen"),
    ],
)
def test_reply_after_an_exercise_reviews_the_attempt(previous_anchor, proposed, stage):
    result = continue_exercise(ts(topic_anchor=previous_anchor, exercise_pending=True), proposed)

    assert result.stage == stage
    assert result.mode == proposed.mode
    assert result.reviewing_attempt


def test_no_exercise_pending_changes_nothing():
    proposed = ts("introduce", 0)
    assert continue_exercise(ts(), proposed) == proposed


def test_attempt_review_note_comes_first():
    assert resolve_teaching_instructions(ts(), "guided_teaching", reviewing_attempt=True).startswith(ATTEMPT_REVIEW_NOTE)
