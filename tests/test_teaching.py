"""
The teaching arc (agent/teaching.py): when the tutor guides with a
question, when it explains, and how the profile and the session bend
that. Deterministic - this decides what a student experiences most
directly, whether they get guided or just told.
"""

import pytest

from langchain_core.messages import AIMessage, HumanMessage

from agent.nodes import apply_verdict, judge_reply, resolve_teaching_instructions
from agent.prompts import ATTEMPT_REVIEW_NOTE, CLOSE_TOPIC_INSTRUCTIONS, DIRECT_MODE_INSTRUCTIONS, VERDICT_NOTES
from agent.state import LearningState, ReplyVerdict, StudentProfile, StyleTracker, TeachingState
from agent.teaching import (
    GUIDED_TURN_CAP,
    GUIDED_TURN_CAP_SHAKY,
    MIN_GUIDED_ATTEMPTS,
    advance_teaching_state,
    close_arc,
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


# (previous state, what tracking saw, profile) -> (mode, stage, turns_in_stage, may_close)
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
    "refer, then mastered": (ts("refer", 0), ls(learning_progress="mastered"), None, ("guided", "check", 0, True)),
    "refer, then engaged": (ts("refer", 0), ls(learning_progress="improving"), None, (None, "check", None)),
    "refer, then stable": (ts("refer", 0), ls(learning_progress="stable"), None, (None, "deepen", None)),
    "refer, then stuck": (ts("refer", 0), ls(learning_progress="stuck"), None, (None, "deepen", None)),
    # The guided loop. Whether a right reply closes the topic is the
    # tutor's verdict (apply_verdict); pacing only says whether it may.
    "right on the easy first question: one more": (ts("introduce", 0), ls(learning_progress="mastered"), None, (None, "check", 1, False)),
    "second reply may close": (ts(), ls(learning_progress="mastered"), None, (None, "check", 2, True)),
    "stuck on the first reply: one more try": (ts("introduce", 0), ls(learning_progress="stuck"), None, (None, "check", 1)),
    "stuck after a nudge: explain": (ts(), ls(learning_progress="stuck"), None, (None, "deepen", None)),
    "still engaged: keep asking": (ts(), ls(learning_progress="stable"), None, (None, "check", 2)),
    "turn cap reached: explain": (ts("check", GUIDED_TURN_CAP - 1), ls(learning_progress="stable"), None, (None, "deepen", None)),
    "turn cap reached but getting it: may close": (
        ts("check", GUIDED_TURN_CAP - 1), ls(learning_progress="mastered"), None, (None, "check", None, True)
    ),
    "explained: recap and check": (ts("deepen", 0), ls(learning_progress="stable"), None, ("guided", "wrap_up", None)),
    # The answer to the recap's check question - not a fresh cycle.
    "answer after the recap may close": (ts("wrap_up", 0), ls(learning_progress="stable"), None, ("guided", "check", 0, True)),
    "explained arc: first reply may close": (ts("introduce", 0, explained=True), ls(), None, (None, "check", 1, True)),
    # Exam prep explains once per topic, then judges answers like any other.
    "exam prep, already explained": (ts("wrap_up", 0, explained=True), ls(intent="exam_prep"), None, ("guided", "check", 0, True)),
    "exam prep marks the arc explained": (ts("introduce", 0), ls(intent="exam_prep"), None, ("direct", "deepen", None)),
    # A concluded topic is not reopened while it stays the same topic.
    "concluded stays concluded": (ts("concluded", 0), ls(learning_progress="stable"), None, ("guided", "concluded", 1, False)),
    "concluded, exam talk": (ts("concluded", 0, explained=True), ls(intent="exam_prep"), None, ("guided", "concluded", None)),
    "concluded, then a new subtopic": (ts("concluded", 0), ls(subtopic="while-loop"), None, ("guided", "introduce", 0)),
}


@pytest.mark.parametrize("previous, learning, profile, expected", TRANSITIONS.values(), ids=TRANSITIONS.keys())
def test_transitions(previous, learning, profile, expected):
    result = advance_teaching_state(previous, learning, profile)

    for field, value in zip(("mode", "stage", "turns_in_stage", "may_close"), expected):
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


def test_solid_topic_may_close_on_the_first_reply():
    def may_close_after_first_reply(familiarity):
        return advance_teaching_state(ts("introduce", 0), ls(learning_progress="mastered"), StudentProfile(**CONFIDENT), familiarity).may_close

    assert may_close_after_first_reply("solid")
    assert not may_close_after_first_reply("new")


def test_session_counts_survive_topic_changes_and_shorten_a_struggling_session():
    stuck = advance_teaching_state(ts(), ls(learning_progress="stuck"))
    assert (stuck.stage, stuck.session_arcs_stuck) == ("deepen", 1)
    assert advance_teaching_state(stuck, ls(topic="other")).session_arcs_stuck == 1

    mastered = close_arc(ts())
    assert (mastered.stage, mastered.session_arcs_mastered) == ("concluded", 1)
    assert advance_teaching_state(mastered, ls(topic="other")).session_arcs_mastered == 1

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


def test_exercise_on_a_concluded_topic_is_judged():
    result = continue_exercise(ts("concluded", 0, exercise_pending=True), ts("concluded", 1))
    assert (result.stage, result.may_close, result.reviewing_attempt) == ("check", True, True)


def test_exercise_attempt_may_close():
    proposed = ts("introduce", 0, topic_anchor=("árvores", "rotações"))
    assert continue_exercise(ts(topic_anchor=(None, None), exercise_pending=True), proposed).may_close
    # A task inside the guided arc keeps the arc's own rule.
    assert not continue_exercise(ts("introduce", 0, exercise_pending=True), ts("check", 1, may_close=False)).may_close


# ---------------- the tutor's verdict ----------------


def test_a_correct_reply_closes_only_when_allowed():
    closing = resolve_teaching_instructions(ts(may_close=True), "guided_teaching", course_outline=["Heap"], verdict="correct")
    assert closing.startswith(CLOSE_TOPIC_INSTRUCTIONS) and "- Heap" in closing

    # Right, but too early to close: confirmed, and the arc goes on.
    early = resolve_teaching_instructions(ts(may_close=False), "guided_teaching", verdict="correct")
    assert CLOSE_TOPIC_INSTRUCTIONS not in early and VERDICT_NOTES["correct"] in early

    wrong = resolve_teaching_instructions(ts(may_close=True), "guided_teaching", verdict="wrong")
    assert VERDICT_NOTES["wrong"] in wrong and CLOSE_TOPIC_INSTRUCTIONS not in wrong

    # Only guided "check" turns are judged.
    assert VERDICT_NOTES["correct"] not in resolve_teaching_instructions(ts("deepen", 0), "step_by_step", verdict="correct")


def turn_state(may_close=True, progress="mastered", stage="check"):
    return {
        "teaching_state": ts(stage, 2, may_close=may_close, exercise_pending=True),
        "learning_state": ls(learning_progress=progress),
        "style_tracker": StyleTracker(active="examples", stuck_streak=2),
    }


def test_correct_reply_closes_the_topic():
    updates = apply_verdict(turn_state(), "correct")

    assert updates["teaching_state"].stage == "concluded"
    assert updates["teaching_state"].session_arcs_mastered == 1
    assert not updates["teaching_state"].exercise_pending
    assert updates["learning_state"].learning_progress == "mastered"
    # A mastered arc for the style in use: the stuck streak resets.
    assert updates["style_tracker"].stuck_streak == 0
    assert updates["style_tracker"].record["examples"].mastered == 1


@pytest.mark.parametrize(
    "may_close, tracker_said, verdict, progress",
    [
        # The tracker's "mastered" on a wrong answer yields to the tutor.
        (True, "mastered", "wrong", "stable"),
        (True, "mastered", "partial", "improving"),
        # Right, but too early to close: progress, not mastery.
        (False, "mastered", "correct", "improving"),
        (True, "stuck", "wrong", "stuck"),
    ],
)
def test_verdict_corrects_the_tracker(may_close, tracker_said, verdict, progress):
    updates = apply_verdict(turn_state(may_close, tracker_said), verdict)

    assert "teaching_state" not in updates
    assert updates["learning_state"].learning_progress == progress


def test_no_verdict_changes_nothing():
    assert apply_verdict(turn_state(), None) == {}
    assert apply_verdict(turn_state(), "none") == {}
    assert apply_verdict(turn_state(stage="deepen"), "correct") == {}


class Judge:
    def __init__(self, verdict="correct", fail=False):
        self.verdict, self.fail, self.prompt = verdict, fail, None

    def with_structured_output(self, _):
        return self

    def invoke(self, prompt):
        if self.fail:
            raise TimeoutError("slow")
        self.prompt = prompt
        return ReplyVerdict(reason="r", verdict=self.verdict)


def judged_state(stage="check", mode="guided"):
    return {
        "teaching_state": ts(stage, 1, mode=mode),
        "messages": [AIMessage(content="Qual a altura de uma AVL?"), HumanMessage(content="O(log n)")],
    }


def test_judge_reads_the_question_the_reply_and_the_material():
    judge = Judge("partial")
    assert judge_reply(judged_state(), "EVIDENCE - h = O(log n)", judge) == "partial"
    assert "Qual a altura de uma AVL?" in judge.prompt and "O(log n)" in judge.prompt
    assert "EVIDENCE - h = O(log n)" in judge.prompt


def test_judge_only_on_guided_check_and_never_breaks_the_turn():
    assert judge_reply(judged_state("deepen"), "", Judge()) is None
    assert judge_reply(judged_state(mode="direct"), "", Judge()) is None
    assert judge_reply(judged_state(), "", Judge(fail=True)) is None
