"""
Earlier and deeper adaptation (PROJECT_LOG 2026-10-07): the stated style
and faster confidence (start personalizing sooner), in-session adaptation,
what the profile knows about the current topic (misconceptions, what
worked, analogy bridge), and resuming a conversation where it stopped.
No LLM calls.
"""

import json
import sqlite3

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.nodes import generate_answer
from agent.plan_profile import apply_profile_to_plan
from agent.profile_notes import select_profile_notes
from agent.profiler import _confidence_for, profile_student, worth_profiling
from agent.state import AnswerPlan, LearningState, StudentProfile, TeachingState, TutorConfig
from agent.teaching import (
    MIN_GUIDED_ATTEMPTS,
    PROFILE_CONFIDENCE_FLOOR,
    advance_teaching_state,
    effective_style,
    personalization_trace,
)
from utils.chat_session import (
    TUTOR_STATE_KEY,
    restore_tutor_state,
    should_ask_style,
    tutor_state_snapshot,
)
from utils.metrics import ensure_metrics_schema, record_turn, thread_trajectory
from utils.student_profile import scope_to_course

SUBJECT = "Software Architecture"

# ---------------- 3b: confidence ----------------


def test_personalization_starts_in_second_conversation():
    assert _confidence_for(0) == 0.0
    assert _confidence_for(1) == PROFILE_CONFIDENCE_FLOOR
    assert [_confidence_for(n) for n in (2, 3, 4, 10)] == [0.6, 0.8, 0.9, 0.9]


def test_short_conversations_are_not_observations():
    assert not worth_profiling("Student: oi\nTutor: Olá!\nStudent: tchau")
    assert worth_profiling("Student: a\nTutor: b\nStudent: c\nTutor: d\nStudent: e")


class _CapturingModel:
    def __init__(self, returns):
        self.returns, self.prompt = returns, None

    def with_structured_output(self, _):
        return self

    def invoke(self, prompt):
        self.prompt = prompt
        return self.returns


def test_profiler_keeps_system_fields_and_sees_trajectory():
    previous = StudentProfile(course="X", stated_style="concise", style_asked=True)
    model = _CapturingModel(StudentProfile(course="Y", stated_style="detailed", style_asked=False))

    updated = profile_student(previous, "Student: oi", model=model, trajectory="turn 1: stage=check")

    assert (updated.course, updated.stated_style, updated.style_asked) == ("X", "concise", True)
    assert "Per-turn signals:\nturn 1: stage=check" in model.prompt


# ---------------- 3a: stated style ----------------


def test_should_ask_style():
    assert should_ask_style(StudentProfile())
    assert not should_ask_style(StudentProfile(style_asked=True))
    assert not should_ask_style(StudentProfile(stated_style="concise"))
    assert not should_ask_style(StudentProfile(sessions_observed=1))


def test_stated_style_applies_before_confidence_observed_wins_after():
    stated = StudentProfile(stated_style="example_first")
    assert effective_style(stated) == ("example_first", "stated")

    observed = stated.model_copy(update={"explanation_style": "concise", "confidence": 0.6})
    assert effective_style(observed) == ("concise", "observed")

    # Observed but not confident yet: the student's own answer still holds.
    early = stated.model_copy(update={"explanation_style": "concise", "confidence": 0.2})
    assert effective_style(early) == ("example_first", "stated")

    assert effective_style(StudentProfile()) == ("unknown", None)


def test_stated_style_shapes_plan_and_is_logged():
    plan = AnswerPlan(strategy="guided_teaching", rationale="r")
    adjusted, levers = apply_profile_to_plan(
        plan, TeachingState(stage="deepen"), LearningState(topic="t"), StudentProfile(stated_style="step_by_step")
    )
    assert adjusted.strategy == "step_by_step"
    assert levers == ["style_steps", "stated_style"]


# ---------------- 3c: in-session adaptation ----------------


def _ts(**kwargs):
    return TeachingState(**{"topic_anchor": ("t", None), "stage": "introduce", **kwargs})


def _replies_until_explained(previous, adapt=True):
    ls = LearningState(topic="t", learning_progress="improving")
    state, replies = previous, 0
    while True:
        state = advance_teaching_state(state, ls, None, adapt_in_session=adapt)
        replies += 1
        if state.stage != "check":
            return replies


def test_stuck_arcs_counted_and_carried_across_topics():
    ls_stuck = LearningState(topic="t", learning_progress="stuck")
    state = advance_teaching_state(_ts(stage="check", turns_in_stage=1), ls_stuck)
    assert (state.stage, state.session_arcs_stuck) == ("deepen", 1)

    moved = advance_teaching_state(state, LearningState(topic="other"))
    assert (moved.stage, moved.session_arcs_stuck) == ("introduce", 1)


def test_mastery_counted():
    ls = LearningState(topic="t", learning_progress="mastered")
    state = advance_teaching_state(_ts(stage="check", turns_in_stage=1), ls)
    assert (state.stage, state.session_arcs_mastered) == ("wrap_up", 1)


def test_struggling_session_shortens_arc_to_minimum():
    struggling = _ts(session_arcs_stuck=2)
    assert _replies_until_explained(struggling) == MIN_GUIDED_ATTEMPTS
    assert _replies_until_explained(struggling, adapt=False) == 5
    # One mastered arc this session: guiding does work, keep the full arc.
    assert _replies_until_explained(_ts(session_arcs_stuck=2, session_arcs_mastered=1)) == 5


def test_session_lever_logged_only_when_adapting():
    previous = _ts(stage="check", turns_in_stage=1, session_arcs_stuck=2)
    ls = LearningState(topic="t", learning_progress="improving")
    proposed = advance_teaching_state(previous, ls, None)

    trace = personalization_trace(previous, ls, None, proposed)
    assert trace["levers"] == ["session_short_arc"]
    assert trace["pacing_changed"] is True

    off = personalization_trace(previous, ls, None, proposed, adapt_in_session=False)
    assert off["levers"] == []


# ---------------- 4: what the profile knows about this topic ----------------

KNOWING = StudentProfile(
    confidence=0.6,
    misconceptions=["client-server: thinks the server starts the communication"],
    what_clicked=["pipe-filter: understood after the assembly-line example"],
    prior_knowledge=["programs in Python"],
)


def test_notes_match_the_current_topic():
    notes, levers = select_profile_notes(
        KNOWING, LearningState(topic="cliente-servidor"), TeachingState(stage="check"), SUBJECT
    )
    assert levers == ["known_misconception"]
    assert "server starts the communication" in notes[0]

    notes, levers = select_profile_notes(
        KNOWING, LearningState(topic="pipe filter"), TeachingState(stage="check"), SUBJECT
    )
    assert levers == ["what_clicked"]


def test_no_notes_for_unrelated_topic_or_unconfident_profile():
    assert select_profile_notes(KNOWING, LearningState(topic="blackboard"), TeachingState(), SUBJECT) == ([], [])
    unsure = KNOWING.model_copy(update={"confidence": 0.2})
    assert select_profile_notes(unsure, LearningState(topic="cliente-servidor"), TeachingState(), SUBJECT) == ([], [])
    assert select_profile_notes(StudentProfile(), LearningState(topic="x"), TeachingState(stage="deepen")) == ([], [])


def test_analogy_bridge_only_on_explanation_turns():
    _, guided = select_profile_notes(KNOWING, LearningState(topic="blackboard"), TeachingState(stage="check"), SUBJECT)
    notes, explain = select_profile_notes(KNOWING, LearningState(topic="blackboard"), TeachingState(stage="deepen"), SUBJECT)

    assert "prior_knowledge_analogy" not in guided
    assert explain == ["prior_knowledge_analogy"]
    assert "programs in Python" in notes[0] and "ANALOGY EXCEPTION" in notes[0]


def test_notes_reach_the_turn_reminder():
    class Model:
        def invoke(self, messages):
            self.messages = messages
            return AIMessage(content="ok")

    model = Model()
    state = {
        "messages": [HumanMessage(content="o que é?")],
        "learning_state": LearningState(topic="t"),
        "answer_plan": AnswerPlan(strategy="guided_teaching", rationale="r"),
        "teaching_state": TeachingState(stage="check"),
        "student_profile": StudentProfile(),
        "evidence": [],
        "profile_notes": ["Earlier this student believed: X."],
    }
    generate_answer(state, config=TutorConfig(subject=SUBJECT), model=model)

    reminder = model.messages[-1].content
    assert "About this student:\n- Earlier this student believed: X." in reminder


def test_course_switch_drops_topic_knowledge_keeps_learner():
    scoped, switched = scope_to_course(KNOWING.model_copy(update={"course": "A", "stated_style": "concise"}), "B")
    assert switched
    assert (scoped.misconceptions, scoped.what_clicked) == ([], [])
    assert scoped.prior_knowledge == ["programs in Python"]
    assert scoped.stated_style == "concise"


def test_thread_trajectory(tmp_path):
    db = tmp_path / "metrics.db"
    ensure_metrics_schema(db)
    session = {"thread_id": "t1", "matricula": "A", "session_index": 0, "arm": "on", "profile": None}
    state = {
        "messages": [HumanMessage(content="q")],
        "learning_state": LearningState(topic="loops", learning_progress="improving", frustration_level=0.2),
        "teaching_state": TeachingState(stage="check"),
        "answer_plan": AnswerPlan(strategy="guided_teaching", rationale="r"),
    }
    record_turn(state, "X", session, db_path=db)

    assert thread_trajectory("t1", db) == (
        "turn 1: topic=loops, stage=check, progress=improving, frustration=0.2, "
        "strategy=guided_teaching, depth=light"
    )
    assert thread_trajectory("missing", db) == ""


# ---------------- 5: resume ----------------


def test_snapshot_roundtrip_resets_frustration():
    final_state = {
        "learning_state": LearningState(topic="t", learning_progress="improving", frustration_level=0.7),
        "teaching_state": TeachingState(topic_anchor=("t", None), stage="check", turns_in_stage=2, session_arcs_stuck=1),
    }
    metadata = {"course": "X", TUTOR_STATE_KEY: tutor_state_snapshot(final_state)}

    # Chainlit may hand metadata back as a dict or as its JSON string.
    for stored in (metadata, json.dumps(metadata)):
        restored = restore_tutor_state(stored)
        assert restored["teaching_state"] == final_state["teaching_state"]
        assert restored["learning_state"].learning_progress == "improving"
        assert restored["learning_state"].frustration_level == 0.0


@pytest.mark.parametrize(
    "metadata",
    [None, {}, "not json", {"tutor_state": "x"}, {"tutor_state": {"learning_state": {"intent": "bogus"}, "teaching_state": {}}}],
)
def test_unusable_metadata_resumes_fresh(metadata):
    assert restore_tutor_state(metadata) == {}


def test_snapshot_requires_both_states():
    assert tutor_state_snapshot({"learning_state": LearningState()}) is None
