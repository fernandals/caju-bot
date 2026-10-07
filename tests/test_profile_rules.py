"""
Tests for the profile rules applied in code (no LLM calls): topic
familiarity matching (agent/familiarity.py), the profile-driven pacing
limits in agent/teaching.py, and the plan rules (agent/plan_profile.py).
"""

import pytest

from agent.familiarity import core_words, topic_familiarity, topics_match
from agent.nodes import resolve_teaching_instructions
from agent.plan_profile import apply_profile_to_plan, leads_with_example
from agent.prompts import EXAMPLE_FIRST_NOTE
from agent.state import AnswerPlan, LearningState, StudentProfile, TeachingState
from agent.teaching import (
    GUIDED_TURN_CAP,
    GUIDED_TURN_CAP_SHAKY,
    MIN_GUIDED_ATTEMPTS,
    advance_teaching_state,
    guided_turn_cap,
)

SUBJECT = "Software Architecture"

# ---------------- familiarity ----------------


@pytest.mark.parametrize(
    "current, profile_topic",
    [
        # Real tracker / profiler strings for the same topic.
        ("cliente-servidor", "Client-Server architecture"),
        ("Arquitetura de Software Estilo Client-Server", "cliente-servidor style in software architecture"),
        ("pipe filter", "Pipe-Filter architectural style"),
        ("Software Architecture pipe-filter style", "pipe-filter architecture"),
        ("client-server and pipe-filter architectures", "pipe-filter architecture"),
    ],
)
def test_same_topic_matches_across_spellings(current, profile_topic):
    assert topics_match(core_words(current, SUBJECT), core_words(profile_topic, SUBJECT))


@pytest.mark.parametrize(
    "current, profile_topic",
    [
        ("blackboard style", "client-server architecture"),
        ("pipe-filter", "client-server"),
        # Nothing left once course-generic words go: never matches.
        ("architecture styles", "architectural styles"),
        ("software architecture basics", "Software Architecture"),
    ],
)
def test_different_or_generic_topics_do_not_match(current, profile_topic):
    assert not topics_match(core_words(current, SUBJECT), core_words(profile_topic, SUBJECT))


def test_topic_familiarity_lists_and_precedence():
    ls = LearningState(topic="cliente-servidor")

    assert topic_familiarity(ls, StudentProfile(solid_topics=["client-server"]), SUBJECT) == "solid"
    assert topic_familiarity(ls, StudentProfile(shaky_topics=["client-server"]), SUBJECT) == "shaky"
    assert topic_familiarity(
        ls, StudentProfile(solid_topics=["client-server"], shaky_topics=["client-server"]), SUBJECT
    ) == "shaky"
    assert topic_familiarity(ls, StudentProfile(), SUBJECT) == "new"
    assert topic_familiarity(LearningState(), StudentProfile(solid_topics=["x"]), SUBJECT) == "new"


# ---------------- pacing ----------------

CONFIDENT = dict(confidence=0.8)


def _arc(profile, familiarity, progress="improving", replies=6):
    """Stages after each student reply, starting from the introduce turn."""
    state = TeachingState(topic_anchor=("t", None), stage="introduce")
    ls = LearningState(topic="t", learning_progress=progress)
    stages = []
    for _ in range(replies):
        state = advance_teaching_state(state, ls, profile, familiarity)
        stages.append(state.stage)
        if state.stage != "check":
            break
    return stages


def test_turn_cap_never_below_minimum():
    worst = StudentProfile(responds_to_guiding_questions="poorly", **CONFIDENT)
    assert guided_turn_cap(worst, "shaky") == MIN_GUIDED_ATTEMPTS == 2


@pytest.mark.parametrize(
    "profile, familiarity, expected_replies",
    [
        (None, "new", GUIDED_TURN_CAP),
        (StudentProfile(**CONFIDENT), "shaky", GUIDED_TURN_CAP_SHAKY),
        (StudentProfile(responds_to_guiding_questions="poorly", **CONFIDENT), "new", MIN_GUIDED_ATTEMPTS),
        # Below the confidence floor the profile changes nothing.
        (StudentProfile(responds_to_guiding_questions="poorly", confidence=0.2), "shaky", GUIDED_TURN_CAP),
    ],
)
def test_replies_before_explaining(profile, familiarity, expected_replies):
    stages = _arc(profile, familiarity)
    assert stages[-1] == "deepen"
    assert len(stages) == expected_replies


def test_stuck_on_first_reply_never_concedes():
    worst = StudentProfile(responds_to_guiding_questions="poorly", **CONFIDENT)
    assert _arc(worst, "shaky", progress="stuck")[:2] == ["check", "deepen"]


def test_solid_topic_accepts_mastery_on_first_reply():
    profile = StudentProfile(**CONFIDENT)
    assert _arc(profile, "solid", progress="mastered") == ["wrap_up"]
    assert _arc(profile, "new", progress="mastered") == ["check", "wrap_up"]


def test_frustration_escape_still_bypasses_the_minimum():
    state = TeachingState(topic_anchor=("t", None), stage="introduce")
    ls = LearningState(topic="t", frustration_level=0.8)
    assert advance_teaching_state(state, ls, None).mode == "direct"


# ---------------- plan rules ----------------

DEEPEN = TeachingState(stage="deepen")
CHECK = TeachingState(stage="check")
LS = LearningState(topic="t")


def _plan(**kwargs):
    return AnswerPlan(**{"strategy": "guided_teaching", "response_depth": "medium", "rationale": "r", **kwargs})


def test_guided_turns_untouched():
    profile = StudentProfile(explanation_style="example_first", **CONFIDENT)
    plan = _plan()
    assert apply_profile_to_plan(plan, CHECK, LS, profile, "shaky") == (plan, [])


def test_student_chosen_strategy_untouched():
    profile = StudentProfile(explanation_style="step_by_step", **CONFIDENT)
    plan = _plan(strategy="exercise_first")
    assert apply_profile_to_plan(plan, DEEPEN, LS, profile) == (plan, [])


def test_shaky_topic_gets_steps():
    plan, levers = apply_profile_to_plan(_plan(), DEEPEN, LS, StudentProfile(**CONFIDENT), "shaky")
    assert plan.strategy == "step_by_step"
    assert levers == ["shaky_topic_steps"]


@pytest.mark.parametrize(
    "style, plan_kwargs, comprehension, expected, lever",
    [
        ("step_by_step", {}, "medium", {"strategy": "step_by_step"}, "style_steps"),
        ("example_first", {}, "medium", {"include_examples": True}, "style_examples"),
        ("concise", {"response_depth": "deep"}, "medium", {"response_depth": "medium"}, "style_depth"),
        ("detailed", {"response_depth": "light"}, "medium", {"response_depth": "medium"}, "style_depth"),
        ("detailed", {"response_depth": "medium"}, "low", {"response_depth": "deep"}, "style_depth"),
    ],
)
def test_style_rules(style, plan_kwargs, comprehension, expected, lever):
    profile = StudentProfile(explanation_style=style, **CONFIDENT)
    ls = LearningState(topic="t", comprehension_level=comprehension)

    plan, levers = apply_profile_to_plan(_plan(**plan_kwargs), DEEPEN, ls, profile)

    assert {k: getattr(plan, k) for k in expected} == expected
    assert levers == [lever]


def test_direct_mode_counts_as_explanation_turn():
    profile = StudentProfile(explanation_style="step_by_step", **CONFIDENT)
    direct = TeachingState(mode="direct", stage="deepen")
    plan, levers = apply_profile_to_plan(_plan(strategy="direct_answer"), direct, LS, profile)
    assert plan.strategy == "step_by_step" and levers == ["style_steps"]


def test_empty_profile_changes_nothing():
    # The study's "off" arm gets StudentProfile() - every rule must be inert.
    plan = _plan()
    assert apply_profile_to_plan(plan, DEEPEN, LS, StudentProfile(), "shaky") == (plan, [])
    assert _arc(StudentProfile(), "shaky") == _arc(None, "new")


def test_example_first_note_reaches_generation():
    profile = StudentProfile(explanation_style="example_first", **CONFIDENT)
    plan, _ = apply_profile_to_plan(_plan(), DEEPEN, LS, profile)

    assert leads_with_example(plan, DEEPEN, profile)
    assert not leads_with_example(plan, CHECK, profile)
    assert EXAMPLE_FIRST_NOTE in resolve_teaching_instructions(DEEPEN, plan.strategy, example_first=True)
    assert EXAMPLE_FIRST_NOTE not in resolve_teaching_instructions(DEEPEN, plan.strategy)
