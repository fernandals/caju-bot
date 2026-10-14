"""
How the student profile reaches the tutor - all in code, no model call:
topic familiarity, the plan rules, the notes about the current topic, the
stated style, the profiler's guarded fields, the course switch, and the
study's with/without-profile assignment and trace.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.familiarity import core_words, topic_familiarity, topics_match
from agent.nodes import generate_answer, resolve_teaching_instructions
from agent.plan_profile import apply_profile_to_plan, leads_with_example
from agent.profile_notes import select_profile_notes
from agent.profiler import _confidence_for, profile_student, worth_profiling
from agent.prompts import EXAMPLE_FIRST_NOTE
from agent.state import AnswerPlan, LearningState, StudentProfile, TeachingState, TutorConfig
from agent.teaching import (
    PROFILE_CONFIDENCE_FLOOR,
    advance_teaching_state,
    effective_style,
    personalization_trace,
)
from utils.chat_session import should_ask_style
from utils.personalization import assign_arm
from utils.student_profile import scope_to_course

SUBJECT = "Software Architecture"
CONFIDENT = dict(confidence=0.8)
DEEPEN, CHECK = TeachingState(stage="deepen"), TeachingState(stage="check")
LS = LearningState(topic="t")


def plan(**kwargs) -> AnswerPlan:
    return AnswerPlan(**{"strategy": "guided_teaching", "response_depth": "medium", "rationale": "r", **kwargs})


# ---------------- topic familiarity ----------------


@pytest.mark.parametrize(
    "current, stored, same",
    [
        # Real tracker / profiler strings for the same topic.
        ("cliente-servidor", "Client-Server architecture", True),
        ("Arquitetura de Software Estilo Client-Server", "cliente-servidor style in software architecture", True),
        ("pipe filter", "Pipe-Filter architectural style", True),
        ("client-server and pipe-filter architectures", "pipe-filter architecture", True),
        ("blackboard style", "client-server architecture", False),
        ("pipe-filter", "client-server", False),
        # Nothing left once course-generic words go: never matches.
        ("architecture styles", "architectural styles", False),
        ("software architecture basics", "Software Architecture", False),
    ],
)
def test_topics_match_across_spellings(current, stored, same):
    assert topics_match(core_words(current, SUBJECT), core_words(stored, SUBJECT)) is same


def test_topic_familiarity():
    topic = LearningState(topic="cliente-servidor")

    assert topic_familiarity(topic, StudentProfile(solid_topics=["client-server"]), SUBJECT) == "solid"
    assert topic_familiarity(topic, StudentProfile(shaky_topics=["client-server"]), SUBJECT) == "shaky"
    both = StudentProfile(solid_topics=["client-server"], shaky_topics=["client-server"])
    assert topic_familiarity(topic, both, SUBJECT) == "shaky"
    assert topic_familiarity(topic, StudentProfile(), SUBJECT) == "new"
    assert topic_familiarity(LearningState(), StudentProfile(solid_topics=["x"]), SUBJECT) == "new"


# ---------------- plan rules ----------------


@pytest.mark.parametrize(
    "profile, familiarity, plan_kwargs, comprehension, expected, levers",
    [
        (StudentProfile(**CONFIDENT), "shaky", {}, "medium", {"strategy": "step_by_step"}, ["shaky_topic_steps"]),
        (StudentProfile(explanation_style="step_by_step", **CONFIDENT), "new", {}, "medium", {"strategy": "step_by_step"}, ["style_steps"]),
        (StudentProfile(explanation_style="example_first", **CONFIDENT), "new", {}, "medium", {"include_examples": True}, ["style_examples"]),
        (StudentProfile(explanation_style="concise", **CONFIDENT), "new", {"response_depth": "deep"}, "medium", {"response_depth": "medium"}, ["style_depth"]),
        (StudentProfile(explanation_style="detailed", **CONFIDENT), "new", {"response_depth": "light"}, "medium", {"response_depth": "medium"}, ["style_depth"]),
        (StudentProfile(explanation_style="detailed", **CONFIDENT), "new", {}, "low", {"response_depth": "deep"}, ["style_depth"]),
        # The student's own answer to the style question applies right away.
        (StudentProfile(stated_style="step_by_step"), "new", {}, "medium", {"strategy": "step_by_step"}, ["style_steps", "stated_style"]),
    ],
)
def test_profile_shapes_explanations(profile, familiarity, plan_kwargs, comprehension, expected, levers):
    learning = LearningState(topic="t", comprehension_level=comprehension)

    adjusted, used = apply_profile_to_plan(plan(**plan_kwargs), DEEPEN, learning, profile, familiarity)

    assert {k: getattr(adjusted, k) for k in expected} == expected
    assert used == levers


def test_plan_rules_leave_alone_what_they_should():
    styled = StudentProfile(explanation_style="step_by_step", **CONFIDENT)
    # Guided turns and a strategy the student chose.
    assert apply_profile_to_plan(plan(), CHECK, LS, styled, "shaky") == (plan(), [])
    assert apply_profile_to_plan(plan(strategy="exercise_first"), DEEPEN, LS, styled) == (plan(strategy="exercise_first"), [])
    # The study's "off" arm gets an empty profile: every rule inert.
    assert apply_profile_to_plan(plan(), DEEPEN, LS, StudentProfile(), "shaky") == (plan(), [])
    # Direct mode counts as an explanation turn.
    direct, levers = apply_profile_to_plan(plan(strategy="direct_answer"), TeachingState(mode="direct", stage="deepen"), LS, styled)
    assert direct.strategy == "step_by_step" and levers == ["style_steps"]


def test_example_first_reaches_generation():
    profile = StudentProfile(explanation_style="example_first", **CONFIDENT)
    adjusted, _ = apply_profile_to_plan(plan(), DEEPEN, LS, profile)

    assert leads_with_example(adjusted, DEEPEN, profile)
    assert not leads_with_example(adjusted, CHECK, profile)
    assert EXAMPLE_FIRST_NOTE in resolve_teaching_instructions(DEEPEN, adjusted.strategy, example_first=True)


# ---------------- stated vs observed style ----------------


def test_style_question_only_until_answered_or_observed():
    assert should_ask_style(StudentProfile())
    assert not should_ask_style(StudentProfile(style_asked=True))
    assert not should_ask_style(StudentProfile(stated_style="concise"))
    assert not should_ask_style(StudentProfile(sessions_observed=1))


def test_stated_style_until_a_confident_observation():
    stated = StudentProfile(stated_style="example_first")
    assert effective_style(stated) == ("example_first", "stated")
    assert effective_style(stated.model_copy(update={"explanation_style": "concise", "confidence": 0.6})) == ("concise", "observed")
    assert effective_style(stated.model_copy(update={"explanation_style": "concise", "confidence": 0.2})) == ("example_first", "stated")
    assert effective_style(StudentProfile()) == ("unknown", None)


# ---------------- what the profile knows about this topic ----------------

KNOWING = StudentProfile(
    confidence=0.6,
    misconceptions=["client-server: thinks the server starts the communication"],
    what_clicked=["pipe-filter: understood after the assembly-line example"],
    prior_knowledge=["programs in Python"],
)


def test_notes_about_the_current_topic():
    notes, levers = select_profile_notes(KNOWING, LearningState(topic="cliente-servidor"), CHECK, SUBJECT)
    assert levers == ["known_misconception"] and "server starts the communication" in notes[0]
    assert select_profile_notes(KNOWING, LearningState(topic="pipe filter"), CHECK, SUBJECT)[1] == ["what_clicked"]

    # Unrelated topic, unconfident profile, empty profile: nothing.
    assert select_profile_notes(KNOWING, LearningState(topic="blackboard"), TeachingState(), SUBJECT) == ([], [])
    unsure = KNOWING.model_copy(update={"confidence": 0.2})
    assert select_profile_notes(unsure, LearningState(topic="cliente-servidor"), TeachingState(), SUBJECT) == ([], [])

    # The analogy with prior knowledge only on explanation turns.
    assert "prior_knowledge_analogy" not in select_profile_notes(KNOWING, LearningState(topic="blackboard"), CHECK, SUBJECT)[1]
    notes, levers = select_profile_notes(KNOWING, LearningState(topic="blackboard"), DEEPEN, SUBJECT)
    assert levers == ["prior_knowledge_analogy"] and "programs in Python" in notes[0]


def test_notes_reach_the_turn_reminder():
    class Model:
        def invoke(self, messages):
            self.messages = messages
            return AIMessage(content="ok")

    model = Model()
    generate_answer(
        {
            "messages": [HumanMessage(content="o que é?")],
            "learning_state": LS,
            "answer_plan": plan(),
            "teaching_state": CHECK,
            "student_profile": StudentProfile(),
            "evidence": [],
            "profile_notes": ["Earlier this student believed: X."],
        },
        config=TutorConfig(subject=SUBJECT),
        model=model,
    )

    assert "About this student:\n- Earlier this student believed: X." in model.messages[-1].content


# ---------------- profiler ----------------


def test_personalization_starts_in_the_second_conversation():
    assert [_confidence_for(n) for n in (0, 1, 2, 3, 4, 10)] == [0.0, PROFILE_CONFIDENCE_FLOOR, 0.6, 0.8, 0.9, 0.9]
    assert not worth_profiling("Student: oi\nTutor: Olá!\nStudent: tchau")
    assert worth_profiling("Student: a\nTutor: b\nStudent: c\nTutor: d\nStudent: e")


def test_profiler_cannot_change_system_fields():
    class Model:
        def with_structured_output(self, _):
            return self

        def invoke(self, prompt):
            self.prompt = prompt
            return StudentProfile(course="other", stated_style="detailed", style_asked=False)

    model = Model()
    previous = StudentProfile(course="X", stated_style="concise", style_asked=True)

    updated = profile_student(previous, "Student: oi", model=model, trajectory="turn 1: stage=check")

    assert (updated.course, updated.stated_style, updated.style_asked) == ("X", "concise", True)
    assert "Per-turn signals:\nturn 1: stage=check" in model.prompt


# ---------------- course switch ----------------


def test_course_switch_keeps_the_learner_and_drops_the_course():
    learner = dict(explanation_style="example_first", frustration_tendency="high", stated_style="concise", confidence=0.9)
    course = dict(solid_topics=["client-server"], shaky_topics=["pipe-filter"], tutor_note="n", misconceptions=["m"], what_clicked=["w"])
    profile = StudentProfile(course="A", prior_knowledge=["Python"], **learner, **course)

    assert scope_to_course(profile, "A") == (profile, False)

    scoped, switched = scope_to_course(profile, "B")
    assert switched and scoped.course == "B"
    assert all(not getattr(scoped, k) for k in course)
    assert {k: getattr(scoped, k) for k in learner} == learner
    assert scoped.prior_knowledge == ["Python"]

    # A profile from before courses were tracked belongs to the current one.
    legacy, switched = scope_to_course(StudentProfile(**course), "A")
    assert not switched and legacy.course == "A" and legacy.solid_topics == ["client-server"]


# ---------------- the study: with / without profile ----------------


def test_study_arm_assignment():
    arms = [assign_arm("2024001", i, "alternate") for i in range(6)]
    assert arms[0::2] == [arms[0]] * 3 and arms[1::2] == [arms[1]] * 3 and arms[0] != arms[1]
    # Counterbalanced: some students start with the profile, some without.
    assert {assign_arm(f"2024{n:03d}", 0, "alternate") for n in range(40)} == {"on", "off"}
    assert {assign_arm("2024001", i, "always") for i in range(4)} == {"on"}
    assert {assign_arm("2024001", i, "off") for i in range(4)} == {"off"}
    assert assign_arm(None, 1, "alternate") == "on"


def _trace(profile, frustration=0.1, familiarity="new", **kwargs):
    previous = TeachingState(topic_anchor=("loops", "for"), stage="check", turns_in_stage=1, **kwargs)
    learning = LearningState(topic="loops", subtopic="for", frustration_level=frustration)
    proposed = advance_teaching_state(previous, learning, profile, familiarity)
    return personalization_trace(previous, learning, profile, proposed, familiarity)


@pytest.mark.parametrize(
    "profile, kwargs, levers, changed",
    [
        (None, {}, [], False),
        (StudentProfile(frustration_tendency="high", **CONFIDENT), {"frustration": 0.5}, ["early_frustration_escape"], True),
        # Already above the normal threshold: the lowered one changed nothing.
        (StudentProfile(frustration_tendency="high", **CONFIDENT), {"frustration": 0.9}, [], False),
        (StudentProfile(responds_to_guiding_questions="poorly", **CONFIDENT), {}, ["poor_guiding_short_arc"], True),
        (StudentProfile(**CONFIDENT), {"familiarity": "shaky"}, ["shaky_topic_short_arc"], None),
        (StudentProfile(**CONFIDENT), {"familiarity": "solid"}, ["solid_topic_fast_mastery"], None),
        (None, {"session_arcs_stuck": 2}, ["session_short_arc"], True),
    ],
)
def test_trace_logs_which_lever_changed_the_pacing(profile, kwargs, levers, changed):
    trace = _trace(profile, **kwargs)

    assert trace["levers"] == levers
    if changed is not None:
        assert trace["pacing_changed"] is changed
