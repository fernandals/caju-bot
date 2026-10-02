"""
Unit tests for the deterministic teaching-state machine
(agent/teaching.py::advance_teaching_state). No LLM calls, no API key
needed - this is pure Python logic and the single biggest lever on what
a student actually experiences (whether they get guided or just told the
answer), so it's worth locking down before real students hit it.

Flagged as an obvious test target in PROJECT_LOG.md (2026-09-09 entry).
"""

import pytest

from agent.state import LearningState, StudentProfile, TeachingState
from agent.teaching import GUIDED_TURN_CAP, advance_teaching_state


def learning_state(**kwargs) -> LearningState:
    defaults = dict(topic="loops", subtopic="for-loop")
    defaults.update(kwargs)
    return LearningState(**defaults)


def teaching_state(**kwargs) -> TeachingState:
    defaults = dict(topic_anchor=("loops", "for-loop"), mode="guided", stage="introduce", turns_in_stage=0)
    defaults.update(kwargs)
    return TeachingState(**defaults)


class TestTopicChange:
    def test_new_topic_resets_to_introduce(self):
        previous = teaching_state(stage="deepen", turns_in_stage=3)
        ls = learning_state(topic="recursion", subtopic=None)

        result = advance_teaching_state(previous, ls)

        assert result.topic_anchor == ("recursion", None)
        assert result.mode == "guided"
        assert result.stage == "introduce"
        assert result.turns_in_stage == 0

    def test_first_turn_ever_is_a_topic_change(self):
        # Default TeachingState() has topic_anchor (None, None), which only
        # matches a LearningState with no topic/subtopic either.
        previous = TeachingState()
        ls = learning_state()

        result = advance_teaching_state(previous, ls)

        assert result.stage == "introduce"
        assert result.turns_in_stage == 0

    def test_subtopic_change_within_same_topic_also_resets(self):
        previous = teaching_state(topic_anchor=("loops", "for-loop"), stage="check", turns_in_stage=2)
        ls = learning_state(topic="loops", subtopic="while-loop")

        result = advance_teaching_state(previous, ls)

        assert result.stage == "introduce"


class TestFrustrationEscapeValve:
    def test_high_frustration_forces_direct_deepen(self):
        previous = teaching_state(stage="check", turns_in_stage=1)
        ls = learning_state(frustration_level=0.9)

        result = advance_teaching_state(previous, ls)

        assert result.mode == "direct"
        assert result.stage == "deepen"

    def test_frustration_just_below_default_threshold_does_not_escape(self):
        previous = teaching_state(stage="check", turns_in_stage=1)
        ls = learning_state(frustration_level=0.59)

        result = advance_teaching_state(previous, ls)

        assert result.mode == "guided"

    def test_sensitive_profile_lowers_escape_threshold(self):
        previous = teaching_state(stage="check", turns_in_stage=1)
        ls = learning_state(frustration_level=0.5)  # below default 0.6, above sensitive 0.45
        profile = StudentProfile(frustration_tendency="high", confidence=0.8)

        result = advance_teaching_state(previous, ls, profile)

        assert result.mode == "direct"
        assert result.stage == "deepen"

    def test_low_confidence_profile_does_not_lower_threshold(self):
        previous = teaching_state(stage="check", turns_in_stage=1)
        ls = learning_state(frustration_level=0.5)
        profile = StudentProfile(frustration_tendency="high", confidence=0.1)

        result = advance_teaching_state(previous, ls, profile)

        # Confidence below PROFILE_CONFIDENCE_FLOOR (0.4) -> default 0.6 threshold applies.
        assert result.mode == "guided"

    def test_exam_prep_intent_forces_direct_deepen(self):
        previous = teaching_state(stage="introduce", turns_in_stage=0)
        ls = learning_state(intent="exam_prep", frustration_level=0.0)

        result = advance_teaching_state(previous, ls)

        assert result.mode == "direct"
        assert result.stage == "deepen"

    def test_exam_prep_on_new_topic_goes_direct_immediately(self):
        # Regression: the topic-change reset used to return before the
        # escape valves, so a cramming student got a guiding question on
        # the first message of every new topic.
        previous = teaching_state(topic_anchor=("loops", "for-loop"))
        ls = learning_state(topic="loops", subtopic="while-loop", intent="exam_prep")

        result = advance_teaching_state(previous, ls)

        assert result.topic_anchor == ("loops", "while-loop")
        assert result.mode == "direct"
        assert result.stage == "deepen"

    def test_frustration_carried_to_new_topic_goes_direct(self):
        previous = teaching_state(topic_anchor=("loops", "for-loop"))
        ls = learning_state(topic="recursion", subtopic=None, frustration_level=0.8)

        result = advance_teaching_state(previous, ls)

        assert result.mode == "direct"

    def test_practice_intent_does_not_force_direct(self):
        previous = teaching_state(stage="introduce", turns_in_stage=0)
        ls = learning_state(intent="practice", frustration_level=0.0)

        result = advance_teaching_state(previous, ls)

        assert result.mode == "guided"


class TestReferStage:
    def test_refer_then_mastered_goes_to_wrap_up(self):
        previous = teaching_state(stage="refer", turns_in_stage=0)
        ls = learning_state(learning_progress="mastered")

        result = advance_teaching_state(previous, ls)

        assert result.stage == "wrap_up"
        assert result.mode == "guided"

    def test_refer_then_improving_resumes_check(self):
        previous = teaching_state(stage="refer", turns_in_stage=0)
        ls = learning_state(learning_progress="improving")

        result = advance_teaching_state(previous, ls)

        assert result.stage == "check"

    @pytest.mark.parametrize("progress", ["stable", "stuck"])
    def test_refer_then_no_engagement_concedes_to_deepen(self, progress):
        previous = teaching_state(stage="refer", turns_in_stage=0)
        ls = learning_state(learning_progress=progress)

        result = advance_teaching_state(previous, ls)

        assert result.stage == "deepen"


class TestGuidedLoop:
    def test_mastered_on_first_reply_asks_one_more_question(self):
        # A right answer to the (deliberately easy) introduce question is
        # not enough to wrap up the topic - confirm with one more.
        previous = teaching_state(stage="introduce", turns_in_stage=0)
        ls = learning_state(learning_progress="mastered")

        result = advance_teaching_state(previous, ls)

        assert result.stage == "check"
        assert result.turns_in_stage == 1

    def test_mastered_after_a_nudge_wraps_up(self):
        previous = teaching_state(stage="check", turns_in_stage=1)
        ls = learning_state(learning_progress="mastered")

        result = advance_teaching_state(previous, ls)

        assert result.stage == "wrap_up"

    def test_stuck_on_first_guided_reply_is_not_conceded_yet(self):
        # Regression guard for the "tracker over-labels stuck early" fix:
        # the very first reply within the arc should get one more guiding
        # question, not the full explanation.
        previous = teaching_state(stage="introduce", turns_in_stage=0)
        ls = learning_state(learning_progress="stuck")

        result = advance_teaching_state(previous, ls)

        assert result.stage == "check"
        assert result.turns_in_stage == 1

    def test_stuck_after_a_nudge_concedes_to_deepen(self):
        previous = teaching_state(stage="check", turns_in_stage=1)
        ls = learning_state(learning_progress="stuck")

        result = advance_teaching_state(previous, ls)

        assert result.stage == "deepen"

    def test_still_engaged_keeps_looping_in_check(self):
        previous = teaching_state(stage="check", turns_in_stage=1)
        ls = learning_state(learning_progress="stable")

        result = advance_teaching_state(previous, ls)

        assert result.stage == "check"
        assert result.turns_in_stage == 2

    def test_guided_turn_cap_forces_concession(self):
        previous = teaching_state(stage="check", turns_in_stage=GUIDED_TURN_CAP - 1)
        ls = learning_state(learning_progress="stable")

        result = advance_teaching_state(previous, ls)

        assert result.stage == "deepen"

    def test_profile_poor_guiding_response_concedes_on_first_reply(self):
        previous = teaching_state(stage="introduce", turns_in_stage=0)
        ls = learning_state(learning_progress="improving")
        profile = StudentProfile(responds_to_guiding_questions="poorly", confidence=0.9)

        result = advance_teaching_state(previous, ls, profile)

        assert result.stage == "deepen"

    def test_profile_poor_guiding_response_ignored_below_confidence_floor(self):
        previous = teaching_state(stage="introduce", turns_in_stage=0)
        ls = learning_state(learning_progress="improving")
        profile = StudentProfile(responds_to_guiding_questions="poorly", confidence=0.2)

        result = advance_teaching_state(previous, ls, profile)

        assert result.stage == "check"


class TestDeepenAndWrapUp:
    def test_deepen_always_advances_to_wrap_up(self):
        previous = teaching_state(stage="deepen", turns_in_stage=0)
        ls = learning_state(learning_progress="stable")

        result = advance_teaching_state(previous, ls)

        assert result.stage == "wrap_up"
        assert result.mode == "guided"

    def test_wrap_up_on_continued_engagement_starts_fresh_cycle(self):
        previous = teaching_state(stage="wrap_up", turns_in_stage=0)
        ls = learning_state(learning_progress="stable")

        result = advance_teaching_state(previous, ls)

        assert result.stage == "introduce"
        assert result.turns_in_stage == 0
