"""
Unit tests for the conditional routing functions in agent/graph.py.
Pure functions over TutorState - no LLM calls needed.

route_after_tracking decides whether a turn takes the cheap "greet"
fast path or goes through the full pedagogical pipeline. A false
positive here (a real question misrouted to "greet") silently skips
planning/retrieval/teaching for that turn, so its edge cases matter.
"""

from langchain_core.messages import HumanMessage

from agent.graph import route_after_planning, route_after_tracking
from agent.state import AnswerPlan, LearningState


def state_with(message: str, learning_state: LearningState | None = None):
    return {
        "messages": [HumanMessage(content=message)],
        "learning_state": learning_state or LearningState(topic=None, subtopic=None),
    }


class TestRouteAfterTracking:
    def test_pure_greeting_goes_to_greet(self):
        assert route_after_tracking(state_with("oi")) == "greet"

    def test_thanks_goes_to_greet(self):
        assert route_after_tracking(state_with("valeu!")) == "greet"

    def test_greeting_with_punctuation_and_emoji_style_noise(self):
        assert route_after_tracking(state_with("Oi!!")) == "greet"

    def test_real_question_goes_to_planning(self):
        state = state_with("como funciona um for loop?")
        assert route_after_tracking(state) == "planning"

    def test_greeting_word_as_prefix_of_real_question_goes_to_planning(self):
        # "ola" is a match anchor at start, but the regex is anchored to the
        # WHOLE (stripped) message via ^...$, so a real question starting
        # with a greeting word must not be misrouted.
        state = state_with("ola, alguem pode me explicar recursao?")
        assert route_after_tracking(state) == "planning"

    def test_topic_already_set_forces_planning_even_if_smalltalk_like(self):
        # If tracking already extracted a topic this turn, the greeting
        # fast path must never fire (would silently drop a real question).
        state = state_with("ok", learning_state=LearningState(topic="loops"))
        assert route_after_tracking(state) == "planning"

    def test_open_question_set_forces_planning(self):
        ls = LearningState(open_question="why does the loop not terminate?")
        state = state_with("obrigado", learning_state=ls)
        assert route_after_tracking(state) == "planning"

    def test_empty_message_defaults_to_planning(self):
        state = state_with("")
        assert route_after_tracking(state) == "planning"

    def test_unmatched_short_message_defaults_to_planning(self):
        # A short, ambiguous message that isn't a confident smalltalk match
        # must fall through to planning, not be dropped.
        state = state_with("hm")
        assert route_after_tracking(state) == "planning"


class TestRouteAfterPlanning:
    def test_needs_retrieval_routes_to_retrieve(self):
        state = {"answer_plan": AnswerPlan(needs_retrieval=True, strategy="guided_teaching", rationale="r")}
        assert route_after_planning(state) == "retrieve"

    def test_no_retrieval_routes_to_generate_answer(self):
        state = {"answer_plan": AnswerPlan(needs_retrieval=False, strategy="direct_answer", rationale="r")}
        assert route_after_planning(state) == "generate_answer"
