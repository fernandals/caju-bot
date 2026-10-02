"""
Unit tests for the student chat-session helpers: stage-driven quick-reply
buttons, edit-message history truncation, thread title cleanup and the
"does this thread need a title" check. No LLM calls.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.state import AnswerPlan, LearningState, TeachingState
from agent.titler import MAX_TITLE_CHARS, clean_title
from utils.chat_session import needs_title, truncate_for_edit
from utils.quick_replies import (
    ACTION_NAME,
    DONT_KNOW,
    EXAMPLE,
    EXERCISE,
    HINT,
    NEXT_TOPIC,
    NOT_FOUND,
    STARTERS,
    TEST_ME,
    TOPICS,
    quick_replies_for,
)


def turn(stage="check", mode="guided", strategy="guided_teaching", greeted=False):
    return {
        "greeted": greeted,
        "teaching_state": TeachingState(mode=mode, stage=stage),
        "answer_plan": AnswerPlan(strategy=strategy, rationale="test"),
    }


class TestQuickReplies:
    @pytest.mark.parametrize("stage", ["introduce", "check"])
    def test_guided_question_offers_hint_and_dont_know(self, stage):
        assert quick_replies_for(turn(stage=stage)) == [HINT, DONT_KNOW]

    def test_refer_offers_not_found(self):
        assert quick_replies_for(turn(stage="refer")) == [NOT_FOUND]

    def test_deepen_offers_example_and_test(self):
        assert quick_replies_for(turn(stage="deepen")) == [EXAMPLE, TEST_ME]

    def test_wrap_up_offers_exercise_and_next_topic(self):
        assert quick_replies_for(turn(stage="wrap_up")) == [EXERCISE, NEXT_TOPIC]

    def test_direct_mode_is_treated_as_an_explanation(self):
        assert quick_replies_for(turn(stage="check", mode="direct")) == [EXAMPLE, TEST_ME]

    @pytest.mark.parametrize("strategy", ["exercise_first", "hint_only"])
    def test_student_mid_attempt_gets_hint_buttons(self, strategy):
        result = quick_replies_for(turn(stage="wrap_up", strategy=strategy))
        assert result == [HINT, DONT_KNOW]

    def test_greeting_turn_offers_starters(self):
        assert quick_replies_for(turn(greeted=True)) == STARTERS

    def test_no_topic_yet_offers_topic_list(self):
        state = turn(stage="deepen", mode="direct")
        state["learning_state"] = LearningState(topic=None)

        assert quick_replies_for(state) == [TOPICS]

    def test_no_teaching_state_no_buttons(self):
        assert quick_replies_for({"greeted": False}) == []

    def test_no_button_shortcuts_to_the_answer_while_guiding(self):
        # A one-click "just tell me" would undercut the guided arc.
        for stage in ("introduce", "check"):
            for reply in quick_replies_for(turn(stage=stage)):
                text = reply.message.lower()
                assert "resposta" not in text and "explica" not in text

    def test_button_texts_do_not_hit_the_greeting_fast_path(self):
        # A button whose text matched the small-talk regex would skip the
        # whole pipeline (agent/graph.py::route_after_tracking).
        from agent.graph import is_smalltalk

        stages = ("introduce", "check", "refer", "deepen", "wrap_up")
        all_replies = STARTERS + [r for s in stages for r in quick_replies_for(turn(stage=s))]
        for reply in all_replies:
            assert not is_smalltalk(reply.message), reply.message


class TestTruncateForEdit:
    def history(self):
        return [
            SystemMessage("sys"),
            HumanMessage("q1", id="u1"),
            AIMessage("a1", id="b1"),
            HumanMessage("q2", id="u2"),
            AIMessage("a2", id="b2"),
        ]

    def test_edit_of_earlier_message_cuts_from_it(self):
        messages = self.history()

        assert truncate_for_edit(messages, "u1") is True
        assert [m.content for m in messages] == ["sys"]

    def test_edit_of_latest_message_drops_its_old_answer(self):
        messages = self.history()

        assert truncate_for_edit(messages, "u2") is True
        assert [m.content for m in messages] == ["sys", "q1", "a1"]

    def test_new_message_is_not_an_edit(self):
        messages = self.history()

        assert truncate_for_edit(messages, "new-id") is False
        assert len(messages) == 5


class TestNeedsTitle:
    @pytest.mark.parametrize("name", [None, "", "olá!", "oi", ACTION_NAME])
    def test_useless_default_names(self, name):
        assert needs_title(name)

    def test_real_question_name_is_kept(self):
        assert not needs_title("quero entender o pipe filter")


class TestCleanTitle:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Estilo Pipe-Filter", "Estilo Pipe-Filter"),
            ('"Estilo Pipe-Filter".', "Estilo Pipe-Filter"),
            ("Título: Cliente-Servidor", "Cliente-Servidor"),
            ("**Revisão para a prova**\nexplicação extra", "Revisão para a prova"),
            ("  Blackboard   e   controle  ", "Blackboard e controle"),
        ],
    )
    def test_normalizes(self, raw, expected):
        assert clean_title(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "   ", '""'])
    def test_empty_is_none(self, raw):
        assert clean_title(raw) is None

    def test_long_title_cut_on_word_boundary(self):
        title = clean_title("palavra " * 20)
        assert len(title) <= MAX_TITLE_CHARS + 1
        assert title.endswith("…")
        assert not title[:-1].endswith(" ")


class TestDialogueWindow:
    def test_excludes_the_app_system_prompt(self):
        from agent.nodes import dialogue_window

        messages = [SystemMessage("app prompt"), HumanMessage("q1"), AIMessage("a1"), HumanMessage("q2")]

        window = dialogue_window(messages, 8)

        assert [m.content for m in window] == ["q1", "a1", "q2"]

    def test_keeps_only_the_last_n_dialogue_messages(self):
        from agent.nodes import dialogue_window

        messages = [SystemMessage("app prompt")] + [HumanMessage(str(i)) for i in range(10)]

        assert [m.content for m in dialogue_window(messages, 3)] == ["7", "8", "9"]
