"""
The chat around the tutor: which path a message takes through the graph,
the topic tracking, the reply buttons, conversation titles, editing a
message, resuming a conversation, and the enrollment check at login.
"""

import contextlib
import io
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.graph import (
    is_about_tutor,
    is_smalltalk,
    is_topics_request,
    route_after_planning,
    route_after_tracking,
    route_start,
)
from agent.nodes import about_tutor, dialogue_window, update_tracking, without_course_topic
from agent.prompts import GENERATE_PROMPT, TUTOR_NAME
from agent.state import AnswerPlan, LearningState, TeachingState, TutorConfig
from agent.titler import MAX_TITLE_CHARS, clean_title
from utils.chat_session import TUTOR_STATE_KEY, needs_title, restore_tutor_state, truncate_for_edit, tutor_state_snapshot
from utils.helpers import is_enrolled, load_roster
from utils.quick_replies import (
    ACTION_NAME,
    DONT_KNOW,
    EXAMPLE,
    EXERCISE,
    HINT,
    NEXT_TOPIC,
    NOT_FOUND,
    STARTERS,
    STYLE_ACTION_NAME,
    TEST_ME,
    TOPICS,
    load_quick_replies,
    quick_replies_for,
    save_quick_replies,
)


def message(text, topic=None, **learning):
    return {"messages": [HumanMessage(content=text)], "learning_state": LearningState(topic=topic, **learning)}


# ---------------- routing ----------------


@pytest.mark.parametrize(
    "text, learning, route",
    [
        ("oi", {}, "greet"),
        ("valeu!", {}, "greet"),
        ("Oi!!", {}, "greet"),
        ("como funciona um for loop?", {}, "planning"),
        # Anchored to the whole message: a greeting word first is not enough.
        ("ola, alguem pode me explicar recursao?", {}, "planning"),
        # Tracking found something to teach: never the greeting shortcut.
        ("ok", {"topic": "loops"}, "planning"),
        ("obrigado", {"open_question": "why does the loop not terminate?"}, "planning"),
        ("", {}, "planning"),
        ("hm", {}, "planning"),
    ],
)
def test_only_pure_small_talk_skips_the_pipeline(text, learning, route):
    assert route_after_tracking(message(text, **learning)) == route


@pytest.mark.parametrize(
    "text, topics",
    [
        (TOPICS.message, True),
        ("quais são os tópicos?", True),
        ("Quais os assuntos da matéria?", True),
        ("que conteúdos posso estudar", True),
        ("o que posso estudar aqui?", True),
        ("e quais temas da disciplina podemos ver?", True),
        ("o que é uma árvore AVL?", False),
        ("quais assuntos caem na prova de árvores?", False),
        ("quais tópicos de heap são importantes?", False),
        (None, False),
    ],
)
def test_topics_request(text, topics):
    assert is_topics_request(text) is topics


def test_topic_list_needs_an_outline():
    outline = TutorConfig(subject="EDB2", course_outline=["Árvores AVL"])
    assert route_start(message(TOPICS.message), outline) == "list_topics"
    assert route_start(message(TOPICS.message), TutorConfig(subject="EDB2")) == "tracking"
    assert route_start(message("o que é uma AVL?"), outline) == "tracking"


@pytest.mark.parametrize(
    "text, about",
    [
        ("qual é o seu nome?", True),
        ("Qual seu nome", True),
        ("oi, qual é o seu nome?", True),
        ("como você se chama?", True),
        ("quem é você?", True),
        ("quem é vc", True),
        ("o que significa CAJU?", True),
        ("o que é o caju bot?", True),
        ("CAJU quer dizer o quê?", True),
        ("o que você faz?", True),
        ("você tem nome?", True),
        ("qual é o nome da rotação dupla?", False),
        ("quem inventou a árvore AVL?", False),
        ("o que você acha da minha resposta?", False),
        (None, False),
    ],
)
def test_about_tutor_request(text, about):
    assert is_about_tutor(text) is about


def test_who_are_you_gets_the_introduction():
    config = TutorConfig(subject="EDB2")
    assert route_start(message("qual é o seu nome?"), config) == "about_tutor"
    with contextlib.redirect_stdout(io.StringIO()):
        out = about_tutor(message("qual é o seu nome?"), config=config)
    # Not a pedagogical turn: no metrics row, no title, the arc untouched.
    assert out["greeted"]
    assert TUTOR_NAME in out["messages"][0].content and "EDB2" in out["messages"][0].content
    # The model knows its name too, in the rest of the conversation.
    assert TUTOR_NAME in GENERATE_PROMPT


def test_retrieval_only_when_planned():
    assert route_after_planning({"answer_plan": AnswerPlan(needs_retrieval=True, strategy="guided_teaching", rationale="r")}) == "retrieve"
    assert route_after_planning({"answer_plan": AnswerPlan(needs_retrieval=False, strategy="direct_answer", rationale="r")}) == "generate_answer"


# ---------------- tracking ----------------
COURSE = "Estruturas de Dados Básicas II"


@pytest.mark.parametrize(
    "topic, subtopic, expected",
    [
        # The course named as the topic: the real one was in subtopic...
        (COURSE, "Heap", ("Heap", None)),
        ("estruturas de dados", "Árvore Binária de Busca", ("Árvore Binária de Busca", None)),
        # ...or there was none.
        (COURSE, None, (None, None)),
        # A real topic is kept, course words in it or not.
        ("Árvores AVL", "rotações", ("Árvores AVL", "rotações")),
        ("Árvores B e Estruturas para Grandes Dados", None, ("Árvores B e Estruturas para Grandes Dados", None)),
        (None, None, (None, None)),
    ],
)
def test_the_course_is_never_the_topic(topic, subtopic, expected):
    learning = without_course_topic(LearningState(topic=topic, subtopic=subtopic), COURSE)
    assert (learning.topic, learning.subtopic) == expected


class Tracker:
    def __init__(self, result):
        self.result, self.prompt = result, None

    def with_structured_output(self, _):
        return self

    def invoke(self, prompt):
        self.prompt = prompt
        return self.result


def test_tracking_knows_the_course_and_its_topics():
    tracker = Tracker(LearningState(topic=COURSE, subtopic="Árvores AVL"))
    config = TutorConfig(subject=COURSE, course_outline=["Heap e Heapsort", "Árvores AVL"])
    with contextlib.redirect_stdout(io.StringIO()):
        out = update_tracking(message("o que é uma árvore AVL?"), model=tracker, config=config)
    assert COURSE in tracker.prompt and "- Heap e Heapsort" in tracker.prompt
    assert (out["learning_state"].topic, out["learning_state"].subtopic) == ("Árvores AVL", None)


def test_tracking_is_told_when_the_tutor_asked_the_exercise_topic():
    tracker = Tracker(LearningState(topic="Heap"))
    state = {**message("heap"), "teaching_state": TeachingState(exercise_topic_pending=True)}
    with contextlib.redirect_stdout(io.StringIO()):
        update_tracking(state, model=tracker)
    assert "which topic the student wants an exercise on" in tracker.prompt


# ---------------- reply buttons ----------------


def turn(stage="check", mode="guided", strategy="guided_teaching", greeted=False):
    return {
        "greeted": greeted,
        "teaching_state": TeachingState(mode=mode, stage=stage),
        "answer_plan": AnswerPlan(strategy=strategy, rationale="r"),
    }


@pytest.mark.parametrize(
    "state, buttons",
    [
        (turn("introduce"), [HINT, DONT_KNOW]),
        (turn("check"), [HINT, DONT_KNOW]),
        (turn("refer"), [NOT_FOUND]),
        (turn("deepen"), [EXAMPLE, TEST_ME]),
        (turn("wrap_up"), [HINT, DONT_KNOW]),
        (turn("concluded"), [EXERCISE, NEXT_TOPIC]),
        (turn("check", mode="direct"), [EXAMPLE, TEST_ME]),
        # Mid-attempt at an exercise or a hint.
        (turn("concluded", strategy="exercise_first"), [HINT, DONT_KNOW]),
        (turn("concluded", strategy="hint_only"), [HINT, DONT_KNOW]),
        (turn(greeted=True), STARTERS),
        ({"greeted": False}, []),
    ],
)
def test_buttons_follow_the_teaching_stage(state, buttons):
    assert quick_replies_for(state) == buttons


def test_buttons_after_the_topic_list_and_without_a_topic():
    listed = {**turn(greeted=True), "messages": [HumanMessage(content=TOPICS.message)]}
    assert quick_replies_for(listed) == [r for r in STARTERS if r is not TOPICS]

    no_topic = {**turn("deepen", mode="direct"), "learning_state": LearningState(topic=None)}
    assert quick_replies_for(no_topic) == [TOPICS]


def test_no_button_skips_the_guided_arc_or_the_pipeline():
    stages = ("introduce", "check", "refer", "deepen", "wrap_up", "concluded")
    every = STARTERS + [r for s in stages for r in quick_replies_for(turn(s))]
    for reply in every:
        # A button matching the small-talk regex would skip the pipeline.
        assert not is_smalltalk(reply.message), reply.message
    for stage in ("introduce", "check"):
        for reply in quick_replies_for(turn(stage)):
            # A one-click "just tell me" would undercut guiding.
            assert "resposta" not in reply.message.lower() and "explica" not in reply.message.lower()


def test_buttons_saved_for_a_resumed_conversation():
    assert load_quick_replies(save_quick_replies([HINT, DONT_KNOW])) == [HINT, DONT_KNOW]
    assert load_quick_replies([]) == []
    assert [load_quick_replies(bad) for bad in (None, "x", [{"label": "only"}])] == [None, None, None]


# ---------------- titles ----------------


@pytest.mark.parametrize(
    "name, needs",
    [
        (None, True),
        ("", True),
        ("olá!", True),
        (ACTION_NAME, True),
        (STYLE_ACTION_NAME, True),
        ("Nova conversa", True),
        ("Conteúdos da disciplina", True),
        ("quem é você?", True),
        ("quero entender o pipe filter", False),
    ],
)
def test_which_thread_names_get_a_generated_title(name, needs):
    assert needs_title(name) is needs


@pytest.mark.parametrize(
    "raw, title",
    [
        ('"Estilo Pipe-Filter".', "Estilo Pipe-Filter"),
        ("Título: Cliente-Servidor", "Cliente-Servidor"),
        ("**Revisão para a prova**\nexplicação extra", "Revisão para a prova"),
        ("  Blackboard   e   controle  ", "Blackboard e controle"),
        ("", None),
        ('""', None),
        (None, None),
    ],
)
def test_model_titles_are_cleaned(raw, title):
    assert clean_title(raw) == title


def test_long_title_is_cut_on_a_word():
    title = clean_title("palavra " * 20)
    assert len(title) <= MAX_TITLE_CHARS + 1 and title.endswith("…") and not title[:-1].endswith(" ")


# ---------------- history ----------------


def history():
    return [SystemMessage("sys"), HumanMessage("q1", id="u1"), AIMessage("a1", id="b1"), HumanMessage("q2", id="u2"), AIMessage("a2", id="b2")]


@pytest.mark.parametrize("edited, kept, is_edit", [("u1", ["sys"], True), ("u2", ["sys", "q1", "a1"], True), ("new", None, False)])
def test_editing_a_message_drops_it_and_what_came_after(edited, kept, is_edit):
    messages = history()
    assert truncate_for_edit(messages, edited) is is_edit
    assert [m.content for m in messages] == (kept or [m.content for m in history()])


def test_dialogue_window_skips_the_app_prompt():
    assert [m.content for m in dialogue_window([SystemMessage("app"), HumanMessage("q1"), AIMessage("a1")], 8)] == ["q1", "a1"]
    assert [m.content for m in dialogue_window([SystemMessage("app")] + [HumanMessage(str(i)) for i in range(10)], 3)] == ["7", "8", "9"]


def test_resumed_conversation_continues_where_it_stopped():
    final = {
        "learning_state": LearningState(topic="t", learning_progress="improving", frustration_level=0.7),
        "teaching_state": TeachingState(topic_anchor=("t", None), stage="check", turns_in_stage=2, session_arcs_stuck=1),
    }
    metadata = {"course": "X", TUTOR_STATE_KEY: tutor_state_snapshot(final)}

    # Chainlit may hand the metadata back as a dict or as its JSON string.
    for stored in (metadata, json.dumps(metadata)):
        restored = restore_tutor_state(stored)
        assert restored["teaching_state"] == final["teaching_state"]
        assert restored["learning_state"].learning_progress == "improving"
        assert restored["learning_state"].frustration_level == 0.0  # a new sitting

    assert tutor_state_snapshot({"learning_state": LearningState()}) is None


@pytest.mark.parametrize(
    "metadata",
    [None, {}, "not json", {"tutor_state": "x"}, {"tutor_state": {"learning_state": {"intent": "bogus"}, "teaching_state": {}}}],
)
def test_unusable_saved_state_resumes_fresh(metadata):
    assert restore_tutor_state(metadata) == {}


# ---------------- enrollment ----------------


def test_roster(tmp_path):
    path = tmp_path / "roster.txt"
    path.write_text("# turma IMD1151 2026.2\n202312345\n\n   \n# comentário\n202398765\n2023 555\n", encoding="utf-8")
    roster = load_roster(path)

    assert roster == {"202312345", "202398765", "2023 555"}
    assert is_enrolled("202312345", roster)
    assert is_enrolled("202312345 \n", roster)  # pasted with a trailing newline
    assert not is_enrolled("999999999", roster)
    assert not is_enrolled("2023555", roster)  # comparison is exact
    # No roster file: nobody gets in.
    assert load_roster(tmp_path / "missing.txt") == set()
    assert not is_enrolled("202312345", set())
