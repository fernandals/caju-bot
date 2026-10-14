"""
What the app writes to and reads from its SQLite files for the
personalization study: per-turn metrics (utils/metrics.py, including
migrating an older metrics.db), the study analysis
(utils/study_analysis.py), and which past conversations the profiler
reads (utils/student_profile.py). On temporary databases.
"""

import json
import sqlite3

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.state import AnswerPlan, LearningState, StudentProfile, TeachingState
from utils.metrics import (
    ensure_metrics_schema,
    load_study_session,
    next_session_index,
    record_turn,
    thread_trajectory,
)
from utils.student_profile import latest_other_thread, thread_course
from utils.study_analysis import build_sessions, compare_arms, informed_sessions, paired_by_student


def turn_state(turn: int, progress="stable", frustration=0.1, stage="check"):
    messages = []
    for _ in range(turn):
        messages += [HumanMessage(content="q"), AIMessage(content="a")]
    return {
        "messages": messages,
        "learning_state": LearningState(topic="loops", learning_progress=progress, frustration_level=frustration),
        "teaching_state": TeachingState(stage=stage),
        "answer_plan": AnswerPlan(strategy="guided_teaching", rationale="r"),
        "student_profile": StudentProfile(),
        "personalization": {"levers": [], "pacing_changed": False, "profile_in_prompt": False},
    }


def session(thread, matricula="A", index=0, arm="on", confidence=0.5):
    return {"thread_id": thread, "matricula": matricula, "session_index": index, "arm": arm, "profile": StudentProfile(confidence=confidence)}


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "metrics.db"
    ensure_metrics_schema(path)
    return path


# ---------------- recording ----------------


def test_older_metrics_db_is_migrated(tmp_path):
    db = tmp_path / "metrics.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE turn_metrics (
            "id" TEXT PRIMARY KEY, "createdAt" TEXT NOT NULL, "discipline" TEXT NOT NULL,
            "topic" TEXT, "subtopic" TEXT, "intent" TEXT, "comprehensionLevel" TEXT,
            "learningProgress" TEXT, "frustrationLevel" REAL, "currentDifficulty" TEXT,
            "strategy" TEXT, "responseDepth" TEXT, "teachingMode" TEXT,
            "teachingStage" TEXT, "studentProfile" TEXT, "citations" TEXT
        );
        INSERT INTO turn_metrics ("id", "createdAt", "discipline") VALUES ('old', '2026-09-01', 'X');
        CREATE TABLE study_sessions ("threadId" TEXT PRIMARY KEY, "matricula" TEXT NOT NULL,
            "sessionIndex" INTEGER NOT NULL, "personalization" TEXT NOT NULL, "profile" TEXT,
            "profileConfidence" REAL, "createdAt" TEXT NOT NULL);
        """
    )
    conn.close()

    ensure_metrics_schema(db)
    record_turn(turn_state(1), "Compiladores", session("t1", "2024001"), "msg1", db_path=db)

    conn = sqlite3.connect(db)
    rows = conn.execute(
        'SELECT "id", "matricula", "threadId", "messageId", "turnIndex", "personalization" FROM turn_metrics ORDER BY "createdAt"'
    ).fetchall()
    discipline = conn.execute('SELECT "discipline" FROM study_sessions').fetchone()
    conn.close()

    assert rows[0] == ("old", None, None, None, None, None)
    assert rows[1][1:] == ("2024001", "t1", "msg1", 1, "on")
    assert discipline == ("Compiladores",)


def test_a_conversation_is_one_study_session(db):
    record_turn(turn_state(1), "X", session("t1", "2024001", arm="off"), db_path=db)
    record_turn(turn_state(2), "X", session("t1", "2024001", arm="on"), db_path=db)

    assert next_session_index("2024001", db) == 1
    assert next_session_index("2024999", db) == 0
    # The arm of the first recorded turn sticks (a resume keeps it).
    assert load_study_session("t1", db)["arm"] == "off"


def test_thread_trajectory_for_the_profiler(db):
    record_turn(turn_state(1, progress="improving", frustration=0.2), "X", session("t1"), db_path=db)

    assert thread_trajectory("t1", db) == (
        "turn 1: topic=loops, stage=check, progress=improving, frustration=0.2, strategy=guided_teaching, depth=light"
    )
    assert thread_trajectory("missing", db) == ""


# ---------------- analysis ----------------


@pytest.fixture
def study(db):
    def conversation(thread, matricula, index, arm, outcomes):
        for turn, (progress, frustration, stage) in enumerate(outcomes, start=1):
            record_turn(turn_state(turn, progress, frustration, stage), "X", session(thread, matricula, index, arm), f"{thread}-{turn}", db_path=db)

    # Student A masters at turn 2 with the profile, never without.
    conversation("a0", "A", 0, "on", [("stable", 0.1, "introduce"), ("mastered", 0.0, "wrap_up")])
    conversation("a1", "A", 1, "off", [("stable", 0.3, "introduce"), ("stuck", 0.6, "deepen")])
    # Student B: only a conversation without the profile so far.
    conversation("b0", "B", 0, "off", [("improving", 0.2, "check")])
    return db


def test_session_outcomes_and_comparison(study):
    sessions = build_sessions(study)
    by_thread = sessions.set_index("threadId")

    assert by_thread.loc["a0", "mastered"] and by_thread.loc["a0", "turnsToMastered"] == 2
    assert not by_thread.loc["a1", "mastered"]
    assert (by_thread.loc["a1", "concededShare"], by_thread.loc["a1", "frustrationPeak"]) == (0.5, 0.6)

    arms = compare_arms(sessions)
    assert arms.loc["mastered", "better"] == "on" and arms.loc["concededShare", "better"] == "on"

    paired = paired_by_student(sessions)
    assert list(paired.index) == ["A"]  # B has no conversation with the profile
    assert paired.loc["A", "mastered"] == 1.0

    sessions.loc[sessions["threadId"] == "b0", "profileConfidence"] = 0.25
    assert set(informed_sessions(sessions)["threadId"]) == {"a0", "a1"}


def test_feedback_joined_from_chainlit_db(study, tmp_path):
    chainlit = tmp_path / "chainlit.db"
    conn = sqlite3.connect(chainlit)
    # As Chainlit stores it: the rating points at the reply's parent run
    # step ("run-a0-2"), not at the reply message ("a0-2").
    conn.executescript(
        'CREATE TABLE feedbacks ("id" TEXT, "forId" TEXT, "threadId" TEXT, "value" INT, "comment" TEXT);'
        'CREATE TABLE steps ("id" TEXT, "type" TEXT, "parentId" TEXT);'
        "INSERT INTO steps VALUES ('run-a0-2', 'run', NULL), ('a0-2', 'assistant_message', 'run-a0-2');"
        "INSERT INTO steps VALUES ('run-a1-2', 'run', NULL), ('a1-2', 'assistant_message', 'run-a1-2');"
        "INSERT INTO feedbacks VALUES ('f1', 'run-a0-2', 'a0', 1, NULL), ('f2', 'run-a1-2', 'a1', 0, NULL);"
    )
    conn.close()

    sessions = build_sessions(study, chainlit).set_index("threadId")

    assert (sessions.loc["a0", "thumbsUp"], sessions.loc["a1", "thumbsDown"]) == (1, 1)


def test_sessions_per_course(db):
    record_turn(turn_state(1), "Software Architecture", session("t-old", index=0), db_path=db)
    record_turn(turn_state(1), "Compiladores", session("t-new", index=1, arm="off"), db_path=db)

    assert set(build_sessions(db)["threadId"]) == {"t-old", "t-new"}
    assert list(build_sessions(db, discipline="Compiladores")["threadId"]) == ["t-new"]


def test_empty_db(db):
    assert build_sessions(db).empty


# ---------------- conversations the profiler reads ----------------


def test_profiler_reads_only_the_current_courses_conversations(tmp_path):
    db = tmp_path / "chainlit.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        'CREATE TABLE threads ("id" TEXT, "userIdentifier" TEXT, "metadata" TEXT);'
        'CREATE TABLE steps ("id" TEXT, "threadId" TEXT, "type" TEXT, "output" TEXT, "createdAt" TEXT);'
    )
    threads = [
        ("legacy", None, "2026-09-01"),
        ("old", {"course": "Software Architecture"}, "2026-09-02"),
        ("new", {"course": "Compiladores"}, "2026-09-03"),
        ("old-latest", {"course": "Software Architecture", "other": 1}, "2026-09-04"),
    ]
    for thread_id, metadata, created in threads:
        conn.execute("INSERT INTO threads VALUES (?, 'A', ?)", (thread_id, json.dumps(metadata) if metadata else None))
        conn.execute("INSERT INTO steps VALUES (?, ?, 'user_message', 'msg', ?)", (f"s-{thread_id}", thread_id, created))
    conn.commit()
    conn.close()

    assert latest_other_thread(db, "A", None)[0] == "old-latest"
    assert latest_other_thread(db, "A", None, course="Compiladores")[0] == "new"
    assert latest_other_thread(db, "A", "new", course="Compiladores")[0] == "legacy"
    # Right after a course switch, untagged threads belong to the old course.
    assert latest_other_thread(db, "A", "new", course="Compiladores", include_untagged=False) is None

    assert thread_course('{"course": "X"}') == "X"
    assert [thread_course(bad) for bad in (None, "not json", "[1, 2]")] == [None, None, None]
