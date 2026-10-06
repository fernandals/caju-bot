"""
Tests for the personalization study: on/off assignment
(utils/personalization.py), the per-turn pacing trace
(agent/teaching.personalization_trace), recording (utils/metrics.py, incl.
migrating a pre-study metrics.db) and the analysis
(utils/study_analysis.py). No LLM calls.
"""

import json
import sqlite3

from langchain_core.messages import AIMessage, HumanMessage

from agent.state import LearningState, StudentProfile, TeachingState
from agent.teaching import advance_teaching_state, personalization_trace
from utils.metrics import (
    ensure_metrics_schema,
    load_study_session,
    next_session_index,
    record_turn,
)
from utils.personalization import assign_arm
from utils.study_analysis import (
    build_sessions,
    compare_arms,
    informed_sessions,
    paired_by_student,
)

# ---------------- assignment ----------------


class TestAssignArm:
    def test_alternates_within_a_student(self):
        arms = [assign_arm("2024001", i, "alternate") for i in range(6)]
        assert arms[0::2] == [arms[0]] * 3
        assert arms[1::2] == [arms[1]] * 3
        assert arms[0] != arms[1]

    def test_stable_across_calls(self):
        assert assign_arm("2024001", 3, "alternate") == assign_arm("2024001", 3, "alternate")

    def test_first_arm_is_counterbalanced_across_students(self):
        firsts = {assign_arm(f"2024{n:03d}", 0, "alternate") for n in range(40)}
        assert firsts == {"on", "off"}

    def test_always_and_off_modes(self):
        assert {assign_arm("2024001", i, "always") for i in range(4)} == {"on"}
        assert {assign_arm("2024001", i, "off") for i in range(4)} == {"off"}

    def test_no_matricula_is_on(self):
        assert assign_arm(None, 1, "alternate") == "on"


# ---------------- pacing trace ----------------


def _ls(**kwargs) -> LearningState:
    return LearningState(**{"topic": "loops", "subtopic": "for", **kwargs})


def _ts(**kwargs) -> TeachingState:
    return TeachingState(**{"topic_anchor": ("loops", "for"), "stage": "check", "turns_in_stage": 1, **kwargs})


class TestPersonalizationTrace:
    def test_no_profile_changes_nothing(self):
        previous, ls = _ts(), _ls()
        proposed = advance_teaching_state(previous, ls, None)

        trace = personalization_trace(previous, ls, None, proposed)

        assert trace == {"levers": [], "pacing_changed": False}

    def test_sensitive_student_escapes_earlier(self):
        profile = StudentProfile(frustration_tendency="high", confidence=0.8)
        previous, ls = _ts(), _ls(frustration_level=0.5)
        proposed = advance_teaching_state(previous, ls, profile)

        trace = personalization_trace(previous, ls, profile, proposed)

        assert proposed.mode == "direct"
        assert trace["levers"] == ["early_frustration_escape"]
        assert trace["pacing_changed"] is True

    def test_lever_in_play_without_changing_the_outcome(self):
        # Already above the normal threshold: the lowered one is irrelevant.
        profile = StudentProfile(frustration_tendency="high", confidence=0.8)
        previous, ls = _ts(), _ls(frustration_level=0.9)
        proposed = advance_teaching_state(previous, ls, profile)

        trace = personalization_trace(previous, ls, profile, proposed)

        assert trace["levers"] == []
        assert trace["pacing_changed"] is False

    def test_skip_guiding(self):
        profile = StudentProfile(responds_to_guiding_questions="poorly", confidence=0.8)
        previous, ls = _ts(), _ls()
        proposed = advance_teaching_state(previous, ls, profile)

        trace = personalization_trace(previous, ls, profile, proposed)

        assert "skip_guiding" in trace["levers"]
        assert trace["pacing_changed"] is True


# ---------------- recording ----------------

_PRE_STUDY_SCHEMA = """
CREATE TABLE turn_metrics (
    "id" TEXT PRIMARY KEY, "createdAt" TEXT NOT NULL, "discipline" TEXT NOT NULL,
    "topic" TEXT, "subtopic" TEXT, "intent" TEXT, "comprehensionLevel" TEXT,
    "learningProgress" TEXT, "frustrationLevel" REAL, "currentDifficulty" TEXT,
    "strategy" TEXT, "responseDepth" TEXT, "teachingMode" TEXT,
    "teachingStage" TEXT, "studentProfile" TEXT, "citations" TEXT
);
INSERT INTO turn_metrics ("id", "createdAt", "discipline") VALUES ('old', '2026-09-01', 'X');
"""


def _state(turn: int, progress="stable", frustration=0.1, stage="check", pacing_changed=False):
    messages = []
    for _ in range(turn):
        messages += [HumanMessage(content="q"), AIMessage(content="a")]
    return {
        "messages": messages,
        "learning_state": LearningState(
            topic="loops", learning_progress=progress, frustration_level=frustration
        ),
        "teaching_state": TeachingState(stage=stage),
        "student_profile": StudentProfile(),
        "personalization": {"levers": [], "pacing_changed": pacing_changed, "profile_in_prompt": False},
    }


def _session(thread, matricula, index, arm, confidence=0.5):
    return {
        "thread_id": thread,
        "matricula": matricula,
        "session_index": index,
        "arm": arm,
        "profile": StudentProfile(confidence=confidence),
    }


class TestRecording:
    def test_migrates_a_pre_study_metrics_db(self, tmp_path):
        db = tmp_path / "metrics.db"
        conn = sqlite3.connect(db)
        conn.executescript(_PRE_STUDY_SCHEMA)
        conn.close()

        ensure_metrics_schema(db)
        record_turn(_state(1), "X", _session("t1", "2024001", 0, "on"), "msg1", db_path=db)

        conn = sqlite3.connect(db)
        rows = conn.execute(
            'SELECT "id", "matricula", "threadId", "messageId", "turnIndex", "personalization" '
            "FROM turn_metrics ORDER BY \"createdAt\""
        ).fetchall()
        conn.close()

        assert rows[0] == ("old", None, None, None, None, None)
        assert rows[1][1:] == ("2024001", "t1", "msg1", 1, "on")

    def test_session_recorded_once_and_counted(self, tmp_path):
        db = tmp_path / "metrics.db"
        ensure_metrics_schema(db)

        session = _session("t1", "2024001", 0, "off")
        record_turn(_state(1), "X", session, db_path=db)
        record_turn(_state(2), "X", {**session, "arm": "on"}, db_path=db)

        assert next_session_index("2024001", db) == 1
        assert next_session_index("2024999", db) == 0
        # The arm of the first recorded turn sticks (resume keeps it).
        assert load_study_session("t1", db)["arm"] == "off"


# ---------------- analysis ----------------


def _record_conversation(db, thread, matricula, index, arm, outcomes):
    for turn, (progress, frustration, stage) in enumerate(outcomes, start=1):
        record_turn(
            _state(turn, progress, frustration, stage),
            "X",
            _session(thread, matricula, index, arm),
            f"{thread}-{turn}",
            db_path=db,
        )


class TestAnalysis:
    def _db(self, tmp_path):
        db = tmp_path / "metrics.db"
        ensure_metrics_schema(db)
        # Student A: masters at turn 2 when "on", never when "off".
        _record_conversation(db, "a0", "A", 0, "on", [("stable", 0.1, "introduce"), ("mastered", 0.0, "wrap_up")])
        _record_conversation(db, "a1", "A", 1, "off", [("stable", 0.3, "introduce"), ("stuck", 0.6, "deepen")])
        # Student B: only an "off" conversation so far.
        _record_conversation(db, "b0", "B", 0, "off", [("improving", 0.2, "check")])
        return db

    def test_session_outcomes(self, tmp_path):
        sessions = build_sessions(self._db(tmp_path)).set_index("threadId")

        assert sessions.loc["a0", "mastered"]
        assert sessions.loc["a0", "turnsToMastered"] == 2
        assert not sessions.loc["a1", "mastered"]
        assert sessions.loc["a1", "concededShare"] == 0.5
        assert sessions.loc["a1", "frustrationPeak"] == 0.6

    def test_feedback_joined_from_chainlit_db(self, tmp_path):
        db = self._db(tmp_path)
        chainlit = tmp_path / "chainlit.db"
        conn = sqlite3.connect(chainlit)
        # As Chainlit stores it: the rating points at the reply's parent
        # run step ("run-a0-2"), not at the reply message ("a0-2").
        conn.executescript(
            'CREATE TABLE feedbacks ("id" TEXT, "forId" TEXT, "threadId" TEXT, "value" INT, "comment" TEXT);'
            'CREATE TABLE steps ("id" TEXT, "type" TEXT, "parentId" TEXT);'
            "INSERT INTO steps VALUES ('run-a0-2', 'run', NULL);"
            "INSERT INTO steps VALUES ('a0-2', 'assistant_message', 'run-a0-2');"
            "INSERT INTO steps VALUES ('run-a1-2', 'run', NULL);"
            "INSERT INTO steps VALUES ('a1-2', 'assistant_message', 'run-a1-2');"
            "INSERT INTO feedbacks VALUES ('f1', 'run-a0-2', 'a0', 1, NULL);"
            "INSERT INTO feedbacks VALUES ('f2', 'run-a1-2', 'a1', 0, NULL);"
        )
        conn.close()

        sessions = build_sessions(db, chainlit).set_index("threadId")

        assert sessions.loc["a0", "thumbsUp"] == 1
        assert sessions.loc["a1", "thumbsDown"] == 1

    def test_compare_and_pair(self, tmp_path):
        sessions = build_sessions(self._db(tmp_path))

        arms = compare_arms(sessions)
        assert arms.loc["mastered", "better"] == "on"
        assert arms.loc["concededShare", "better"] == "on"

        paired = paired_by_student(sessions)
        assert list(paired.index) == ["A"]  # B has no "on" conversation
        assert paired.loc["A", "mastered"] == 1.0

    def test_informed_filter(self, tmp_path):
        sessions = build_sessions(self._db(tmp_path))
        sessions.loc[sessions["threadId"] == "b0", "profileConfidence"] = 0.25

        assert set(informed_sessions(sessions)["threadId"]) == {"a0", "a1"}

    def test_empty_db(self, tmp_path):
        db = tmp_path / "metrics.db"
        ensure_metrics_schema(db)
        assert build_sessions(db).empty
