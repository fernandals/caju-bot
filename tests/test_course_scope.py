"""
A deployment's volume outlives its course: when the container is rebuilt
for a different course, the stored profiles and conversations from the
previous one must not leak into the new course's personalization.
Covers utils/student_profile (scope_to_course, latest_other_thread's
course filter), the profiler keeping the course stamp, and the study's
per-course filter. No LLM calls.
"""

import json
import sqlite3

from agent.profiler import profile_student
from agent.state import StudentProfile
from utils.metrics import ensure_metrics_schema, record_turn
from utils.student_profile import latest_other_thread, scope_to_course, thread_course
from utils.study_analysis import build_sessions
from test_personalization_study import _session, _state  # same folder (pytest rootdir)

OLD, NEW = "Software Architecture", "Compiladores"

LEARNER = dict(
    explanation_style="example_first",
    responds_to_guiding_questions="poorly",
    frustration_tendency="high",
    sessions_observed=4,
    confidence=0.9,
)
COURSE = dict(
    solid_topics=["client-server"],
    shaky_topics=["pipe-filter"],
    tutor_note="Use the pipe-filter diagram.",
)


# ---------------- scope_to_course ----------------


def test_same_course_unchanged():
    profile = StudentProfile(course=OLD, **LEARNER, **COURSE)
    assert scope_to_course(profile, OLD) == (profile, False)


def test_legacy_profile_claimed_by_current_course():
    scoped, switched = scope_to_course(StudentProfile(**LEARNER, **COURSE), OLD)

    assert not switched
    assert scoped.course == OLD
    assert scoped.solid_topics == COURSE["solid_topics"]


def test_course_switch_keeps_learner_drops_course_knowledge():
    scoped, switched = scope_to_course(StudentProfile(course=OLD, **LEARNER, **COURSE), NEW)

    assert switched
    assert scoped.course == NEW
    assert (scoped.solid_topics, scoped.shaky_topics, scoped.tutor_note) == ([], [], "")
    assert {k: getattr(scoped, k) for k in LEARNER} == LEARNER


def test_profiler_keeps_course_stamp():
    class Model:
        def with_structured_output(self, _):
            return self

        def invoke(self, _):
            # The model echoes a wrong course - code must not trust it.
            return StudentProfile(course="something else")

    updated = profile_student(StudentProfile(course=NEW), "Student: oi", model=Model())
    assert updated.course == NEW


# ---------------- latest_other_thread ----------------


def _chainlit_db(tmp_path, threads):
    """threads: [(thread_id, metadata dict or None, created_at)]"""
    db = tmp_path / "chainlit.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        'CREATE TABLE threads ("id" TEXT, "userIdentifier" TEXT, "metadata" TEXT);'
        'CREATE TABLE steps ("id" TEXT, "threadId" TEXT, "type" TEXT, "output" TEXT, "createdAt" TEXT);'
    )
    for thread_id, metadata, created in threads:
        conn.execute(
            "INSERT INTO threads VALUES (?, 'A', ?)",
            (thread_id, json.dumps(metadata) if metadata is not None else None),
        )
        conn.execute(
            "INSERT INTO steps VALUES (?, ?, 'user_message', ?, ?)",
            (f"s-{thread_id}", thread_id, f"msg {thread_id}", created),
        )
    conn.commit()
    conn.close()
    return db


def test_thread_course_tolerates_bad_metadata():
    assert thread_course('{"course": "X"}') == "X"
    assert thread_course(None) is None
    assert thread_course("not json") is None
    assert thread_course("[1, 2]") is None


def test_only_current_course_threads(tmp_path):
    db = _chainlit_db(
        tmp_path,
        [
            ("legacy", None, "2026-09-01"),
            ("old", {"course": OLD}, "2026-09-02"),
            ("new", {"course": NEW}, "2026-09-03"),
            ("old-latest", {"course": OLD, "other": 1}, "2026-09-04"),
        ],
    )

    assert latest_other_thread(db, "A", None)[0] == "old-latest"
    assert latest_other_thread(db, "A", None, course=NEW)[0] == "new"
    assert latest_other_thread(db, "A", "new", course=NEW)[0] == "legacy"
    # After a switch, untagged threads belong to the old course.
    assert latest_other_thread(db, "A", "new", course=NEW, include_untagged=False) is None


# ---------------- study ----------------


def test_study_sessions_per_course(tmp_path):
    db = tmp_path / "metrics.db"
    ensure_metrics_schema(db)
    record_turn(_state(1), OLD, _session("t-old", "A", 0, "on"), db_path=db)
    record_turn(_state(1), NEW, _session("t-new", "A", 1, "off"), db_path=db)

    assert set(build_sessions(db)["threadId"]) == {"t-old", "t-new"}
    assert list(build_sessions(db, discipline=NEW)["threadId"]) == ["t-new"]


def test_study_sessions_discipline_column_migrated(tmp_path):
    db = tmp_path / "metrics.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        'CREATE TABLE study_sessions ("threadId" TEXT PRIMARY KEY, "matricula" TEXT NOT NULL,'
        ' "sessionIndex" INTEGER NOT NULL, "personalization" TEXT NOT NULL, "profile" TEXT,'
        ' "profileConfidence" REAL, "createdAt" TEXT NOT NULL);'
    )
    conn.close()

    ensure_metrics_schema(db)
    record_turn(_state(1), NEW, _session("t1", "A", 0, "on"), db_path=db)

    conn = sqlite3.connect(db)
    assert conn.execute('SELECT "discipline" FROM study_sessions').fetchone() == (NEW,)
    conn.close()
