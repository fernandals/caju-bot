"""
Per-turn pedagogical metrics, plus the per-conversation records of the
personalization study.

The agent graph (agent/graph.py) computes rich pedagogical signals every
turn - learning state, answer plan, teaching stage, evidence used. This
module captures one row per turn into a dedicated SQLite file, kept
deliberately separate from Chainlit's chainlit.db (whose schema is the
framework's and shifts between versions).

IDENTIFIABLE since 2026-10-06. The first version was anonymous by
omission (no matrícula, no thread id, no cross-turn link) - enough to see
"the class is stuck on topic X", but not whether personalization helps a
student over time. Rows now carry the matrícula, the conversation
(thread) id, the turn number and the Chainlit message id (to join the
student's 👍/👎 from chainlit.db). No names: the login name is free text
anyway; the matrícula is the identity, as in student_profiles.db.

`study_sessions` holds one row per conversation: which personalization
arm it ran in (utils/personalization.py) and a snapshot of the student's
REAL profile at its start (even when the tutor was not shown it), so the
analysis knows what the tutor could have known at each point.

The file lives in data/chats/ (the already-mounted Railway volume, see
DEPLOY.md). The professor pulls it on demand and views aggregates in the
local Streamlit app (app.py) - there is no second deployed surface.
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path("data/chats/metrics.db")

# id is a fresh random uuid per write.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS turn_metrics (
    "id" TEXT PRIMARY KEY,
    "createdAt" TEXT NOT NULL,
    "discipline" TEXT NOT NULL,
    "topic" TEXT,
    "subtopic" TEXT,
    "intent" TEXT,
    "comprehensionLevel" TEXT,
    "learningProgress" TEXT,
    "frustrationLevel" REAL,
    "currentDifficulty" TEXT,
    "strategy" TEXT,
    "responseDepth" TEXT,
    "teachingMode" TEXT,
    "teachingStage" TEXT,
    "studentProfile" TEXT,
    "citations" TEXT
);

CREATE TABLE IF NOT EXISTS study_sessions (
    "threadId" TEXT PRIMARY KEY,
    "matricula" TEXT NOT NULL,
    "sessionIndex" INTEGER NOT NULL,
    "personalization" TEXT NOT NULL,
    "profile" TEXT,
    "profileConfidence" REAL,
    "createdAt" TEXT NOT NULL
);
"""

# Added to turn_metrics for the personalization study. Created by
# ALTER TABLE so a metrics.db from before the study keeps its rows (with
# NULLs in these columns) instead of failing every insert.
_STUDY_COLUMNS = {
    "matricula": "TEXT",
    "threadId": "TEXT",
    "messageId": "TEXT",
    "turnIndex": "INTEGER",
    "personalization": "TEXT",
    "profileConfidence": "REAL",
    "profileLevers": "TEXT",
    "pacingChanged": "INTEGER",
    "profileInPrompt": "INTEGER",
    "planChanged": "INTEGER",
    "topicFamiliarity": "TEXT",
}


def ensure_metrics_schema(db_path: Path = DB_PATH) -> None:
    """Create the tables if missing and add any study column an older
    metrics.db lacks. Same pattern as chainlit_app._ensure_sqlite_schema()."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_SCHEMA)

        existing = {row[1] for row in conn.execute('PRAGMA table_info("turn_metrics")')}
        for column, sql_type in _STUDY_COLUMNS.items():
            if column not in existing:
                conn.execute(f'ALTER TABLE turn_metrics ADD COLUMN "{column}" {sql_type}')

        conn.commit()
    finally:
        conn.close()


def next_session_index(matricula: str, db_path: Path = DB_PATH) -> int:
    """
    0-based number of this student's next conversation: how many they have
    already had a turn in. A chat opened and left empty is never recorded,
    so it doesn't advance the count (or flip the on/off alternation).
    """
    try:
        conn = sqlite3.connect(db_path)
        try:
            (count,) = conn.execute(
                'SELECT COUNT(*) FROM study_sessions WHERE "matricula" = ?',
                (matricula,),
            ).fetchone()
        finally:
            conn.close()
        return int(count)
    except Exception:  # noqa: BLE001 - a wrong index only shifts the alternation
        logger.warning("Could not count study sessions for %s", matricula, exc_info=True)
        return 0


def load_study_session(thread_id: str, db_path: Path = DB_PATH) -> dict | None:
    """The study_sessions row of a conversation, for resuming it in the
    same arm. None if it was never recorded (e.g. from before the study)."""
    try:
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                'SELECT "matricula", "sessionIndex", "personalization", "profile" '
                'FROM study_sessions WHERE "threadId" = ?',
                (thread_id,),
            ).fetchone()
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        logger.warning("Could not read study session %s", thread_id, exc_info=True)
        return None

    if row is None:
        return None

    return {
        "matricula": row[0],
        "session_index": row[1],
        "arm": row[2],
        "profile_json": row[3],
    }


def record_study_session(session: dict, db_path: Path = DB_PATH) -> None:
    """
    Insert the conversation's study_sessions row on its first recorded
    turn. Idempotent (INSERT OR IGNORE): every later turn of the same
    conversation is a no-op, so the arm and profile snapshot stay those of
    the conversation's start.

    `session` keys: thread_id, matricula, session_index, arm, profile
    (the student's real StudentProfile at the start).
    """
    try:
        if not session.get("thread_id") or not session.get("matricula"):
            return

        profile = session.get("profile")

        conn = sqlite3.connect(db_path, timeout=10)
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO study_sessions
                    ("threadId", "matricula", "sessionIndex", "personalization",
                     "profile", "profileConfidence", "createdAt")
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session["thread_id"],
                    session["matricula"],
                    session.get("session_index", 0),
                    session.get("arm", "on"),
                    profile.model_dump_json() if profile is not None else None,
                    _field(profile, "confidence"),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        logger.warning("Failed to record study session", exc_info=True)


def _field(obj, name, default=None):
    """Read `name` off a pydantic model or a dict, tolerating None."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _turn_citations(final_state) -> list[dict]:
    """Structured view of the material pulled for this turn.

    evidence[i] lines up with retrieved_docs[i] - assess_documents builds
    both from the same kept list in the same order.
    """
    docs = final_state.get("retrieved_docs") or []
    evidence = final_state.get("evidence") or []

    citations = []
    for doc, ev in zip(docs, evidence):
        meta = getattr(doc, "metadata", {}) or {}
        citations.append(
            {
                "file": meta.get("file_path"),
                "section_id": meta.get("section_id"),
                "section_title": meta.get("section_title"),
                "page_start": meta.get("page_start"),
                "citation": _field(ev, "citation"),
            }
        )
    return citations


def _turn_index(final_state) -> int:
    """1-based number of this student turn within the conversation."""
    return sum(
        1
        for msg in final_state.get("messages") or []
        if getattr(msg, "type", None) == "human"
    )


def record_turn(
    final_state,
    discipline: str,
    session: dict | None = None,
    message_id: str | None = None,
    db_path: Path = DB_PATH,
) -> None:
    """
    Insert one row for the turn that just finished (and, on the first
    turn of a conversation, its study_sessions row). Must never break the
    chat: any failure is logged and swallowed, mirroring the resilience of
    Chainlit's own SQLAlchemyDataLayer.execute_sql.

    `session` is the conversation's study record (see
    record_study_session); `message_id` is the Chainlit id of the tutor's
    reply, which chainlit.db's feedbacks.forId points at.
    """
    try:
        if not final_state:
            return

        session = session or {}

        if session:
            record_study_session(session, db_path)

        ls = final_state.get("learning_state")
        ap = final_state.get("answer_plan")
        ts = final_state.get("teaching_state")
        sp = final_state.get("student_profile")
        pz = final_state.get("personalization") or {}
        real_profile = session.get("profile")

        row = {
            "id": uuid.uuid4().hex,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "discipline": discipline,
            "topic": _field(ls, "topic"),
            "subtopic": _field(ls, "subtopic"),
            "intent": _field(ls, "intent"),
            "comprehensionLevel": _field(ls, "comprehension_level"),
            "learningProgress": _field(ls, "learning_progress"),
            "frustrationLevel": _field(ls, "frustration_level"),
            "currentDifficulty": _field(ls, "current_difficulty"),
            "strategy": _field(ap, "strategy"),
            "responseDepth": _field(ap, "response_depth"),
            "teachingMode": _field(ts, "mode"),
            "teachingStage": _field(ts, "stage"),
            "studentProfile": _field(sp, "explanation_style"),
            "citations": json.dumps(_turn_citations(final_state), ensure_ascii=False),
            "matricula": session.get("matricula"),
            "threadId": session.get("thread_id"),
            "messageId": message_id,
            "turnIndex": _turn_index(final_state),
            "personalization": session.get("arm"),
            # The real profile's confidence, also in the "off" arm - how
            # much the tutor COULD have known about this student.
            "profileConfidence": _field(real_profile if real_profile is not None else sp, "confidence"),
            "profileLevers": json.dumps(pz.get("levers") or []),
            "pacingChanged": int(bool(pz.get("pacing_changed"))),
            "profileInPrompt": int(bool(pz.get("profile_in_prompt"))),
            "planChanged": int(bool(pz.get("plan_changed"))),
            "topicFamiliarity": pz.get("familiarity"),
        }

        columns = ", ".join(f'"{c}"' for c in row)
        placeholders = ",".join("?" * len(row))
        # timeout: a couple of connected students can finish a turn at
        # once and briefly contend for the write lock on this one file.
        conn = sqlite3.connect(db_path, timeout=10)
        try:
            conn.execute(
                f"INSERT INTO turn_metrics ({columns}) VALUES ({placeholders})",
                tuple(row.values()),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        logger.warning("Failed to record turn metrics", exc_info=True)
