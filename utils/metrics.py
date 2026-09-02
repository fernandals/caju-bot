"""
Anonymous per-turn pedagogical metrics.

The agent graph (agent/graph.py) computes rich pedagogical signals every
turn - learning state, answer plan, teaching stage, evidence used - and
today throws them away once the reply is sent. This module captures one
row per turn into a dedicated SQLite file, kept deliberately separate
from Chainlit's chainlit.db (whose schema is the framework's and shifts
between versions).

Anonymity is by omission, not by hashing: no enrollment id, no Chainlit
thread/user id, nothing that ties a row back to a student. Each row is an
independent event with no cross-turn correlation - enough to see "the
class is stuck on topic X", not "student Y progressed over the term".
That is the tradeoff the professor accepted.

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

# id is a fresh random uuid per write - no link to student or thread.
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
"""

_COLUMNS = (
    "id",
    "createdAt",
    "discipline",
    "topic",
    "subtopic",
    "intent",
    "comprehensionLevel",
    "learningProgress",
    "frustrationLevel",
    "currentDifficulty",
    "strategy",
    "responseDepth",
    "teachingMode",
    "teachingStage",
    "studentProfile",
    "citations",
)


def ensure_metrics_schema(db_path: Path = DB_PATH) -> None:
    """Create the turn_metrics table if missing. Same pattern as
    chainlit_app._ensure_sqlite_schema()."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


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


def record_turn(final_state, discipline: str, db_path: Path = DB_PATH) -> None:
    """
    Insert one anonymous row for the turn that just finished. Must never
    break the chat: any failure is logged and swallowed, mirroring the
    resilience of Chainlit's own SQLAlchemyDataLayer.execute_sql.
    """
    try:
        if not final_state:
            return

        ls = final_state.get("learning_state")
        ap = final_state.get("answer_plan")
        ts = final_state.get("teaching_state")
        sp = final_state.get("student_profile")

        row = (
            uuid.uuid4().hex,
            datetime.now(timezone.utc).isoformat(),
            discipline,
            _field(ls, "topic"),
            _field(ls, "subtopic"),
            _field(ls, "intent"),
            _field(ls, "comprehension_level"),
            _field(ls, "learning_progress"),
            _field(ls, "frustration_level"),
            _field(ls, "current_difficulty"),
            _field(ap, "strategy"),
            _field(ap, "response_depth"),
            _field(ts, "mode"),
            _field(ts, "stage"),
            _field(sp, "current_profile"),
            json.dumps(_turn_citations(final_state), ensure_ascii=False),
        )

        placeholders = ",".join("?" * len(_COLUMNS))
        # timeout: a couple of connected students can finish a turn at
        # once and briefly contend for the write lock on this one file.
        conn = sqlite3.connect(db_path, timeout=10)
        try:
            conn.execute(
                f"INSERT INTO turn_metrics VALUES ({placeholders})", row
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        logger.warning("Failed to record turn metrics", exc_info=True)
