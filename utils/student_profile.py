"""
Per-student personalization profile (agent.state.StudentProfile),
persisted by matrícula on the same mounted volume as the other SQLite
files (see DEPLOY.md).

Unlike data/chats/metrics.db, this store IS identifiable - it is keyed by
matrícula, because personalization needs identity. It holds only
slow-moving learning-style traits and a short free-text note, never
transcripts. The profiler that fills it runs once per session (see
agent/profiler.py and the on_chat_start catch-up in chainlit_app.py).
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from agent.state import StudentProfile

logger = logging.getLogger(__name__)

DB_PATH = Path("data/chats/student_profiles.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS student_profiles (
    "matricula" TEXT PRIMARY KEY,
    "profile" TEXT NOT NULL,
    "last_profiled_thread" TEXT,
    "updatedAt" TEXT NOT NULL
);
"""

_MESSAGE_STEP_TYPES = ("user_message", "assistant_message")

# Profile fields about the COURSE, not the learner: they are reset when the
# deployment switches to a different course. Everything else
# (explanation_style, responds_to_guiding_questions, frustration_tendency,
# sessions_observed / confidence) describes how the student learns and
# carries over.
_COURSE_FIELDS = ("solid_topics", "shaky_topics", "tutor_note")


def scope_to_course(profile: StudentProfile, course: str) -> tuple[StudentProfile, bool]:
    """
    (profile valid for `course`, whether it was switched from another one).

    The volume (and so this store) outlives a deployment's course: a
    container rebuilt for next semester's course would otherwise read
    topics like "client-server" from the previous one into the new
    course's pacing (agent/familiarity.py). A profile from before courses
    were recorded (course=None) is claimed by the current course - it was
    built in the only course deployed until then.
    """
    if profile.course == course:
        return profile, False

    if profile.course is None:
        return profile.model_copy(update={"course": course}), False

    reset = {field: StudentProfile.model_fields[field].get_default(call_default_factory=True) for field in _COURSE_FIELDS}
    return profile.model_copy(update={**reset, "course": course}), True


def ensure_profile_schema(db_path: Path = DB_PATH) -> None:
    """Create the student_profiles table if missing. Mirrors
    utils.metrics.ensure_metrics_schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def load_profile(
    matricula: str, db_path: Path = DB_PATH
) -> tuple[StudentProfile, str | None]:
    """
    (profile, last_profiled_thread_id) for a matrícula. Returns a fresh
    default profile and None for a student not seen before, and also on
    any read error - a missing profile must never block a session.
    """
    try:
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT profile, last_profiled_thread "
                "FROM student_profiles WHERE matricula = ?",
                (matricula,),
            ).fetchone()
        finally:
            conn.close()

        if row:
            return StudentProfile.model_validate_json(row[0]), row[1]
    except Exception:  # noqa: BLE001 - degrade to a default profile
        logger.warning(
            "Could not load profile for %s; using default", matricula, exc_info=True
        )

    return StudentProfile(), None


def save_profile(
    matricula: str,
    profile: StudentProfile,
    last_profiled_thread: str | None,
    db_path: Path = DB_PATH,
) -> None:
    """Upsert a student's profile. Swallows errors (logged) like record_turn."""
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        try:
            conn.execute(
                """
                INSERT INTO student_profiles
                    (matricula, profile, last_profiled_thread, updatedAt)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(matricula) DO UPDATE SET
                    profile = excluded.profile,
                    last_profiled_thread = excluded.last_profiled_thread,
                    updatedAt = excluded.updatedAt
                """,
                (
                    matricula,
                    profile.model_dump_json(),
                    last_profiled_thread,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - never break a session over profiling
        logger.warning("Could not save profile for %s", matricula, exc_info=True)


COURSE_METADATA_KEY = "course"


def thread_course(metadata: str | None) -> str | None:
    """The course a Chainlit thread was tagged with (metadata JSON), if any."""
    try:
        value = json.loads(metadata or "{}")
    except (TypeError, ValueError):
        return None
    return value.get(COURSE_METADATA_KEY) if isinstance(value, dict) else None


def latest_other_thread(
    chainlit_db_path: str | Path,
    matricula: str,
    exclude_thread_id: str | None,
    course: str | None = None,
    include_untagged: bool = True,
) -> tuple[str, str] | None:
    """
    (thread_id, transcript) for the student's most recently active
    conversation in Chainlit's own DB that is not `exclude_thread_id`
    (the session starting now), or None if there is no earlier one.

    With `course`, only conversations tagged with that course in the
    thread metadata (chainlit_app tags each new one) - plus untagged ones
    from before tagging existed, unless `include_untagged` is False
    (after a course switch, untagged threads belong to the old course).

    The transcript is a plain "Student:/Tutor:" script built from the
    persisted message steps - enough for the profiler, no internals.
    """
    chainlit_db_path = Path(chainlit_db_path)

    if not chainlit_db_path.is_file():
        return None

    try:
        conn = sqlite3.connect(chainlit_db_path)
        conn.create_function("thread_course", 1, thread_course, deterministic=True)
        try:
            row = conn.execute(
                """
                SELECT t."id"
                FROM threads t JOIN steps s ON s."threadId" = t."id"
                WHERE t."userIdentifier" = ?
                  AND t."id" != ?
                  AND s."type" IN ('user_message', 'assistant_message')
                  AND (
                      ? IS NULL
                      OR thread_course(t."metadata") = ?
                      OR (? AND thread_course(t."metadata") IS NULL)
                  )
                GROUP BY t."id"
                ORDER BY MAX(s."createdAt") DESC
                LIMIT 1
                """,
                (matricula, exclude_thread_id or "", course, course, int(include_untagged)),
            ).fetchone()

            if not row:
                return None

            thread_id = row[0]

            steps = conn.execute(
                """
                SELECT "type", "output" FROM steps
                WHERE "threadId" = ?
                  AND "type" IN ('user_message', 'assistant_message')
                ORDER BY "createdAt" ASC
                """,
                (thread_id,),
            ).fetchall()
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - no previous transcript is fine
        logger.warning(
            "Could not read previous thread for %s", matricula, exc_info=True
        )
        return None

    lines = [
        f"{'Student' if step_type == 'user_message' else 'Tutor'}: {output}"
        for step_type, output in steps
        if output
    ]

    if not lines:
        return None

    return thread_id, "\n".join(lines)
