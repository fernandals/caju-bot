"""
Analysis of the personalization study (utils/personalization.py): does a
conversation where the tutor uses the student's profile go better than one
where it doesn't?

Reads metrics.db (turn_metrics + study_sessions, see utils/metrics.py)
and, optionally, Chainlit's chainlit.db for the student's 👍/👎 on each
reply. Used by the professor's metrics tab (app.py) and by
scripts/export_study.py.

Outcome per conversation:
- mastered / turnsToMastered: did the student reach "mastered" on some
  topic, and at which of their turns. Fewer turns = understood sooner.
- concededShare: share of turns where the tutor gave up guiding and
  explained everything (deepen / direct). Lower = the guided approach
  worked more often.
- stuckShare, frustration (start / peak / end / mean).
- thumbsUp / thumbsDown.

All of these come from the tracker's own judgments (an LLM), not from an
external test - read them as signals, not as grades.
"""

import sqlite3
from pathlib import Path

import pandas as pd

from agent.teaching import PROFILE_CONFIDENCE_FLOOR

SESSION_COLUMNS = [
    "threadId",
    "matricula",
    "sessionIndex",
    "personalization",
    "profileConfidence",
    "createdAt",
    "turns",
    "mastered",
    "turnsToMastered",
    "concededShare",
    "stuckShare",
    "frustrationStart",
    "frustrationPeak",
    "frustrationEnd",
    "frustrationMean",
    "pacingChangedTurns",
    "planChangedTurns",
    "thumbsUp",
    "thumbsDown",
]

# Lower is better for these; higher is better for the rest.
LOWER_IS_BETTER = {
    "turnsToMastered",
    "concededShare",
    "stuckShare",
    "frustrationPeak",
    "frustrationEnd",
    "frustrationMean",
    "thumbsDown",
}

COMPARED_METRICS = [
    "mastered",
    "turnsToMastered",
    "concededShare",
    "stuckShare",
    "frustrationMean",
    "frustrationEnd",
    "thumbsUp",
    "thumbsDown",
]


def _read(db_path: Path, query: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


# Chainlit stores a 👍/👎 against the reply's PARENT run step (the
# on_message / action-callback run), not the reply message itself - so
# map it back to the assistant message under that run, whose id is what
# turn_metrics.messageId holds. The direct match is kept in case a
# Chainlit upgrade starts pointing at the message.
_FEEDBACK_QUERY = """
SELECT s."id" AS "messageId", f."value" AS "value"
FROM feedbacks f
JOIN steps s
  ON s."type" = 'assistant_message'
 AND (s."id" = f."forId" OR s."parentId" = f."forId")
"""


def load_feedback(chainlit_db: Path | None) -> pd.DataFrame:
    """messageId -> value (Chainlit: 1 = 👍, 0 = 👎). Empty if unavailable."""
    empty = pd.DataFrame(columns=["messageId", "value"])

    if chainlit_db is None or not Path(chainlit_db).is_file():
        return empty

    try:
        return _read(Path(chainlit_db), _FEEDBACK_QUERY)
    except Exception:  # noqa: BLE001 - feedback is optional
        return empty


def _session_outcomes(turns: pd.DataFrame) -> dict:
    turns = turns.sort_values("turnIndex")
    frustration = turns["frustrationLevel"].dropna()
    mastered_at = turns.loc[turns["learningProgress"] == "mastered", "turnIndex"]
    conceded = (turns["teachingStage"] == "deepen") | (turns["teachingMode"] == "direct")

    return {
        "turns": len(turns),
        "mastered": bool(len(mastered_at)),
        "turnsToMastered": mastered_at.min() if len(mastered_at) else None,
        "concededShare": conceded.mean(),
        "stuckShare": (turns["learningProgress"] == "stuck").mean(),
        "frustrationStart": frustration.iloc[0] if len(frustration) else None,
        "frustrationPeak": frustration.max() if len(frustration) else None,
        "frustrationEnd": frustration.iloc[-1] if len(frustration) else None,
        "frustrationMean": frustration.mean() if len(frustration) else None,
        "pacingChangedTurns": int(turns["pacingChanged"].fillna(0).sum()),
        "planChangedTurns": int(turns["planChanged"].fillna(0).sum()),
        "thumbsUp": int((turns["feedback"] == 1).sum()),
        "thumbsDown": int((turns["feedback"] == 0).sum()),
    }


def build_sessions(
    metrics_db: Path, chainlit_db: Path | None = None, discipline: str | None = None
) -> pd.DataFrame:
    """
    One row per study conversation (SESSION_COLUMNS). Turns recorded before
    the study (no threadId) are ignored. With `discipline`, only that
    course's conversations - one volume can outlive a course, and a
    comparison should not mix two.
    """
    sessions = _read(metrics_db, "SELECT * FROM study_sessions")
    if discipline is not None:
        sessions = sessions[sessions["discipline"].isna() | (sessions["discipline"] == discipline)]
    turns = _read(
        metrics_db, 'SELECT * FROM turn_metrics WHERE "threadId" IS NOT NULL'
    )

    if sessions.empty or turns.empty:
        return pd.DataFrame(columns=SESSION_COLUMNS)

    feedback = load_feedback(chainlit_db)
    # A student can change their rating; any one per message is fine here.
    feedback = feedback.drop_duplicates("messageId", keep="last")
    turns = turns.merge(feedback.rename(columns={"value": "feedback"}), on="messageId", how="left")

    outcomes = pd.DataFrame(
        [
            {"threadId": thread_id, **_session_outcomes(group)}
            for thread_id, group in turns.groupby("threadId")
        ]
    )

    merged = sessions.drop(columns=["profile"]).merge(outcomes, on="threadId", how="inner")
    return merged[SESSION_COLUMNS].sort_values(["matricula", "sessionIndex"]).reset_index(drop=True)


def informed_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Conversations that started with a profile confident enough to change
    the tutor's pacing (agent/teaching.PROFILE_CONFIDENCE_FLOOR). Before
    that, "on" and "off" behave almost the same, so including them dilutes
    the comparison.
    """
    return sessions[sessions["profileConfidence"].fillna(0) >= PROFILE_CONFIDENCE_FLOOR]


def compare_arms(sessions: pd.DataFrame) -> pd.DataFrame:
    """Mean of each outcome per arm (rows: metric; columns: on, off, n)."""
    if sessions.empty:
        return pd.DataFrame()

    data = sessions.assign(mastered=sessions["mastered"].astype(float))
    means = data.groupby("personalization")[COMPARED_METRICS].mean().T
    means = means.reindex(columns=["on", "off"])
    means["on − off"] = means["on"] - means["off"]
    means["better"] = [
        _better(metric, diff) for metric, diff in means["on − off"].items()
    ]
    return means


def _better(metric: str, diff) -> str:
    if pd.isna(diff) or diff == 0:
        return ""
    if (diff < 0) == (metric in LOWER_IS_BETTER):
        return "on"
    return "off"


def paired_by_student(sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Each student compared against themselves: per student who has
    conversations in BOTH arms, their mean "on" minus their mean "off" for
    each outcome. This is the comparison the alternation design is for -
    differences between students (some just learn faster) cancel out.
    """
    if sessions.empty:
        return pd.DataFrame()

    data = sessions.assign(mastered=sessions["mastered"].astype(float))
    per_arm = data.groupby(["matricula", "personalization"])[COMPARED_METRICS].mean()
    wide = per_arm.unstack("personalization")

    if "on" not in wide.columns.get_level_values(1) or "off" not in wide.columns.get_level_values(1):
        return pd.DataFrame()

    diff = wide.xs("on", axis=1, level=1) - wide.xs("off", axis=1, level=1)
    return diff.dropna(how="all")
