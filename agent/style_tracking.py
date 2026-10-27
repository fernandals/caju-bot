"""
Deterministic style tracking - no LLM call.

The style a student picks (first-conversation question, or the settings
panel) is where the tutor starts, not where it stays. Every guided arc
that ends - the student mastered the topic, or got stuck - is tallied
under the style in use; after STYLE_SWITCH_STUCK_ARCS stuck arcs in a row
the tutor moves to another style. Nothing changes without that evidence:
the tutor never tries another style on its own, and the student can
always pick one back.
"""

from typing import Literal, get_args

from agent.state import ArcTally, Style, StudentProfile, StyleTracker, TeachingState

STYLES: tuple[Style, ...] = get_args(Style)

# Stuck arcs in a row, in the current style, before switching.
STYLE_SWITCH_STUCK_ARCS = 3

ArcOutcome = Literal["mastered", "stuck"]


def start_tracker(profile: StudentProfile) -> StyleTracker:
    """A student's first tracker: the style they chose, else step by step."""
    if profile.stated_style != "unknown":
        return choose_style(StyleTracker(), profile.stated_style)
    return StyleTracker()


def choose_style(tracker: StyleTracker, style: Style) -> StyleTracker:
    """The student picked a style: use it, with a clean streak."""
    return tracker.model_copy(update={"active": style, "source": "chosen", "stuck_streak": 0})


def arc_outcome(previous: TeachingState, current: TeachingState) -> ArcOutcome | None:
    """
    How a guided arc ended this turn, from the arc counters
    (agent/teaching.py). Pacing only ever ends one stuck: a mastered arc is
    closed after the reply, by the tutor's verdict, and recorded there
    (agent/nodes.apply_verdict).
    """
    if current.session_arcs_mastered > previous.session_arcs_mastered:
        return "mastered"
    if current.session_arcs_stuck > previous.session_arcs_stuck:
        return "stuck"
    return None


def _score(tally: ArcTally | None) -> int:
    return (tally.mastered - tally.stuck) if tally else 0


def next_style(tracker: StyleTracker) -> Style:
    """
    The style to move to: the best record among the others (mastered minus
    stuck arcs; an untried style scores 0), ties in STYLES order.
    """
    others = [s for s in STYLES if s != tracker.active]
    return max(others, key=lambda s: (_score(tracker.record.get(s)), -others.index(s)))


def record_arc(tracker: StyleTracker, outcome: ArcOutcome) -> tuple[StyleTracker, Style | None]:
    """
    (updated tracker, the style switched to - or None). The arc counts for
    the style that was in use during it.
    """
    tally = tracker.record.get(tracker.active, ArcTally())
    tally = tally.model_copy(update={outcome: getattr(tally, outcome) + 1})
    record = {**tracker.record, tracker.active: tally}

    streak = tracker.stuck_streak + 1 if outcome == "stuck" else 0
    updated = tracker.model_copy(update={"record": record, "stuck_streak": streak})

    if streak < STYLE_SWITCH_STUCK_ARCS:
        return updated, None

    switched_to = next_style(updated)
    return updated.model_copy(update={"active": switched_to, "source": "arcs", "stuck_streak": 0}), switched_to
