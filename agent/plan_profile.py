"""
Deterministic profile rules for the instructional plan - no LLM call.

The planner model (plan_instruction) never sees the student profile, and
the generator only gets it as a soft hint that the stage rules override.
So the student's style (agent/style_tracking.py) and the profile's topic
familiarity are applied here, in code, after the planner has run -
testable, and logged per turn for the personalization study.

The plan is only adjusted on explanation turns (direct mode, or the
"deepen" stage): the guided turns stay one question, and the profile must
not turn them into a lecture. The explanation style still shapes them -
turn_style picks the style block (agent/prompts.STYLE_INSTRUCTIONS) for
guided and explanation turns alike. Strategies the student chose by
intent (exercise_first / hint_only) are left alone.
"""

from agent.familiarity import Familiarity
from agent.state import AnswerPlan, LearningState, StudentProfile, StyleTracker, TeachingState
from agent.teaching import effective_familiarity

_STUDENT_DRIVEN_STRATEGIES = {"exercise_first", "hint_only"}


def is_explanation_turn(teaching_state: TeachingState) -> bool:
    return teaching_state.mode == "direct" or teaching_state.stage == "deepen"


def apply_profile_to_plan(
    plan: AnswerPlan,
    teaching_state: TeachingState,
    learning_state: LearningState,
    profile: StudentProfile | None,
    familiarity: Familiarity = "new",
    tracker: StyleTracker | None = None,
) -> tuple[AnswerPlan, list[str]]:
    """
    (adjusted plan, levers that changed it). Levers:
    - shaky_topic_steps: topic struggled with before -> numbered steps
    - style_steps: step_by_step student -> numbered steps
    - style_examples: examples student -> include an example
    - style_exercises: exercises student -> include an exercise
    - arc_style: added next to a style_* lever when the style is one the
      tutor moved to after stuck arcs, not the student's choice
      (agent/style_tracking.py)
    """
    if not is_explanation_turn(teaching_state) or plan.strategy in _STUDENT_DRIVEN_STRATEGIES:
        return plan, []

    style = tracker.active if tracker else None
    updates: dict = {}
    levers: list[str] = []

    if plan.strategy != "step_by_step":
        if effective_familiarity(familiarity, profile) == "shaky":
            updates["strategy"] = "step_by_step"
            levers.append("shaky_topic_steps")
        elif style == "step_by_step":
            updates["strategy"] = "step_by_step"
            levers.append("style_steps")

    if style == "examples" and not plan.include_examples:
        updates["include_examples"] = True
        levers.append("style_examples")

    if style == "exercises" and not plan.include_exercises:
        updates["include_exercises"] = True
        levers.append("style_exercises")

    if not updates:
        return plan, []

    if tracker and tracker.source == "arcs" and any(l.startswith("style_") for l in levers):
        levers.append("arc_style")

    return plan.model_copy(update=updates), levers


# Guided stages whose reply is a guiding question. "refer" (point to the
# material) and "wrap_up" (recap) are short nudges, left as is.
_GUIDING_STAGES = {"introduce", "check"}


def turn_style(
    plan: AnswerPlan, teaching_state: TeachingState, tracker: StyleTracker | None
) -> tuple[str, str] | None:
    """
    (style, "guiding" / "explaining") - which style block this turn's reply
    gets (agent/prompts.STYLE_INSTRUCTIONS) - or None: no tracker (the
    study's "off" arm), a strategy the student chose, or a refer / wrap_up
    turn.
    """
    if tracker is None or plan.strategy in _STUDENT_DRIVEN_STRATEGIES:
        return None

    style = tracker.active

    if is_explanation_turn(teaching_state):
        return style, "explaining"

    if teaching_state.stage in _GUIDING_STAGES:
        return style, "guiding"

    return None
