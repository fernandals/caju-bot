"""
Deterministic profile rules for the instructional plan - no LLM call.

The planner model (plan_instruction) never sees the student profile, and
the generator only gets it as a soft hint that the stage rules override.
So the profile's explanation_style and topic familiarity are applied
here, in code, after the planner has run - testable, and logged per turn
for the personalization study.

Only on explanation turns (direct mode, or the "deepen" stage): the guided
turns are one question with no examples by design, and the profile must
not turn them into a lecture. Strategies the student chose by intent
(exercise_first / hint_only) are left alone.
"""

from agent.familiarity import Familiarity
from agent.state import AnswerPlan, LearningState, StudentProfile, TeachingState
from agent.teaching import effective_familiarity, profile_confident

_STUDENT_DRIVEN_STRATEGIES = {"exercise_first", "hint_only"}


def is_explanation_turn(teaching_state: TeachingState) -> bool:
    return teaching_state.mode == "direct" or teaching_state.stage == "deepen"


def _style(profile: StudentProfile | None) -> str:
    return profile.explanation_style if profile_confident(profile) else "unknown"  # type: ignore[union-attr]


def apply_profile_to_plan(
    plan: AnswerPlan,
    teaching_state: TeachingState,
    learning_state: LearningState,
    profile: StudentProfile | None,
    familiarity: Familiarity = "new",
) -> tuple[AnswerPlan, list[str]]:
    """
    (adjusted plan, levers that changed it). Levers:
    - shaky_topic_steps: topic struggled with before -> numbered steps
    - style_steps: step_by_step student -> numbered steps
    - style_examples: example_first student -> include an example
    - style_depth: concise caps depth at medium; detailed raises it to at
      least medium (deep when comprehension is low)
    """
    if not is_explanation_turn(teaching_state) or plan.strategy in _STUDENT_DRIVEN_STRATEGIES:
        return plan, []

    style = _style(profile)
    updates: dict = {}
    levers: list[str] = []

    if plan.strategy != "step_by_step":
        if effective_familiarity(familiarity, profile) == "shaky":
            updates["strategy"] = "step_by_step"
            levers.append("shaky_topic_steps")
        elif style == "step_by_step":
            updates["strategy"] = "step_by_step"
            levers.append("style_steps")

    if style == "example_first" and not plan.include_examples:
        updates["include_examples"] = True
        levers.append("style_examples")

    depth = plan.response_depth
    if style == "concise" and depth == "deep":
        depth = "medium"
    elif style == "detailed":
        depth = "deep" if learning_state.comprehension_level == "low" else (
            "medium" if depth == "light" else depth
        )

    if depth != plan.response_depth:
        updates["response_depth"] = depth
        levers.append("style_depth")

    if not updates:
        return plan, []

    return plan.model_copy(update=updates), levers


def leads_with_example(
    plan: AnswerPlan, teaching_state: TeachingState, profile: StudentProfile | None
) -> bool:
    """Whether the generator should open with the example (example_first)."""
    return (
        is_explanation_turn(teaching_state)
        and plan.include_examples
        and _style(profile) == "example_first"
    )
