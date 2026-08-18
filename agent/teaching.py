from agent.state import LearningState, TeachingState

FRUSTRATION_ESCAPE_THRESHOLD = 0.6
DIRECT_INTENTS = {"exam_prep", "practice"}


def advance_teaching_state(
    previous: TeachingState,
    learning_state: LearningState,
) -> TeachingState:
    """
    Deterministically computes the next TeachingState from the just-updated
    LearningState. No LLM call: reuses signals `update_tracking` already
    infers every turn (comprehension_level, learning_progress,
    frustration_level, intent) instead of asking a model to judge pacing.

    Policy (see plan): guided-first with escape valves, and at most one
    nudge before conceding and giving the full explanation.
    """

    anchor = (learning_state.topic, learning_state.subtopic)

    if anchor != previous.topic_anchor:
        return TeachingState(
            topic_anchor=anchor,
            mode="guided",
            stage="introduce",
            turns_in_stage=0,
        )

    if (
        learning_state.frustration_level > FRUSTRATION_ESCAPE_THRESHOLD
        or learning_state.intent in DIRECT_INTENTS
    ):
        return previous.model_copy(
            update={"mode": "direct", "stage": "deepen"}
        )

    if previous.stage in ("introduce", "check"):
        # The student's reply to the guiding question (asked during
        # "introduce", or re-asked during a "check" nudge) is what just
        # updated `learning_state` in the tracking node this same turn, so
        # it already reflects how well they engaged with THIS reply -
        # evaluate it now rather than always spending one turn in "check"
        # regardless of quality.
        strong_engagement = (
            learning_state.learning_progress in ("improving", "mastered")
            or learning_state.comprehension_level == "high"
        )

        no_engagement_or_already_nudged = (
            learning_state.learning_progress == "stuck"
            or previous.turns_in_stage >= 1
        )

        if strong_engagement or no_engagement_or_already_nudged:
            return previous.model_copy(
                update={"mode": "guided", "stage": "deepen", "turns_in_stage": 0}
            )

        # Partial engagement, first miss: one nudge allowed.
        return previous.model_copy(
            update={
                "mode": "guided",
                "stage": "check",
                "turns_in_stage": previous.turns_in_stage + 1,
            }
        )

    if previous.stage == "deepen":
        return previous.model_copy(
            update={"mode": "guided", "stage": "wrap_up", "turns_in_stage": 0}
        )

    # wrap_up: same topic, student keeps engaging -> a fresh micro-cycle
    return previous.model_copy(
        update={"mode": "guided", "stage": "introduce", "turns_in_stage": 0}
    )
