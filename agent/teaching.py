from agent.state import LearningState, StudentProfile, TeachingState

FRUSTRATION_ESCAPE_THRESHOLD = 0.6

# A lower escape threshold for students the profiler has flagged as
# frustrating easily - concede to a direct explanation sooner for them.
FRUSTRATION_ESCAPE_THRESHOLD_SENSITIVE = 0.45

# Below this profile confidence we ignore the personalization signals and
# use the default pacing - a barely-formed profile shouldn't bend the arc.
PROFILE_CONFIDENCE_FLOOR = 0.4

# Intents that skip the guided arc and go straight to a full explanation.
# Only exam_prep: the student is cramming against a deadline, so Socratic
# pacing works against them. "practice" and "solve_problem" are NOT here -
# those students want to do the work themselves, so they stay guided and
# the planning node picks an exercise_first / hint_only strategy.
DIRECT_INTENTS = {"exam_prep"}

# How many consecutive guided turns (the "introduce" turn plus "check"
# nudges) to spend on one topic before conceding to a full explanation.
# While the student stays engaged and unfrustrated the arc loops in
# "check"; this is the ceiling on that loop.
GUIDED_TURN_CAP = 5


def _frustration_threshold(profile: StudentProfile | None) -> float:
    if (
        profile is not None
        and profile.confidence >= PROFILE_CONFIDENCE_FLOOR
        and profile.frustration_tendency == "high"
    ):
        return FRUSTRATION_ESCAPE_THRESHOLD_SENSITIVE

    return FRUSTRATION_ESCAPE_THRESHOLD


def _guiding_questions_ineffective(profile: StudentProfile | None) -> bool:
    return (
        profile is not None
        and profile.confidence >= PROFILE_CONFIDENCE_FLOOR
        and profile.responds_to_guiding_questions == "poorly"
    )


def advance_teaching_state(
    previous: TeachingState,
    learning_state: LearningState,
    student_profile: StudentProfile | None = None,
) -> TeachingState:
    """
    Deterministically computes the next TeachingState from the just-updated
    LearningState. No LLM call: reuses signals `update_tracking` already
    infers every turn (comprehension_level, learning_progress,
    frustration_level, intent) instead of asking a model to judge pacing.

    Policy (see plan): guided-first with escape valves. While the student
    stays engaged and unfrustrated, the arc loops in "check" asking fresh
    guiding questions; it concedes to a full explanation only when the
    student is stuck, the guided turn cap is hit, or an escape valve fires
    (frustration / exam_prep).

    `student_profile` (loaded once per session, never mutated here) tunes
    two things when it is confident enough: a frustration-prone student
    escapes to a direct answer sooner, and a student who does poorly with
    guiding questions skips the one-nudge "check" step.
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
        learning_state.frustration_level > _frustration_threshold(student_profile)
        or learning_state.intent in DIRECT_INTENTS
    ):
        return previous.model_copy(
            update={"mode": "direct", "stage": "deepen"}
        )

    if previous.stage in ("introduce", "check"):
        # The student's reply to the last guiding question is what just
        # updated `learning_state` in the tracking node this same turn, so
        # it already reflects how well they engaged with THIS reply.

        # They've demonstrated the concept back -> recap and move on.
        if learning_state.learning_progress == "mastered":
            return previous.model_copy(
                update={"mode": "guided", "stage": "wrap_up", "turns_in_stage": 0}
            )

        guided_turns = previous.turns_in_stage + 1

        # `stuck` concedes to the full explanation - but NOT on the
        # student's first reply within the arc. The tracker (a small model)
        # over-labels "stuck"/"low" early, before there is real evidence of
        # repeated struggle, and one premature "stuck" should not collapse
        # the whole guided arc. Give at least one guiding nudge first; if
        # they really are stuck the label persists (and frustration
        # typically rises, tripping the escape valve above).
        stuck_after_a_nudge = (
            learning_state.learning_progress == "stuck" and guided_turns >= 2
        )

        # Concede to the full explanation only when guiding has genuinely
        # stalled, has gone on longer than is productive, or this student
        # is known not to benefit from guiding questions. A student who is
        # still engaging - even partially - and is not frustrated (the
        # frustration escape valve above already handled that) keeps being
        # guided rather than handed the answer.
        concede = (
            stuck_after_a_nudge
            or guided_turns >= GUIDED_TURN_CAP
            or _guiding_questions_ineffective(student_profile)
        )

        if concede:
            return previous.model_copy(
                update={"mode": "guided", "stage": "deepen", "turns_in_stage": 0}
            )

        # Still engaged, not there yet -> ask the next guiding question.
        return previous.model_copy(
            update={
                "mode": "guided",
                "stage": "check",
                "turns_in_stage": guided_turns,
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
