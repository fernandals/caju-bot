from agent.familiarity import Familiarity
from agent.state import LearningState, StudentProfile, TeachingState

# Inclusive (>=): the tracker emits round values (0.6, 0.7), and an
# exclusive check let a clearly impatient student sit at exactly 0.6
# forever without escaping (see evals/tracking_eval.py).
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

# The tutor's own decision to stop guiding and explain (the student is
# "stuck", or the guided-turn cap below is reached) never happens before
# the student has replied to this many guiding questions - whatever the
# profile says. The escape valves (frustration, exam_prep) and the
# student's explicit request (planning's "refer" path) are not the tutor
# giving up, and still apply at any point.
MIN_GUIDED_ATTEMPTS = 2

# How many consecutive guided turns (student replies to guiding questions)
# to spend on one topic before conceding to a full explanation. While the
# student stays engaged and unfrustrated the arc loops in "check"; this is
# the ceiling on that loop. The profile can lower it (guided_turn_cap):
GUIDED_TURN_CAP = 5

# ...to this on a topic the student has struggled with before: a long
# Socratic loop on ground that already failed tends to frustrate, so guide
# briefly, then explain step by step (agent/plan_profile.py).
GUIDED_TURN_CAP_SHAKY = 3

# ...and to the minimum for a student known not to benefit from guiding
# questions.
GUIDED_TURN_CAP_POOR_GUIDING = MIN_GUIDED_ATTEMPTS

# In-session adaptation (no profile needed): once this many guided arcs in
# the CURRENT conversation have ended with the student stuck, and none
# with the topic mastered, guiding is not working today - the cap drops to
# the minimum for the rest of the conversation.
SESSION_STUCK_ARCS = 2


def profile_confident(profile: StudentProfile | None) -> bool:
    return profile is not None and profile.confidence >= PROFILE_CONFIDENCE_FLOOR


def _frustration_threshold(profile: StudentProfile | None) -> float:
    if profile_confident(profile) and profile.frustration_tendency == "high":  # type: ignore[union-attr]
        return FRUSTRATION_ESCAPE_THRESHOLD_SENSITIVE

    return FRUSTRATION_ESCAPE_THRESHOLD


def _guiding_questions_ineffective(profile: StudentProfile | None) -> bool:
    return (
        profile_confident(profile)
        and profile.responds_to_guiding_questions == "poorly"  # type: ignore[union-attr]
    )


def effective_style(profile: StudentProfile | None) -> tuple[str, str | None]:
    """
    (explanation style to apply, where it came from: "observed" /
    "stated" / None). The profiler's observed style wins once the profile
    is confident; before that, what the student chose when asked in their
    first conversation applies right away - it is their own answer, not
    an inference that needs evidence.
    """
    if profile is None:
        return "unknown", None

    if profile_confident(profile) and profile.explanation_style != "unknown":
        return profile.explanation_style, "observed"

    if profile.stated_style != "unknown":
        return profile.stated_style, "stated"

    return "unknown", None


def session_struggling(teaching_state: TeachingState) -> bool:
    """Guiding has repeatedly failed in this conversation."""
    return (
        teaching_state.session_arcs_stuck >= SESSION_STUCK_ARCS
        and teaching_state.session_arcs_mastered == 0
    )


def effective_familiarity(
    familiarity: Familiarity, profile: StudentProfile | None
) -> Familiarity:
    """Topic familiarity counts only once the profile is confident."""
    return familiarity if profile_confident(profile) else "new"


def guided_turn_cap(
    profile: StudentProfile | None,
    familiarity: Familiarity = "new",
    struggling_this_session: bool = False,
) -> int:
    """Guided turns before conceding, for this student on this topic."""
    cap = GUIDED_TURN_CAP

    if struggling_this_session:
        cap = min(cap, MIN_GUIDED_ATTEMPTS)

    if effective_familiarity(familiarity, profile) == "shaky":
        cap = min(cap, GUIDED_TURN_CAP_SHAKY)

    if _guiding_questions_ineffective(profile):
        cap = min(cap, GUIDED_TURN_CAP_POOR_GUIDING)

    return max(cap, MIN_GUIDED_ATTEMPTS)


def advance_teaching_state(
    previous: TeachingState,
    learning_state: LearningState,
    student_profile: StudentProfile | None = None,
    familiarity: Familiarity = "new",
    adapt_in_session: bool = True,
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

    `student_profile` (loaded once per session, never mutated here) and
    `familiarity` (agent/familiarity.py: is the current topic in the
    profile's solid/shaky lists?) tune the arc when the profile is
    confident enough:
    - a frustration-prone student escapes to a direct answer sooner;
    - the guided-turn cap drops to GUIDED_TURN_CAP_SHAKY on a shaky topic
      and to the minimum for a student who does poorly with guiding
      questions - never below MIN_GUIDED_ATTEMPTS;
    - on a solid topic one correct reply is enough to recap and move on.

    `adapt_in_session` (off in the study's "off" arm): once guiding has
    ended stuck SESSION_STUCK_ARCS times in this conversation with nothing
    mastered, the cap drops to the minimum for the rest of it. The
    counters live on the TeachingState and survive topic changes.
    """

    anchor = (learning_state.topic, learning_state.subtopic)

    # Escape valves are about the student, not the topic, so they apply on
    # a topic change too: a student cramming for an exam (or already
    # frustrated - the tracker carries frustration across topics) should
    # not get a Socratic opener on every new topic before being answered.
    escape = (
        learning_state.frustration_level >= _frustration_threshold(student_profile)
        or learning_state.intent in DIRECT_INTENTS
    )

    if anchor != previous.topic_anchor:
        return TeachingState(
            topic_anchor=anchor,
            mode="direct" if escape else "guided",
            stage="deepen" if escape else "introduce",
            turns_in_stage=0,
            session_arcs_stuck=previous.session_arcs_stuck,
            session_arcs_mastered=previous.session_arcs_mastered,
        )

    if escape:
        return previous.model_copy(
            update={"mode": "direct", "stage": "deepen"}
        )

    if previous.stage == "refer":
        # Last turn we pointed at the material instead of answering
        # directly (see plan_instruction's direct_answer override). What
        # the student did with that nudge decides what happens now.

        if learning_state.learning_progress == "mastered":
            return previous.model_copy(
                update={"mode": "guided", "stage": "wrap_up", "turns_in_stage": 0}
            )

        if learning_state.learning_progress == "improving":
            # They engaged with the pointer rather than insisting - resume
            # the guided arc.
            return previous.model_copy(
                update={"mode": "guided", "stage": "check", "turns_in_stage": 0}
            )

        # stable or stuck: they did not engage with the pointer - one nudge
        # was enough, concede to the full explanation.
        return previous.model_copy(
            update={"mode": "guided", "stage": "deepen", "turns_in_stage": 0}
        )

    if previous.stage in ("introduce", "check"):
        # The student's reply to the last guiding question is what just
        # updated `learning_state` in the tracking node this same turn, so
        # it already reflects how well they engaged with THIS reply.

        guided_turns = previous.turns_in_stage + 1
        familiar = effective_familiarity(familiarity, student_profile)

        # They've demonstrated the concept back -> recap and move on - but
        # NOT on the student's first reply within the arc, mirroring the
        # `stuck` rule below. The "introduce" question is deliberately
        # easy (often just decoding the concept's name), so one right
        # answer to it is not evidence of understanding the topic; the
        # tracker does label that "mastered" once it is allowed to. One
        # more question confirms it; if they have really got it the label
        # persists and this fires on the next reply. Exception: a topic the
        # profile says they already master - one right answer confirms it.
        replies_to_confirm_mastery = 1 if familiar == "solid" else 2

        if (
            learning_state.learning_progress == "mastered"
            and guided_turns >= replies_to_confirm_mastery
        ):
            return previous.model_copy(
                update={
                    "mode": "guided",
                    "stage": "wrap_up",
                    "turns_in_stage": 0,
                    "session_arcs_mastered": previous.session_arcs_mastered + 1,
                }
            )

        # `stuck` concedes to the full explanation - but NOT on the
        # student's first reply within the arc. The tracker (a small model)
        # over-labels "stuck"/"low" early, before there is real evidence of
        # repeated struggle, and one premature "stuck" should not collapse
        # the whole guided arc. Give at least one guiding nudge first; if
        # they really are stuck the label persists (and frustration
        # typically rises, tripping the escape valve above).
        stuck_after_a_nudge = (
            learning_state.learning_progress == "stuck"
            and guided_turns >= MIN_GUIDED_ATTEMPTS
        )

        # Concede to the full explanation only when guiding has genuinely
        # stalled or has gone on longer than is productive for this
        # student on this topic. A student who is still engaging - even
        # partially - and is not frustrated (the frustration escape valve
        # above already handled that) keeps being guided rather than
        # handed the answer.
        concede = stuck_after_a_nudge or guided_turns >= guided_turn_cap(
            student_profile,
            familiar,
            struggling_this_session=adapt_in_session and session_struggling(previous),
        )

        if concede:
            return previous.model_copy(
                update={
                    "mode": "guided",
                    "stage": "deepen",
                    "turns_in_stage": 0,
                    "session_arcs_stuck": previous.session_arcs_stuck + int(stuck_after_a_nudge),
                }
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


def personalization_trace(
    previous: TeachingState,
    learning_state: LearningState,
    student_profile: StudentProfile | None,
    proposed: TeachingState,
    familiarity: Familiarity = "new",
    adapt_in_session: bool = True,
) -> dict:
    """
    What the student profile did to THIS turn's pacing, for the
    personalization study (utils/metrics.py records it per turn).

    - levers: the profile-driven pacing rules in play this turn
      ("early_frustration_escape" = the lowered threshold is what tripped
      the escape; "poor_guiding_short_arc" / "shaky_topic_short_arc" = a
      lowered guided-turn cap; "solid_topic_fast_mastery" = one correct
      reply is enough on this topic; "session_short_arc" = guiding kept
      failing in this conversation, so the cap is at the minimum).
    - pacing_changed: whether `proposed` differs from what the same turn
      would have produced with no profile. One-step counterfactual only:
      once pacing diverges, later turns start from different states.
    """
    levers: list[str] = []

    frustration = learning_state.frustration_level
    if (
        _frustration_threshold(student_profile) <= frustration < FRUSTRATION_ESCAPE_THRESHOLD
        and learning_state.intent not in DIRECT_INTENTS
    ):
        levers.append("early_frustration_escape")

    if _guiding_questions_ineffective(student_profile):
        levers.append("poor_guiding_short_arc")

    familiar = effective_familiarity(familiarity, student_profile)
    if familiar == "shaky":
        levers.append("shaky_topic_short_arc")
    elif familiar == "solid":
        levers.append("solid_topic_fast_mastery")

    if adapt_in_session and session_struggling(previous):
        levers.append("session_short_arc")

    baseline = advance_teaching_state(previous, learning_state, None, adapt_in_session=False)

    return {
        "levers": levers,
        "pacing_changed": proposed != baseline,
    }
