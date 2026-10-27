import re

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


# "me passa um exercício", "quero mais exercícios", "quero praticar": the
# student asks to practise. Decided here rather than left to the planner,
# which answered the quick-reply button's own message with guided teaching
# - and an exercise not marked as one is not reviewed as an attempt.
_EXERCISE_REQUEST_RE = re.compile(
    r"\b(?:passa|passe|passar|d[áa]|manda|mande|quero|queria|gostaria\s+de|prefiro|vamos|bora|"
    r"faz|fa[çc]a|cria|crie|prop[õo]e|proponha|tem|teria)\b[\w\s,]{0,40}?\b(?:exerc[íi]cios?|praticar)\b",
    re.IGNORECASE,
)


def is_exercise_request(text: str | None) -> bool:
    """The student's message asks for an exercise / to practise."""
    return bool(text) and bool(_EXERCISE_REQUEST_RE.search(text))


def is_course_review(learning_state: LearningState | None) -> bool:
    """
    Exam prep with no topic named ("tenho prova, me ajuda a revisar?" -
    the review starter button): a review of the whole course, answered as
    a study guide over the course outline with as many citations as
    possible (agent/nodes: retrieve_documents, assess_documents,
    resolve_teaching_instructions). Exam prep on a named topic stays a
    regular direct-mode explanation.
    """
    return (
        learning_state is not None
        and learning_state.intent == "exam_prep"
        and not learning_state.topic
    )

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


def continue_exercise(previous: TeachingState, proposed: TeachingState) -> TeachingState:
    """
    The reply after an exercise reviews the student's attempt at it.

    An exercise is often asked for with no topic named ("me passa um
    exercício"), and the attempt is what finally names one - which
    advance_teaching_state reads as a topic change and restarts the arc
    at "introduce" (do not explain anything, just ask), so a wrong attempt
    went uncorrected. With an exercise pending, a newly named or refined
    topic stays in the same arc, at "check" (react to what they did) - as
    does an exercise asked for on a concluded topic. A different topic
    altogether is still a topic change.
    """
    if not previous.exercise_pending:
        return proposed

    reviewing = proposed.model_copy(update={"reviewing_attempt": True})

    previous_topic = previous.topic_anchor[0]
    same_arc = previous_topic is None or previous_topic == proposed.topic_anchor[0]

    # An exercise asked for on a concluded topic is judged like any other.
    # A worked exercise is evidence enough on its own: right, it may close.
    # (A task posed inside the guided arc - the "exercises" style - is
    # already at "check" and keeps the arc's own may_close.)
    if proposed.mode == "guided" and proposed.stage in ("introduce", "concluded") and same_arc:
        reviewing = reviewing.model_copy(update={"stage": "check", "turns_in_stage": 0, "may_close": True})

    # The same for an exercise asked for mid-arc.
    if previous.exercise_requested and reviewing.mode == "guided" and reviewing.stage == "check":
        reviewing = reviewing.model_copy(update={"may_close": True})

    return reviewing


def close_arc(teaching_state: TeachingState) -> TeachingState:
    """
    The student showed they have the topic (agent/nodes.generate_answer,
    from the tutor's verdict on their reply): the arc ends mastered and the
    topic is concluded - it is not reopened while the topic stays the same.
    """
    return teaching_state.model_copy(
        update={
            "mode": "guided",
            "stage": "concluded",
            "turns_in_stage": 0,
            "may_close": False,
            "exercise_pending": False,
            "session_arcs_mastered": teaching_state.session_arcs_mastered + 1,
        }
    )


def advance_teaching_state(
    previous: TeachingState,
    learning_state: LearningState,
    student_profile: StudentProfile | None = None,
    familiarity: Familiarity = "new",
    adapt_in_session: bool = True,
) -> TeachingState:
    """
    Deterministically advances the guided-teaching arc from the previous
    TeachingState and the freshly updated LearningState (learning_progress,
    frustration_level, intent) instead of asking a model to judge pacing.

    Policy: guided-first with escape valves. While the student stays
    engaged and unfrustrated, the arc loops in "check" asking fresh
    guiding questions; it concedes to a full explanation only when the
    student is stuck, the guided turn cap is hit, or an escape valve fires
    (frustration / exam_prep). After an explanation ("deepen") comes a
    recap with one question checking it landed ("wrap_up"), then "check"
    again.

    Whether the student has GOT the topic is not decided here: in "check"
    the tutor judges the reply against the material and, when `may_close`
    allows it, closes the arc (close_arc, called from
    agent/nodes.generate_answer). The tracker's own "mastered" - judged
    without the material - only opens that door at the turn cap. A
    concluded topic stays concluded until the topic changes.

    `student_profile` (loaded once per session, never mutated here) and
    `familiarity` (agent/familiarity.py: is the current topic in the
    profile's solid/shaky lists?) tune the arc when the profile is
    confident enough:
    - a frustration-prone student escapes to a direct answer sooner;
    - the guided-turn cap drops to GUIDED_TURN_CAP_SHAKY on a shaky topic
      and to the minimum for a student who does poorly with guiding
      questions - never below MIN_GUIDED_ATTEMPTS;
    - on a solid topic one correct reply is enough to close.

    `adapt_in_session` (off in the study's "off" arm): once guiding has
    ended stuck SESSION_STUCK_ARCS times in this conversation with nothing
    mastered, the cap drops to the minimum for the rest of it. The
    counters live on the TeachingState and survive topic changes.
    """

    anchor = (learning_state.topic, learning_state.subtopic)

    frustrated = learning_state.frustration_level >= _frustration_threshold(student_profile)
    exam = learning_state.intent in DIRECT_INTENTS

    # Escape valves are about the student, not the topic, so they apply on
    # a topic change too: a student cramming for an exam (or already
    # frustrated - the tracker carries frustration across topics) should
    # not get a Socratic opener on every new topic before being answered.
    if anchor != previous.topic_anchor:
        escape = frustrated or exam
        return TeachingState(
            topic_anchor=anchor,
            mode="direct" if escape else "guided",
            stage="deepen" if escape else "introduce",
            turns_in_stage=0,
            explained=escape,
            session_arcs_stuck=previous.session_arcs_stuck,
            session_arcs_mastered=previous.session_arcs_mastered,
        )

    # Not reopened: a follow-up on a concluded topic is consolidation.
    if previous.stage == "concluded":
        return previous.model_copy(
            update={"mode": "guided", "turns_in_stage": previous.turns_in_stage + 1, "may_close": False}
        )

    # Exam prep explains once per topic: once it has, the student answering
    # the check question is judged like anyone else - otherwise every right
    # answer got the whole explanation again and the topic never closed.
    if frustrated or (exam and not previous.explained):
        return previous.model_copy(
            update={"mode": "direct", "stage": "deepen", "explained": True, "may_close": False}
        )

    if previous.stage == "refer":
        # Last turn we pointed at the material instead of answering
        # directly (see plan_instruction's direct_answer override). What
        # the student did with that nudge decides what happens now.

        if learning_state.learning_progress in ("improving", "mastered"):
            # They engaged with the pointer rather than insisting - resume
            # the guided arc; what they found is judged like any reply.
            return previous.model_copy(
                update={"mode": "guided", "stage": "check", "turns_in_stage": 0, "may_close": True}
            )

        # stable or stuck: they did not engage with the pointer - one nudge
        # was enough, concede to the full explanation.
        return previous.model_copy(
            update={"mode": "guided", "stage": "deepen", "turns_in_stage": 0, "explained": True, "may_close": False}
        )

    if previous.stage == "deepen":
        return previous.model_copy(
            update={"mode": "guided", "stage": "wrap_up", "turns_in_stage": 0, "may_close": False}
        )

    if previous.stage == "wrap_up":
        # The student's answer to the recap's check question - after a full
        # explanation, one right answer is enough.
        return previous.model_copy(
            update={"mode": "guided", "stage": "check", "turns_in_stage": 0, "may_close": True}
        )

    # introduce / check: the student's reply to the last guiding question
    # is what just updated `learning_state` in the tracking node this same
    # turn, so it already reflects how well they engaged with THIS reply.
    guided_turns = previous.turns_in_stage + 1
    familiar = effective_familiarity(familiarity, student_profile)

    # Replies before a right one may close the topic. The "introduce"
    # question is deliberately easy (often just decoding the concept's
    # name), so one right answer to it is not evidence of understanding;
    # one more question confirms it. Exception: a topic the profile says
    # they already master, or one already explained in this arc.
    replies_to_close = 1 if familiar == "solid" or previous.explained else 2

    # `stuck` concedes to the full explanation - but NOT on the student's
    # first reply within the arc. The tracker (a small model) over-labels
    # "stuck"/"low" early, before there is real evidence of repeated
    # struggle, and one premature "stuck" should not collapse the whole
    # guided arc. Give at least one guiding nudge first; if they really
    # are stuck the label persists (and frustration typically rises,
    # tripping the escape valve above).
    stuck_after_a_nudge = (
        learning_state.learning_progress == "stuck"
        and guided_turns >= MIN_GUIDED_ATTEMPTS
    )

    # Concede to the full explanation only when guiding has genuinely
    # stalled or has gone on longer than is productive for this student on
    # this topic - unless the tracker sees the student getting it, when
    # this reply gets its chance to close instead. A student who is still
    # engaging - even partially - and is not frustrated (the frustration
    # escape valve above already handled that) keeps being guided rather
    # than handed the answer.
    at_cap = guided_turns >= guided_turn_cap(
        student_profile,
        familiar,
        struggling_this_session=adapt_in_session and session_struggling(previous),
    )
    concede = stuck_after_a_nudge or (at_cap and learning_state.learning_progress != "mastered")

    if concede:
        return previous.model_copy(
            update={
                "mode": "guided",
                "stage": "deepen",
                "turns_in_stage": 0,
                "explained": True,
                "may_close": False,
                "session_arcs_stuck": previous.session_arcs_stuck + int(stuck_after_a_nudge),
            }
        )

    # Still engaged -> react to the reply; the tutor closes the topic if
    # it is right and enough replies are in.
    return previous.model_copy(
        update={
            "mode": "guided",
            "stage": "check",
            "turns_in_stage": guided_turns,
            "may_close": guided_turns >= replies_to_close,
        }
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
