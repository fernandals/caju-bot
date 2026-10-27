from typing import Literal, Optional

from langchain_core.documents import Document
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


# How the tutor shapes its replies for a student (agent/prompts.STYLE_INSTRUCTIONS).
Style = Literal["step_by_step", "examples", "exercises"]


class StudentProfile(BaseModel):
    """
    Slow-moving, cross-session picture of how one student learns. Loaded
    at the start of a conversation and NOT modified during it - refreshed
    once per session by the profiler (agent/profiler.py) over the
    student's previous conversation. Persisted per matrícula in
    utils/student_profile.py.
    """

    responds_to_guiding_questions: Literal["well", "poorly", "unknown"] = Field(
        default="unknown",
        description=(
            "Whether the Socratic 'guiding question' approach works for this "
            "student. 'poorly' = they disengage, keep asking to just be told, "
            "or get frustrated when not given the answer directly."
        ),
    )

    frustration_tendency: Literal["low", "medium", "high", "unknown"] = Field(
        default="unknown",
        description=(
            "This student's general disposition to get frustrated or stuck, "
            "across a whole conversation - not a single moment."
        ),
    )

    solid_topics: list[str] = Field(
        default_factory=list,
        description=(
            "Topics/subtopics the student has demonstrated a solid grasp of "
            "across sessions."
        ),
    )

    shaky_topics: list[str] = Field(
        default_factory=list,
        description=(
            "Topics/subtopics the student has repeatedly struggled with "
            "across sessions."
        ),
    )

    misconceptions: list[str] = Field(
        default_factory=list,
        description=(
            "Wrong beliefs the student has voiced about specific concepts, "
            "each short and concrete, naming the concept (e.g. "
            "'client-server: thinks the server starts the communication'). "
            "Only beliefs actually voiced, not guesses. Drop one once the "
            "student has clearly corrected it."
        ),
    )

    what_clicked: list[str] = Field(
        default_factory=list,
        description=(
            "Explanations, examples or kinds of question that visibly moved "
            "this student forward, naming the concept (e.g. 'pipe-filter: "
            "understood after the assembly-line example from the material')."
        ),
    )

    prior_knowledge: list[str] = Field(
        default_factory=list,
        description=(
            "What the student has shown they know OUTSIDE this course and "
            "can build on (e.g. 'programs in Python', 'knows basic "
            "networking'). Only what they mentioned or demonstrated."
        ),
    )

    tutor_note: str = Field(
        default="",
        description=(
            "2-3 sentences addressed to the tutor: how to teach this student "
            "well. This is the field the tutor's prompt actually reads."
        ),
    )

    sessions_observed: int = Field(
        default=0,
        description="How many past conversations have fed into this profile. System-managed.",
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How reliable this profile is (grows with sessions_observed). System-managed.",
    )

    last_updated: Optional[str] = Field(
        default=None,
        description="ISO timestamp of the last profiler update. System-managed.",
    )

    course: Optional[str] = Field(
        default=None,
        description=(
            "Course the topic lists and tutor_note refer to. System-managed "
            "(utils/student_profile.scope_to_course)."
        ),
    )

    stated_style: Style | Literal["unknown"] = Field(
        default="unknown",
        description=(
            "Explanation style the student CHOSE when asked in their first "
            "conversation. System-managed - set by the app, never by the "
            "profiler."
        ),
    )

    style_asked: bool = Field(
        default=False,
        description="Whether the student already answered (or skipped) the style question. System-managed.",
    )

class ArcTally(BaseModel):
    mastered: int = 0
    stuck: int = 0


class StyleTracker(BaseModel):
    """
    Which style the tutor uses with a student, and how guided arcs went in
    each (agent/style_tracking.py). Decided in code, not by the profiler:
    the student's choice sets it, and STYLE_SWITCH_STUCK_ARCS stuck arcs
    in a row move it to another style. Persisted per matrícula
    (utils/student_profile.py); unlike StudentProfile it changes during a
    conversation - a switch applies from the next arc on.
    """

    active: Style = "step_by_step"
    source: Literal["default", "chosen", "arcs"] = "default"
    record: dict[str, ArcTally] = Field(default_factory=dict)
    stuck_streak: int = 0


class LearningState(BaseModel):
    topic: Optional[str] = Field(
        default=None,
        description="Main subject or concept currently being discussed."
    )

    subtopic: Optional[str] = Field(
        default=None,
        description="More specific concept or sub-area related to the current topic."
    )

    intent: Literal[
        "learn",
        "review",
        "practice",
        "solve_problem",
        "exam_prep",
        "debug_confusion"
    ] = Field(
        default="learn",
        description="Current learning objective."
    )

    comprehension_level: Literal[
        "low",
        "medium",
        "high"
    ] = Field(
        default="medium",
        description="Estimated understanding of the current topic."
    )

    learning_progress: Literal[
        "stuck",
        "stable",
        "improving",
        "mastered"
    ] = Field(
        default="stable",
        description=(
            "Where the student stands on the current topic. 'mastered' = they "
            "have shown the core idea (explained it back, applied it, or "
            "answered the guiding questions correctly); 'improving' = "
            "partially right."
        )
    )

    current_difficulty: Optional[str] = Field(
        default=None,
        description=(
            "Most important misconception, confusion, "
            "or learning obstacle currently observed."
        )
    )

    open_question: Optional[str] = Field(
        default=None,
        description=(
            "Main unresolved question the student is trying to answer."
        )
    )

    frustration_level: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Estimated frustration/confusion level."
    )

class TeachingState(BaseModel):
    """
    Tracks where the conversation is within a multi-turn guided-teaching
    arc for the current topic. Advanced deterministically (see
    agent/teaching.py), not by an LLM call.
    """

    topic_anchor: tuple[Optional[str], Optional[str]] = Field(
        default=(None, None),
        description="(topic, subtopic) this teaching arc is tracking.",
    )

    mode: Literal["guided", "direct"] = Field(
        default="guided",
        description=(
            "'guided' = pace the response according to `stage` below. "
            "'direct' = skip the arc and answer straightforwardly (escape valve)."
        )
    )

    stage: Literal["introduce", "check", "deepen", "wrap_up", "refer", "concluded", "not_covered"] = Field(
        default="introduce",
        description=(
            "'introduce' = orient briefly, end with one guiding question. "
            "'check' = evaluate the student's reply to that question. "
            "'deepen' = give the full grounded explanation. "
            "'wrap_up' = after an explanation: brief recap and one question "
            "checking it landed. "
            "'refer' = one-turn nudge toward the source material instead of "
            "answering directly, before conceding to 'deepen'. "
            "'concluded' = the student showed they have the topic; it is not "
            "reopened while the topic stays the same. "
            "'not_covered' = the course material has nothing on the topic "
            "(set after retrieval, agent/nodes.generate_answer)."
        )
    )

    turns_in_stage: int = Field(
        default=0,
        description="How many consecutive turns have been spent in the current stage.",
    )

    # Conversation-level, carried across topic changes (agent/teaching.py,
    # in-session adaptation): how guided arcs ended so far in THIS
    # conversation.
    session_arcs_stuck: int = Field(
        default=0,
        description="Guided arcs in this conversation that ended because the student was stuck.",
    )

    session_arcs_mastered: int = Field(
        default=0,
        description="Guided arcs in this conversation that ended with the student mastering the topic.",
    )

    # Whether this arc has had a full explanation (deepen / direct mode):
    # exam prep then stops re-explaining, and one right answer may close.
    explained: bool = Field(
        default=False,
        description="This arc already gave the full explanation.",
    )

    # The pending exercise was one the student asked for (strategy
    # exercise_first), not a task inside the guided arc ("exercises" style).
    exercise_requested: bool = Field(
        default=False,
        description="The last reply gave an exercise the student asked for.",
    )

    # Per turn, from agent/teaching.advance_teaching_state: whether a right
    # reply this turn may close the topic (agent/nodes.generate_answer).
    may_close: bool = Field(
        default=False,
        description="A fully correct reply this turn concludes the topic.",
    )

    # The exercise loop (agent/teaching.continue_exercise): set by
    # plan_instruction when a reply poses an exercise, so the next reply
    # checks the student's attempt instead of moving on.
    exercise_pending: bool = Field(
        default=False,
        description="The last reply gave the student an exercise to attempt.",
    )

    reviewing_attempt: bool = Field(
        default=False,
        description="This reply answers a message that may be the student's attempt at that exercise.",
    )

class AnswerPlan(BaseModel):
    needs_retrieval: bool = Field(
        default=True,
        description=(
            "Whether external instructional context retrieval is necessary "
            "to answer the student's question effectively."
        )
    )

    strategy: Literal[
        "direct_answer",
        "guided_teaching",
        "exercise_first",
        "hint_only",
        "step_by_step"
    ] = Field(
        description=(
            "Pedagogical strategy selected for the response. "
            "'direct_answer' = concise explanation, "
            "'guided_teaching' = progressive conceptual guidance, "
            "'exercise_first' = encourage reasoning before explanation, "
            "'hint_only' = provide minimal guidance, "
            "'step_by_step' = explicit sequential reasoning."
        )
    )

    response_depth: Literal[
        "light",
        "medium",
        "deep"
    ] = Field(
        default="light",
        description=(
            "Desired explanation depth. "
            "'light' = brief and concise, "
            "'medium' = balanced explanation, "
            "'deep' = detailed instructional explanation."
        )
    )

    include_examples: bool = Field(
        default=False,
        description="Whether the final response should include illustrative examples."
    )

    include_exercises: bool = Field(
        default=False,
        description="Whether the final response should include exercises or practice questions."
    )

    include_analogies: bool = Field(
        default=False,
        description="Whether analogies or intuitive comparisons would help the explanation."
    )

    confidence: float = Field(
        default=0.5,
        description="Confidence score (0-1) representing reliability of the instructional plan."
    )

    rationale: str = Field(
        description=("Short explanation of why this plan was selected.")
    )

class ReplyVerdict(BaseModel):
    """The tutor's judgement of the student's reply (agent/nodes.judge_reply)."""

    reason: str = Field(description="One short sentence: what the reply gets right or wrong.")
    verdict: Literal["correct", "partial", "wrong", "none"] = Field(
        description=(
            "'correct' = answers what was asked, right by the material; "
            "'partial' = partly right; 'wrong' = wrong; "
            "'none' = not an attempt (a question, 'não sei', small talk)."
        )
    )
    shows_core_idea: bool = Field(
        default=False,
        description=(
            "Whether the student's replies in this conversation, taken "
            "together, show they grasp the central idea of the topic - not "
            "just one small step or what its name suggests."
        ),
    )


class ChunkEvidence(BaseModel):
    """
    Evidence extracted from a retrieved chunk.
    """
    doc_id: str = Field(
        default="",
        description=(
            "Reference identifier (DOC_1, DOC_2...). Populated by the "
            "system after extraction; leave as-is."
        )
    )

    citation: str = Field(
        default="",
        description=(
            "Exact citation that MUST be used when information from this "
            "chunk appears in the final answer. Populated by the system "
            "from document metadata; leave as-is."
        )
    )

    evidence: list[str] = Field(
        default_factory=list,
        description="Atomic factual statements directly supported by this chunk."
    )

class TutorConfig(BaseModel):
    subject: str = Field(
        description="Main academic subject or course domain of the tutoring system."
    )

    allow_direct_answers: bool = Field(
        default=True,
        description=(
            "Whether the student can shortcut the guided-teaching arc by "
            "explicitly asking for a direct answer (e.g. 'just tell me'). "
            "Guided-first is always the default pacing; this only gates "
            "that one explicit-request escape hatch — set False for a "
            "course that wants guidance enforced with no shortcuts. Does "
            "not affect the frustration/exam_prep escape valves, which "
            "always apply regardless."
        )
    )

    course_level: str = Field(
        default="beginner",
        description=(
            "Overall instructional difficulty level of the course. "
            "Examples: beginner, intermediate, advanced."
        )
    )

    answer_language: str = Field(
        default="Português",
        description="Language in which the tutor should generate responses."
    )

    max_sentences: int = Field(
        default=6,
        description="Approximate maximum number of sentences expected in the tutor response."
    )

    course_outline: list[str] = Field(
        default_factory=list,
        description=(
            "The course's main topics, in teaching order (agent/outline.py) - "
            "what the tutor lists when asked what can be studied. Empty "
            "when it couldn't be built."
        )
    )

class TutorState(MessagesState):
    student_profile: StudentProfile

    learning_state: LearningState = LearningState()  # type: ignore

    teaching_state: TeachingState = TeachingState()  # type: ignore

    retrieved_docs: list[Document]

    answer_plan: AnswerPlan

    # True only on turns that took the greeting fast path (no planning /
    # retrieval / generation). Persists in the session state, so every
    # node that can run a real turn must set it back to False.
    greeted: bool

    # What the student profile did on this turn (set by plan_instruction,
    # recorded by utils/metrics.py for the personalization study):
    # {"levers": [...], "pacing_changed": bool, "plan_changed": bool,
    #  "profile_in_prompt": bool, "familiarity": "solid" | "shaky" | "new"}
    personalization: dict

    # False in the study's "off" arm (utils/personalization.py): also turns
    # off adaptation that is not profile-based (in-session adaptation in
    # agent/teaching.py). Missing = on (CLI, tests).
    personalization_on: bool

    # Whether this turn starts a topic (agent/nodes.plan_instruction) - a new
    # topic with nothing in the material is not taught (stage not_covered).
    new_arc: bool

    # The style tracker (agent/style_tracking.py); None in the study's
    # "off" arm and outside the app - no style, nothing tracked.
    style_tracker: Optional[StyleTracker]

    # What the profile knows about THIS topic (misconceptions, what worked
    # before, an analogy bridge), picked in code by plan_instruction and
    # repeated to the generator right before the reply (agent/profile_notes.py).
    profile_notes: list[str]

    evidence: list[ChunkEvidence] = Field(default_factory=list) # type: ignore