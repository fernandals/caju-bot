from typing import Literal, Optional

from langchain_core.documents import Document
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class StudentProfile(BaseModel):
    """
    Slow-moving, cross-session picture of how one student learns. Loaded
    at the start of a conversation and NOT modified during it - refreshed
    once per session by the profiler (agent/profiler.py) over the
    student's previous conversation. Persisted per matrícula in
    utils/student_profile.py.
    """

    explanation_style: Literal[
        "concise", "detailed", "example_first", "step_by_step", "unknown"
    ] = Field(
        default="unknown",
        description=(
            "How this student best receives an explanation. "
            "'concise' = wants the short answer; "
            "'detailed' = wants the full picture; "
            "'example_first' = needs a concrete case before the abstraction; "
            "'step_by_step' = wants sequential reasoning. "
            "'unknown' until there is clear evidence."
        ),
    )

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
        description="Estimated progression relative to previous turns."
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

    stage: Literal["introduce", "check", "deepen", "wrap_up"] = Field(
        default="introduce",
        description=(
            "'introduce' = orient briefly, end with one guiding question. "
            "'check' = evaluate the student's reply to that question. "
            "'deepen' = give the full grounded explanation. "
            "'wrap_up' = brief recap, invite practice or the next topic."
        )
    )

    turns_in_stage: int = Field(
        default=0,
        description="How many consecutive turns have been spent in the current stage.",
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

class TutorState(MessagesState):
    student_profile: StudentProfile

    learning_state: LearningState = LearningState()  # type: ignore

    teaching_state: TeachingState = TeachingState()  # type: ignore

    retrieved_docs: list[Document]

    answer_plan: AnswerPlan

    evidence: list[ChunkEvidence] = Field(default_factory=list) # type: ignore