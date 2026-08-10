from typing import Literal, Optional

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class StudentProfile(BaseModel):
    asks_exercise: int = Field(
        default=0,
        description="Number of times the student requested exercises or practice activities."
    )

    asks_detail: int = Field(
        default=0,
        description="Number of times the student requested detailed explanations."
    )

    asks_objectivity: int = Field(
        default=0,
        description="Number of times the student preferred concise or objective explanations."
    )

    current_profile: str = Field(
        default="neutral",
        description=(
            "Current inferred student interaction profile. "
            "Possible behaviors include: "
            "'analytical' (logical and reasoning-oriented), "
            "'explorer' (curious and analogy-driven), "
            "'objective' (prefers concise answers), "
            "'neutral' (no strong preference detected)."
        )
    )

    confidence: float = Field(
        default=0.0,
        description="Confidence score (0-1) representing how reliable the inferred profile is."
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

class RetrievedDocument(BaseModel):
    content: str = Field(
        description="Retrieved instructional content or chunk used as contextual grounding."
    )

    source: str = Field(
        description=(
            "Human-readable instructional source reference, such as lesson name, "
            "chapter, slide section, document title, or material reference."
        )
    )

    relevance_score: float = Field(
        description="Relevance score (0-1) indicating how useful the document is for the current question."
    )

    difficulty_level: str = Field(
        description=(
            "Estimated instructional difficulty level of the retrieved content, "
            "such as beginner, intermediate, or advanced."
        )
    )

class ChunkEvidence(BaseModel):
    """
    Evidence extracted from a retrieved chunk.
    """
    doc_id: str = Field(
        description="Reference identifier (DOC_1, DOC_2...)"
    )

    
    section: str = Field(
        description="Section where the evidence comes from."
    )
    pages: str = Field(
        description="Page interval."
    )

    citation: str = Field(
        description=(
            "Exact citation that MUST be used when information from this chunk "
            "appears in the final answer. An example citation format is: [Chapter 1, Section 1.1 Introduction to Algebra]."
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
        default=False,
        description="Whether the tutor is allowed to provide direct answers instead of guided reasoning."
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

    retrieved_docs: list[RetrievedDocument]

    answer_plan: AnswerPlan

    evidence: list[ChunkEvidence] = Field(default_factory=list) # type: ignore