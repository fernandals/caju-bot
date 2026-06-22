from pydantic import BaseModel, Field
from typing import Literal, Optional
from langgraph.graph import MessagesState

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
        description=(
            "Primary student intention in the current interaction. "
            "'learn' = understand a new concept, "
            "'review' = revisit known material, "
            "'practice' = train through exercises, "
            "'solve_problem' = solve a specific question/problem, "
            "'exam_prep' = prepare for assessments, "
            "'debug_confusion' = resolve misunderstanding or confusion."
        )
    )

    comprehension_level: Literal[
        "low",
        "medium",
        "high"
    ] = Field(
        default="medium",
        description=(
            "Estimated student understanding of the topic. "
            "'low' = significant confusion or beginner level, "
            "'medium' = partial understanding, "
            "'high' = mostly understands the concept."
        )
    )

    response_style: Literal[
        "concise",
        "detailed",
        "interactive"
    ] = Field(
        default="detailed",
        description=(
            "Preferred explanation style inferred from the interaction. "
            "'concise' = short and direct, "
            "'detailed' = deeper explanations, "
            "'interactive' = guided reasoning with engagement."
        )
    )

    wants_examples: bool = Field(
        default=False,
        description="Whether the student explicitly or implicitly requested examples."
    )

    wants_exercises: bool = Field(
        default=False,
        description="Whether the student wants exercises or practice problems."
    )

    wants_step_by_step: bool = Field(
        default=False,
        description="Whether the student prefers step-by-step explanations or reasoning."
    )

    frustration_level: float = Field(
        default=0.0,
        description=(
            "Estimated frustration or confusion level from 0 to 1. "
            "Higher values indicate stronger confusion, frustration, or difficulty."
        )
    )

    confidence: float = Field(
        default=0.5,
        description="Confidence score (0-1) representing reliability of the learning state analysis."
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

    concepts_to_cover: list[str] = Field(
        default=[],
        description=(
            "List of important concepts, ideas, or subtopics "
            "that should be addressed in the response."
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

    confidence: float = Field(
        default=0.5,
        description="Confidence score (0-1) representing reliability of the instructional plan."
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