from pydantic import BaseModel
from typing import Literal, Optional
from langgraph.graph import MessagesState

class StudentProfile(BaseModel):
    asks_exercise: int = 0
    asks_detail: int = 0
    asks_objectivity: int = 0

    current_profile: str = "neutral" # "analytical", "explorer", "objective", "neutral"
    confidence: float = 0.0

class LearningState(BaseModel):
    topic: Optional[str] = None
    subtopic: Optional[str] = None

    intent: Literal[
        "learn",
        "review",
        "practice",
        "solve_problem",
        "exam_prep",
        "debug_confusion"
    ] = "learn"

    comprehension_level: Literal[
        "low",
        "medium",
        "high"
    ] = "medium"

    response_style: Literal[
        "concise",
        "detailed",
        "interactive"
    ] = "detailed"

    wants_examples: bool = False
    wants_exercises: bool = False
    wants_step_by_step: bool = False

    frustration_level: float = 0.0

    confidence: float = 0.5

class AnswerPlan(BaseModel):
    needs_retrieval: bool = True

    strategy: Literal[
        "direct_answer",
        "guided_teaching",
        "exercise_first",
        "hint_only",
        "step_by_step"
    ]

    concepts_to_cover: list[str] = []

    include_examples: bool = False
    include_exercises: bool = False
    include_analogies: bool = False

    response_depth: Literal[
        "light",
        "medium",
        "deep"
    ] = "light"

    confidence: float = 0.5

class RetrievedDocument(BaseModel):
    content: str
    source: str
    relevance_score: float
    difficulty_level: str

class TutorConfig(BaseModel):
    subject: str
    allow_direct_answers: bool = False
    course_level: str = "beginner" # beginner, intermediate, advanced
    answer_language: str = "Português"
    max_sentences: int = 6

class TutorState(MessagesState):
    student_profile: StudentProfile
    
    learning_state: LearningState = LearningState() # type: ignore

    retrieved_docs: list[RetrievedDocument]

    answer_plan: AnswerPlan