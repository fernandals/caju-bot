from dataclasses import dataclass
from typing import Any


@dataclass
class ModelRegistry:
    generation_llm: Any
    tracking_llm: Any
    planning_llm: Any
    grading_llm: Any
    # Writes the reply that checks a student's exercise attempt
    # (TeachingState.reviewing_attempt): gpt-4o-mini, the generation
    # default, praised wrong code or flagged side issues instead of the bug.
    review_llm: Any