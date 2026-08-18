from dataclasses import dataclass
from typing import Any


@dataclass
class ModelRegistry:
    generation_llm: Any
    tracking_llm: Any
    planning_llm: Any
    grading_llm: Any