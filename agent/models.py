from dataclasses import dataclass

@dataclass
class ModelRegistry:
    generation_llm: any  # type: ignore
    tracking_llm: any    # type: ignore
    planning_llm: any    # type: ignore
    grading_llm: any     # type: ignore