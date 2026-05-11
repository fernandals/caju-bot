from dataclasses import dataclass

@dataclass
class ModelRegistry:
    generation_llm: any
    tracking_llm: any
    planning_llm: any
    grading_llm: any