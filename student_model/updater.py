from agent.state import LearningState
from student_model.profile import StudentProfile
from utils.helpers import softmax

def update_student_profile(profile, learning_state: LearningState):
    """
    Slow-changing behavioral adaptation.
    """

    alpha = 0.1

    if learning_state.response_style == "detailed":
        profile.prefers_detailed = (
            (1 - alpha) * profile.prefers_detailed + alpha
        )

    if learning_state.wants_examples:
        profile.prefers_examples = (
            (1 - alpha) * profile.prefers_examples + alpha
        )

    if learning_state.wants_exercises:
        profile.prefers_exercises = (
            (1 - alpha) * profile.prefers_exercises + alpha
        )

    if learning_state.response_style == "interactive":
        profile.prefers_interactive = (
            (1 - alpha) * profile.prefers_interactive + alpha
        )

    return profile

def update_conversation_topic(topic: str, user_message: str) -> str:
    # só funciona com LLM então vamos mudar isso aqui pra a llm que atualiza o topico vai atualizar também o perfil do aluno, dai a gente pode usar o mesmo prompt pra isso e evitar chamadas desnecessárias
    return topic
