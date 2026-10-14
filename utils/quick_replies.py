"""
Ready-made reply buttons shown under the tutor's messages.

Which buttons appear is driven by the teaching arc (agent/teaching.py):
each stage offers the 2 replies a student most often wants at that point.
Clicking one posts its `message` as if the student had typed it, so it
goes through the normal pipeline - the text is phrased to be an
unambiguous signal for the tracker ("Não sei." -> one non-attempt, not
yet "stuck"; "Me dá uma dica?" -> engaged, wants a hint).

Deliberately there is NO "just give me the answer" button while the arc
is guiding: a one-click shortcut would undercut the guided pacing. The
student can still type that, which goes through the planner's "refer"
handling as before.
"""

from typing import NamedTuple

from agent.graph import is_topics_request
from agent.nodes import latest_student_question


class QuickReply(NamedTuple):
    label: str
    message: str
    icon: str  # lucide icon name (rendered by Chainlit's action buttons)


ACTION_NAME = "quick_reply"

# Thread metadata key holding the last reply's buttons: Chainlit doesn't
# persist actions, so a resumed conversation puts them back from here
# (chainlit_app.on_chat_resume).
QUICK_REPLIES_KEY = "quick_replies"


def save_quick_replies(replies: list["QuickReply"]) -> list[dict]:
    return [r._asdict() for r in replies]


def load_quick_replies(saved) -> list["QuickReply"] | None:
    """The saved buttons; None if nothing usable was saved (older threads)."""
    if not isinstance(saved, list):
        return None
    try:
        return [QuickReply(**r) for r in saved]
    except TypeError:
        return None

# Shown under the welcome message (and after a bare greeting). Course-
# agnostic on purpose: one deploy serves one course, and the knowledge
# base's chapter titles are often missing, so topic-specific starters
# can't be generated reliably.
TOPICS = QuickReply(
    "Quais assuntos posso estudar?",
    "Quais assuntos da disciplina posso estudar com você?",
    "list",
)

STARTERS = [
    TOPICS,
    QuickReply(
        "Revisar para a prova",
        "Tenho prova em breve. Me ajuda a revisar os pontos principais?",
        "notebook-pen",
    ),
    QuickReply(
        "Quero praticar",
        "Me passa um exercício para eu praticar.",
        "pencil",
    ),
]

HINT = QuickReply("Me dá uma dica", "Me dá uma dica?", "lightbulb")
DONT_KNOW = QuickReply("Não sei", "Não sei.", "circle-help")
EXAMPLE = QuickReply("Me dá um exemplo", "Pode me dar um exemplo?", "shapes")
TEST_ME = QuickReply(
    "Me testa", "Me faz uma pergunta para eu testar se entendi.", "circle-check"
)
EXERCISE = QuickReply("Quero um exercício", "Me passa um exercício sobre isso.", "pencil")
NEXT_TOPIC = QuickReply(
    "Próximo tópico",
    "Vamos para o próximo tópico. O que mais eu deveria estudar?",
    "arrow-right",
)
NOT_FOUND = QuickReply(
    "Não encontrei no material",
    "Não encontrei no material, pode me explicar?",
    "search-x",
)

_BY_STAGE = {
    "introduce": [HINT, DONT_KNOW],
    "check": [HINT, DONT_KNOW],
    "refer": [NOT_FOUND],
    "deepen": [EXAMPLE, TEST_ME],
    "wrap_up": [EXERCISE, NEXT_TOPIC],
}


def quick_replies_for(final_state) -> list[QuickReply]:
    """Buttons to attach to the tutor's reply for this finished turn."""
    if final_state.get("greeted"):
        # Just listed the topics (agent/nodes.list_topics) - no point
        # offering the same list again.
        if is_topics_request(latest_student_question(final_state.get("messages") or [])):
            return [r for r in STARTERS if r is not TOPICS]
        return STARTERS

    teaching_state = final_state.get("teaching_state")
    if teaching_state is None:
        return []

    # No topic yet (e.g. "help me review" before naming what): the tutor
    # is asking what to cover, so offer the topic list, not stage replies.
    learning_state = final_state.get("learning_state")
    if learning_state is not None and not learning_state.topic:
        return [TOPICS]

    plan = final_state.get("answer_plan")
    strategy = getattr(plan, "strategy", None)

    # Direct mode always delivers a full explanation, whatever the stage.
    if teaching_state.mode == "direct":
        return _BY_STAGE["deepen"]

    # These strategies replace the stage block with a problem / a hint
    # (see resolve_teaching_instructions) - the student is mid-attempt.
    if strategy in ("exercise_first", "hint_only"):
        return [HINT, DONT_KNOW]

    return _BY_STAGE.get(teaching_state.stage, [])


# ---------------- STYLE QUESTION ----------------
# Asked once, in a student's first conversation (utils/chat_session
# .should_ask_style), so personalization can start before the profiler has
# observed anything. The answer is stored as StudentProfile.stated_style.
STYLE_ACTION_NAME = "style_choice"
STYLE_QUESTION = "Antes de começar: como você prefere que eu explique?"
STYLE_SKIP = "skip"

STYLE_CHOICES = [
    # (label, stated_style value, lucide icon)
    ("Exemplo primeiro", "example_first", "shapes"),
    ("Passo a passo", "step_by_step", "list-ordered"),
    ("Direto ao ponto", "concise", "zap"),
    ("Explicação completa", "detailed", "book-open"),
    ("Pular", STYLE_SKIP, "skip-forward"),
]

STYLE_LABELS = {value: label for label, value, _ in STYLE_CHOICES}
