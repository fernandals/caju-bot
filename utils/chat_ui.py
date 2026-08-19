import re
import uuid

import langchain_core.messages
import streamlit as st

from agent.state import StudentProfile, TutorState
from utils.helpers import load_chats, save_chat

TUTOR_AVATAR = "🎓"
STUDENT_AVATAR = "🙋"

# Wraps our own citation markers - e.g. "[SAIA-Chapter13.pdf, Chapter 13,
# Section 13.1 ...]" - in backticks so they render as a distinct inline-code
# "badge" instead of blending into plain text. Deliberately uses Markdown's
# native code-span syntax rather than raw HTML: the surrounding text is
# LLM-generated, and injecting unsafe_allow_html on model output is an XSS
# risk not worth taking for a cosmetic touch. Matched narrowly (must
# contain ".pdf") so it can't accidentally wrap unrelated bracketed text.
_CITATION_PATTERN = re.compile(r"(\[[^\[\]]*?\.pdf[^\[\]]*?\])")


def _highlight_citations(text: str) -> str:
    return _CITATION_PATTERN.sub(r"`\1`", text)


def render_chat(graph, tutor_prompt, config, discipline: str, student_id: str):
    """
    Shared chat UI: message history, input loop, graph invocation, and a
    sidebar of past conversations scoped to (discipline, student_id).

    Used by both pages/chat.py (multi-discipline dev tool, student_id="dev")
    and student_app.py (the pilot's single-course, roster-gated entrypoint).
    """

    session_key = (discipline, student_id)

    # ---------------- RESET ON DISCIPLINE/STUDENT CHANGE ----------------
    if "chat_session_key" not in st.session_state:
        st.session_state.chat_session_key = session_key

    if st.session_state.chat_session_key != session_key:
        st.session_state.chat_session_key = session_key
        st.session_state.chat = []
        st.session_state.state = TutorState(
            messages=[tutor_prompt],
            student_profile=StudentProfile(),
        )  # type: ignore

    # ---------------- INIT ----------------
    if "state" not in st.session_state:
        st.session_state.state = TutorState(
            messages=[tutor_prompt],
            student_profile=StudentProfile(),
        )  # type: ignore

    if "chat_id" not in st.session_state:
        st.session_state.chat_id = str(uuid.uuid4())

    if "chat" not in st.session_state:
        st.session_state.chat = []

    # ---------------- CHAT HISTORY ----------------
    for msg in st.session_state.chat:
        avatar = TUTOR_AVATAR if msg["role"] == "assistant" else STUDENT_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(_highlight_citations(msg["content"]))

    # ---------------- INPUT ----------------
    if user_input := st.chat_input("Digite sua pergunta..."):
        st.session_state.chat.append({"role": "user", "content": user_input})

        with st.chat_message("user", avatar=STUDENT_AVATAR):
            st.markdown(user_input)

        st.session_state.state["messages"].append(
            langchain_core.messages.HumanMessage(content=user_input)
        )

        with st.chat_message("assistant", avatar=TUTOR_AVATAR):  # noqa: SIM117
            with st.spinner("Pensando..."):

                final_state = None

                for state in graph.stream(st.session_state.state, stream_mode="values"):  # type: ignore
                    final_state = state

                st.session_state.state = final_state

                answer = final_state["messages"][-1].content  # type: ignore

                st.markdown(_highlight_citations(answer))

        st.session_state.chat.append({"role": "assistant", "content": answer})
        save_chat(
            st.session_state.chat_id,
            st.session_state.chat,
            discipline,
            student_id,
        )

    # ---------------- SIDEBAR: PAST CONVERSATIONS ----------------
    st.sidebar.markdown("### 💬 Conversas")

    if st.sidebar.button("➕ Nova conversa", use_container_width=True):
        st.session_state.chat = []
        st.session_state.chat_id = str(uuid.uuid4())

    st.sidebar.markdown("---")

    chats = [
        chat
        for chat in load_chats()
        if chat.get("discipline") == discipline
        and chat.get("student_id") == student_id
    ]

    if not chats:
        st.sidebar.caption("Nenhuma conversa ainda. Faça sua primeira pergunta!")

    for chat in chats:

        if chat["messages"]:
            title = chat["messages"][0]["content"][:30]
        else:
            title = "Conversa vazia"

        is_active = chat["chat_id"] == st.session_state.get("chat_id")

        if st.sidebar.button(
            f"{'📍 ' if is_active else ''}{title}",
            key=f"chat_{chat['chat_id']}",
            use_container_width=True,
        ):
            st.session_state.chat = chat["messages"]
            st.session_state.chat_id = chat["chat_id"]
