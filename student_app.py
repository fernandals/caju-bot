import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent.chat_pipeline import load_pipeline
from rag.knowledge_base import list_knowledge_bases
from utils.chat_ui import render_chat
from utils.helpers import is_enrolled, load_roster

load_dotenv()

# ---------------- SINGLE BAKED-IN COURSE ----------------
# One knowledge base ships per container - no discipline picker.
kbs = list_knowledge_bases()

if len(kbs) != 1:
    st.set_page_config(page_title="Tutor - erro de configuração")
    st.error(
        f"Este container deveria conter exatamente uma disciplina, mas "
        f"encontrou {len(kbs)}. Verifique a imagem/dados montados."
    )
    st.stop()

discipline = kbs[0]["name"]

st.set_page_config(
    page_title=f"Tutor - {discipline}",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- ROSTER GATE ----------------
roster_path = Path(os.getenv("ROSTER_PATH", "data/roster.txt"))
roster = load_roster(roster_path)

if "student_id" not in st.session_state:
    st.session_state.student_id = None

if not st.session_state.student_id:

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<h2 style='text-align: center;'>🎓 {discipline}</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: gray;'>"
        "Identifique-se com sua matrícula para começar"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    _, center, _ = st.columns([1, 2, 1])

    with center:
        entered_id = st.text_input("Número de matrícula", key="enrollment_input")

        if st.button("Entrar", use_container_width=True):
            if not roster:
                st.error(
                    "Nenhuma lista de matrículas configurada para esta "
                    "disciplina. Avise o professor."
                )
            elif is_enrolled(entered_id, roster):
                st.session_state.student_id = entered_id.strip()
                st.rerun()
            else:
                st.error("Matrícula não reconhecida. Confira com o professor.")

    st.stop()

student_id = st.session_state.student_id

# ---------------- PIPELINE ----------------
@st.cache_resource(show_spinner=True)
def get_pipeline(name):
    return load_pipeline(name)

graph, tutor_prompt, config = get_pipeline(discipline)

# ---------------- HEADER ----------------
st.markdown(
    f"""
    <div style="padding: 0.5rem 0 1rem 0;">
        <h1 style="margin-bottom: 0;">🎓 {discipline}</h1>
        <p style="color: gray; margin-top: 0.2rem;">
            Tire suas dúvidas com base no material da disciplina.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.caption(f"Matrícula: {student_id}")

    if st.button("Trocar matrícula", use_container_width=True):
        st.session_state.student_id = None
        st.session_state.pop("chat_session_key", None)
        st.rerun()

    st.markdown("---")

# ---------------- CHAT UI ----------------
render_chat(graph, tutor_prompt, config, discipline=discipline, student_id=student_id)
