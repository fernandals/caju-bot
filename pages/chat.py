import streamlit as st
from dotenv import load_dotenv

from agent.chat_pipeline import load_pipeline
from rag.knowledge_base import list_knowledge_bases
from utils.chat_ui import render_chat

# ---------------- CONFIG ----------------
load_dotenv()

st.set_page_config(
    page_title="Tutor RAG",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("👩‍🎓 Tutor Inteligente")

# ---------------- DISCIPLINA ----------------
kbs = list_knowledge_bases()

if not kbs:
    st.warning("Nenhuma disciplina disponível. Seu professor deve criar uma primeiro.")
    st.stop()

kb_names = [kb["name"] for kb in kbs]

# ---------------- STATE ----------------
if "selected_kb" not in st.session_state:
    st.session_state.selected_kb = None

# ---------------- SELECT ----------------
selected_kb_temp = st.selectbox(
    "Escolha a disciplina",
    kb_names,
    index=None,
    placeholder="Selecione uma disciplina"
)

# ---------------- CONFIRM ----------------
if st.button("Confirmar disciplina"):
    if not selected_kb_temp:
        st.warning("Selecione uma disciplina primeiro.")
        st.stop()

    st.session_state.selected_kb = selected_kb_temp

# ---------------- GATE ----------------
if not st.session_state.selected_kb:
    st.info("Selecione e confirme uma disciplina para começar.")
    st.stop()

selected_kb = st.session_state.selected_kb

# ---------------- CACHE PIPELINE ----------------
@st.cache_resource(show_spinner=True)
def get_pipeline(name):
    return load_pipeline(name)

graph, tutor_prompt, config = get_pipeline(selected_kb)

# ---------------- CHAT UI ----------------
render_chat(graph, tutor_prompt, config, discipline=selected_kb, student_id="dev")