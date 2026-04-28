import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage

from rag.knowledge_base import list_knowledge_bases
from agent.chat_pipeline import load_pipeline

from student_model.profile import StudentProfile
from agent.state import TutorState

# ---------------- CONFIG ----------------
load_dotenv()

st.set_page_config(page_title="Tutor RAG", layout="wide")
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

# ---------------- RESET AO TROCAR DISCIPLINA ----------------
if "current_kb" not in st.session_state:
    st.session_state.current_kb = selected_kb

if st.session_state.current_kb != selected_kb:
    st.session_state.current_kb = selected_kb
    st.session_state.chat = []
    st.session_state.state = TutorState(
        messages=[tutor_prompt],
        student_profile=StudentProfile(),
        current_topic=None
    )

# ---------------- INIT ----------------
if "state" not in st.session_state:
    st.session_state.state = TutorState(
        messages=[tutor_prompt],
        student_profile=StudentProfile(),
        current_topic=None
    )

if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- CHAT UI ----------------
# if not st.session_state.chat:
#     st.info("Faça uma pergunta sobre a disciplina selecionada.")

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- INPUT ----------------
if user_input := st.chat_input("Digite sua pergunta..."):
    st.session_state.chat.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.state["messages"].append(
        HumanMessage(content=user_input)
    )

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):

            final_state = None

            for step in graph.stream(st.session_state.state):  # type: ignore
                for _, state in step.items():
                    final_state = state

            st.session_state.state = final_state

            answer = final_state["messages"][-1].content
            st.markdown(answer)

    st.session_state.chat.append({"role": "assistant", "content": answer})

# ---------------- SIDEBAR ----------------
st.sidebar.title("Perfil do Aluno")

profile = st.session_state.state["student_profile"]

st.sidebar.metric("Perfil", profile.current_profile)
st.sidebar.metric("Confiança", f"{profile.confidence:.2f}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Sinais comportamentais:**")
st.sidebar.markdown(f"- Pede exercício: {profile.asks_exercise:.2f}")
st.sidebar.markdown(f"- Pede detalhe: {profile.asks_detail:.2f}")
st.sidebar.markdown(f"- Pede objetividade: {profile.asks_objectivity:.2f}")