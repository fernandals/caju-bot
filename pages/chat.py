import uuid

import langchain_core.messages
import streamlit as st
from dotenv import load_dotenv

from agent.chat_pipeline import load_pipeline
from agent.state import StudentProfile, TutorState
from rag.knowledge_base import list_knowledge_bases
from utils.helpers import load_chats, save_chat

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
    ) # type: ignore

# ---------------- INIT ----------------
if "state" not in st.session_state:
    st.session_state.state = TutorState(
        messages=[tutor_prompt],
        student_profile=StudentProfile(),
    ) # type: ignore

if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())

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
        langchain_core.messages.HumanMessage(content=user_input)
    )

    with st.chat_message("assistant"):  # noqa: SIM117
        with st.spinner("Pensando..."):
            
            final_state = None

            for state in graph.stream(st.session_state.state, stream_mode="values"): # type: ignore
                final_state = state

            st.session_state.state = final_state

            print("\n========== FINAL STATE ==========")
            print(final_state)

            answer = final_state["messages"][-1].content # type: ignore
            
            st.markdown(answer)

    st.session_state.chat.append({"role": "assistant", "content": answer})
    save_chat(st.session_state.chat_id, st.session_state.chat, selected_kb)

st.sidebar.title("Conversas")

if st.sidebar.button("➕ Nova conversa"):
    st.session_state.chat = []
    st.session_state.chat_id = str(uuid.uuid4())

st.sidebar.markdown("---")

chats = load_chats()

for i, chat in enumerate(chats):

    if chat["discipline"] != selected_kb:
        continue
    if chat["messages"]:
        title = chat["messages"][0]["content"][:30]
    else:
        title = "Conversa vazia"

    if st.sidebar.button(title, key=f"chat_{chat['chat_id']}"):
        st.session_state.chat = chat["messages"]
        st.session_state.chat_id = chat["chat_id"]