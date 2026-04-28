import streamlit as st
from pathlib import Path
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from rag.loader import load_documents, parse_documents
from rag.splitter import spliting_documents
#from rag.vectorstore import build_vectorstore
from rag.knowledge_base import load_knowledge_base
from agent.tools import build_retrieve_tool
from agent.graph import build_graph
import agent.prompts as prompts
from student_model.profile import StudentProfile
from agent.state import TutorState, TutorConfig

# ---------------------- CONFIG ----------------------
load_dotenv()

st.set_page_config(page_title="Tutor RAG", layout="wide")
st.title("Tutor Inteligente")

PDF_PATH = Path("pdfs/")

# ---------------------- PIPELINE ----------------------
@st.cache_resource(show_spinner=True)
def load_pipeline():
    #docs = load_documents(PDF_PATH)
    #parsed_docs = parse_documents(docs)
    #chunks_table = spliting_documents(parsed_docs)
    #retriever = build_vectorstore(chunks_table)

    retriever = load_knowledge_base("Software Architecture")
    retrieve_tool = build_retrieve_tool(retriever)

    config = TutorConfig(subject="Software Architecture")

    tutor_prompt = SystemMessage(content=prompts.SYSTEM_PROMPT.format(
        domain=config.subject,
        max_sentences=config.max_sentences,
        course_level=config.course_level,
        answer_language=config.answer_language
    ))

    response_model = init_chat_model(
        os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=float(os.getenv("MODEL_TEMPERATURE", 0))
    ).bind_tools([retrieve_tool])

    graph = build_graph(config, retrieve_tool, response_model)

    return graph, tutor_prompt, config

# ---------------------- INIT ----------------------
graph, tutor_prompt, config = load_pipeline()

if "state" not in st.session_state:
    st.session_state.state = TutorState(
        messages=[tutor_prompt],
        student_profile=StudentProfile(),
        current_topic=None
    )

if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------------- CHAT UI ----------------------
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------- INPUT ----------------------
if user_input := st.chat_input("Digite sua pergunta..."):
    st.session_state.chat.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Add to state
    st.session_state.state["messages"].append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):

            final_state = None

            for step in graph.stream(st.session_state.state):  # type: ignore
                for node_name, state in step.items():
                    final_state = state

            # Update global state
            st.session_state.state = final_state

            answer = final_state["messages"][-1].content

            st.markdown(answer)

    st.session_state.chat.append({"role": "assistant", "content": answer})

# ---------------------- SIDEBAR DEBUG ----------------------
st.sidebar.title("Debug / Perfil do Aluno")

profile = st.session_state.state["student_profile"]

st.sidebar.markdown(f"**Perfil atual:** {profile.current_profile}")
st.sidebar.markdown(f"**Confiança:** {profile.confidence:.2f}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Sinais comportamentais:**")
st.sidebar.markdown(f"- Pede exercício: {profile.asks_exercise:.2f}")
st.sidebar.markdown(f"- Pede detalhe: {profile.asks_detail:.2f}")
st.sidebar.markdown(f"- Pede objetividade: {profile.asks_objectivity:.2f}")