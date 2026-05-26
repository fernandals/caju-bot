import streamlit as st
from pathlib import Path
import tempfile
import re

from rag.knowledge_base import create_and_save_knowledge_base

st.set_page_config(page_title="Criar Disciplina", layout="centered")

st.title("👨‍🏫 Criar disciplina")
st.caption("Envie materiais e gere a base de conhecimento da disciplina")

st.markdown("---")

# ---------------- FORM ----------------
discipline_name = st.text_input("Nome da disciplina")

uploaded_files = st.file_uploader(
    "Envie os PDFs da disciplina",
    type=["pdf"],
    accept_multiple_files=True
)

# ----------------- NORMALIZATION ----------------
def normalize_name(name):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)

discipline_name = normalize_name(discipline_name)

# ---------------- ACTION ----------------
if st.button("🚀 Criar", use_container_width=True):

    if not discipline_name:
        st.warning("Digite o nome da disciplina")
        st.stop()

    if not uploaded_files:
        st.warning("Envie pelo menos um PDF")
        st.stop()

    # Criar pasta temporária
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        for file in uploaded_files:
            file_path = tmp_path / file.name
            with open(file_path, "wb") as f:
                f.write(file.read())

        # Processo pesado → spinner
        with st.spinner("Criando base de conhecimento..."):
            create_and_save_knowledge_base(tmp_path, discipline_name) # type: ignore

    st.success(f"Base '{discipline_name}' criada com sucesso!")