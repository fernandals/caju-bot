import re
import tempfile
from pathlib import Path

import streamlit as st

from rag.knowledge_base import create_and_save_knowledge_base

st.set_page_config(page_title="Criar Disciplina", page_icon="👨‍🏫", layout="centered")

st.title("👨‍🏫 Criar disciplina")
st.caption(
    "Envie os PDFs da disciplina e gere a base de conhecimento usada pelo "
    "tutor. O próximo passo, fora desta interface, é empacotar essa base "
    "numa imagem Docker e publicar (ver DEPLOY.md)."
)

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
            kb = create_and_save_knowledge_base(tmp_path, discipline_name)  # type: ignore

    st.success(f"Base '{discipline_name}' criada com sucesso!")

    if kb.stats:
        st.caption(
            f"{kb.stats['documents']} arquivo(s), "
            f"{kb.stats['sections']} seção(ões) detectada(s), "
            f"{kb.stats['chunks']} trecho(s) indexado(s)."
        )

        if kb.stats["sections"] <= kb.stats["documents"]:
            st.warning(
                "Poucas ou nenhuma seção foi detectada nos arquivos enviados. "
                "Isso pode indicar que a formatação do material não foi "
                "reconhecida corretamente, o que deixa as referências das "
                "respostas menos específicas (sem capítulo/seção). Revise os "
                "PDFs enviados se possível."
            )

        if kb.stats.get("low_text_files"):
            files = ", ".join(kb.stats["low_text_files"])
            st.warning(
                f"Os seguintes arquivos têm pouco ou nenhum texto extraível "
                f"e provavelmente são PDFs escaneados ou baseados em imagem: "
                f"{files}. O tutor não conseguirá responder perguntas sobre "
                f"o conteúdo desses arquivos."
            )
