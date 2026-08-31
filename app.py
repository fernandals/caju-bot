import json
import re
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from rag.knowledge_base import (
    create_and_save_knowledge_base,
    list_material_sections,
)

st.set_page_config(
    page_title="Tutor — Painel do professor", page_icon="👨‍🏫", layout="centered"
)

tab_create, tab_metrics = st.tabs(
    ["👨‍🏫 Criar disciplina", "📊 Métricas da turma"]
)


# ==========================================================
# Aba 1 — Criar disciplina
# ==========================================================

def normalize_name(name):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


with tab_create:
    st.title("👨‍🏫 Criar disciplina")
    st.caption(
        "Envie os PDFs da disciplina e gere a base de conhecimento usada pelo "
        "tutor. O próximo passo, fora desta interface, é empacotar essa base "
        "numa imagem Docker e publicar (ver DEPLOY.md)."
    )

    st.markdown("---")

    discipline_name = normalize_name(st.text_input("Nome da disciplina"))

    uploaded_files = st.file_uploader(
        "Envie os PDFs da disciplina",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("🚀 Criar", width="stretch"):

        if not discipline_name:
            st.warning("Digite o nome da disciplina")
            st.stop()

        if not uploaded_files:
            st.warning("Envie pelo menos um PDF")
            st.stop()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            for file in uploaded_files:
                with open(tmp_path / file.name, "wb") as f:
                    f.write(file.read())

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


# ==========================================================
# Aba 2 — Métricas da turma
# ==========================================================

DEFAULT_METRICS_DB = Path("data/chats/metrics.db")

_LABELS = {
    "comprehensionLevel": {"low": "baixo", "medium": "médio", "high": "alto"},
    "learningProgress": {
        "stuck": "travado",
        "stable": "estável",
        "improving": "melhorando",
        "mastered": "dominado",
    },
}
_ORDER = {
    "comprehensionLevel": ["low", "medium", "high"],
    "learningProgress": ["stuck", "stable", "improving", "mastered"],
}


def _read_metrics_db(path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(path)
    try:
        df = pd.read_sql_query("SELECT * FROM turn_metrics", conn)
    finally:
        conn.close()

    if not df.empty:
        df["createdAt"] = pd.to_datetime(
            df["createdAt"], errors="coerce", utc=True
        )
    return df


def _ordered_counts(df: pd.DataFrame, column: str) -> pd.Series:
    counts = df[column].value_counts()
    counts = counts.reindex(_ORDER[column]).fillna(0).astype(int)
    return counts.rename(index=_LABELS[column])


def _citation_dicts(df: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    for raw in df["citations"].dropna():
        try:
            out.extend(json.loads(raw))
        except (TypeError, ValueError):
            continue
    return out


def _s(value) -> str | None:
    return str(value) if value is not None else None


def render_metrics_panel() -> None:
    st.title("📊 Métricas da turma")
    st.caption(
        "Sinais pedagógicos **anônimos**, um por turno de conversa — sem "
        "identificação de aluno e sem ligação entre turnos. Dá pra ver onde a "
        "turma trava, não pra acompanhar um aluno específico. Baixe o "
        "`metrics.db` do volume do Railway (ver DEPLOY.md) e aponte abaixo."
    )

    uploaded = st.file_uploader(
        "Arquivo metrics.db", type=["db", "sqlite", "sqlite3"]
    )
    path_input = st.text_input(
        "…ou caminho local", value=str(DEFAULT_METRICS_DB)
    )

    if uploaded is not None:
        db_path = Path(tempfile.gettempdir()) / "uploaded_metrics.db"
        db_path.write_bytes(uploaded.getvalue())
    else:
        db_path = Path(path_input).expanduser()

    if not db_path.is_file():
        st.info("Nenhum arquivo de métricas encontrado ainda.")
        return

    try:
        df = _read_metrics_db(db_path)
    except Exception as exc:  # noqa: BLE001 - surface any read error to the user
        st.error(f"Não consegui ler o arquivo: {exc}")
        return

    if df.empty:
        st.info("O arquivo existe, mas ainda não tem turnos registrados.")
        return

    disciplines = sorted(df["discipline"].dropna().unique())
    discipline = disciplines[0] if disciplines else None

    if len(disciplines) > 1:
        discipline = st.selectbox("Disciplina", disciplines)
        df = df[df["discipline"] == discipline]

    total = len(df)
    stuck = int((df["learningProgress"] == "stuck").sum())
    frustration = df["frustrationLevel"].dropna()

    a, b, c = st.columns(3)
    a.metric("Turnos registrados", total)
    b.metric("Turnos travados", f"{stuck}", f"{stuck / total:.0%}" if total else None)
    c.metric(
        "Frustração média",
        f"{frustration.mean():.2f}" if not frustration.empty else "—",
    )

    st.markdown("---")

    left, right = st.columns(2)
    with left:
        st.markdown("**Compreensão estimada**")
        st.bar_chart(_ordered_counts(df, "comprehensionLevel"))
    with right:
        st.markdown("**Progresso na conversa**")
        st.bar_chart(_ordered_counts(df, "learningProgress"))

    st.markdown("**Tópicos mais frequentes**")
    st.bar_chart(
        df["topic"].fillna("(sem tópico)").value_counts().head(15)
    )

    st.markdown("**Estratégias pedagógicas usadas**")
    st.bar_chart(df["strategy"].fillna("(n/d)").value_counts())

    daily = (
        df.dropna(subset=["createdAt"])
        .assign(dia=lambda d: d["createdAt"].dt.date)
        .groupby("dia")["frustrationLevel"]
        .mean()
    )
    if not daily.empty:
        st.markdown("**Frustração média ao longo do tempo**")
        st.line_chart(daily)

    _render_coverage(df, discipline)

    with st.expander("Ver dados brutos"):
        st.dataframe(df, width="stretch", hide_index=True)


def _render_coverage(df: pd.DataFrame, discipline: str | None) -> None:
    st.markdown("---")
    st.markdown("### Cobertura do material")

    citations = _citation_dicts(df)

    if not citations:
        st.caption("Nenhuma citação registrada ainda.")
        return

    cites = pd.DataFrame(citations)
    label = (
        cites["file"].fillna("?")
        + " § "
        + cites["section_title"]
        .fillna(cites.get("section_id"))
        .fillna("?")
        .astype(str)
    )
    st.markdown("**Seções mais consultadas**")
    st.bar_chart(label.value_counts().head(15))

    if not discipline:
        return

    all_sections = list_material_sections(discipline)
    if not all_sections:
        return

    seen = {(c.get("file"), _s(c.get("section_id"))) for c in citations}
    never = [
        s for s in all_sections if (s["file"], _s(s["section_id"])) not in seen
    ]

    st.markdown(
        f"**Seções que nunca apareceram numa resposta** "
        f"({len(never)} de {len(all_sections)})"
    )
    if never:
        st.dataframe(
            pd.DataFrame(never)[["file", "section_id", "section_title"]],
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("Todas as seções do material já foram usadas ao menos uma vez.")


with tab_metrics:
    render_metrics_panel()
