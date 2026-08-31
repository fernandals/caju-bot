import asyncio
import functools
import os
import sqlite3
from pathlib import Path

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from agent.chat_pipeline import load_pipeline
from agent.state import StudentProfile, TutorState
from rag.knowledge_base import (
    describe_course_materials,
    list_knowledge_bases,
    resolve_source_pdf,
)
from utils.citations import highlight_citations
from utils.helpers import is_enrolled, load_roster

load_dotenv()

# ---------------- SINGLE BAKED-IN COURSE ----------------
# One knowledge base ships per container - no discipline picker. Unlike
# the old Streamlit entrypoint (which stayed up and showed an error page),
# this fails fast at import: a container shipped with the wrong number of
# knowledge bases is a build mistake, not a runtime condition to survive.
kbs = list_knowledge_bases()

if len(kbs) != 1:
    raise RuntimeError(
        f"Este container deveria conter exatamente uma disciplina, mas "
        f"encontrou {len(kbs)}. Verifique a imagem/dados montados."
    )

DISCIPLINE = kbs[0]["name"]

ROSTER = load_roster(Path(os.getenv("ROSTER_PATH", "data/roster.txt")))


# ---------------- "LEIA-ME" PAGE ----------------
# The welcome/readme page is generated per container from the baked-in
# knowledge base (course name + the materials it indexed) so it always
# matches the course actually shipped, with nothing hardcoded. Chainlit
# re-reads this file on every /project/settings request; the app language
# is pinned to pt-BR (.chainlit/config.toml), so it looks for
# chainlit_pt-BR.md and falls back to the static chainlit.md if writing
# the generated file fails for any reason.
def _render_readme() -> str:
    lines = [
        f"# Tutor Virtual — {DISCIPLINE}",
        "",
        f"Este assistente ajuda você a estudar **{DISCIPLINE}**. Ele responde "
        "às suas perguntas **com base no material da disciplina** "
        "disponibilizado pelo professor — não busca informações fora desse "
        "conteúdo.",
        "",
        "Sempre que possível, o tutor indica o capítulo e a seção do material "
        "em que se baseou, para você conferir na fonte.",
    ]

    try:
        materials = describe_course_materials(DISCIPLINE)
    except Exception:
        materials = []

    if materials:
        lines += ["", "## Material coberto", ""]
        for m in materials:
            if m["chapter"] is not None and m["title"]:
                lines.append(f"- **Capítulo {m['chapter']}** — {m['title']}")
            elif m["chapter"] is not None:
                lines.append(f"- **Capítulo {m['chapter']}** ({m['file']})")
            else:
                lines.append(f"- {m['file']}")

    lines += [
        "",
        "## Como usar",
        "",
        "- Pergunte sobre conceitos, definições e exemplos vistos na disciplina.",
        "- Peça explicações passo a passo, comparações ou mais exemplos.",
        "- Use **Novo chat** para recomeçar do zero; as conversas anteriores "
        "continuam salvas na barra lateral.",
        "",
        "O tutor é um apoio ao estudo e pode cometer erros — confirme sempre as "
        "informações importantes no material e nas aulas.",
        "",
    ]

    return "\n".join(lines)


try:
    Path("chainlit_pt-BR.md").write_text(_render_readme(), encoding="utf-8")
except OSError:
    pass


@functools.lru_cache(maxsize=1)
def _get_pipeline():
    """Built once per container process (mirrors the old @st.cache_resource)."""
    return load_pipeline(DISCIPLINE)


def _run_graph_sync(graph, state):
    """
    agent/graph.py's nodes call the LLM synchronously (.invoke, not
    .ainvoke), so this blocks on network I/O. Must be run off the asyncio
    event loop (see on_message) or it freezes the app for every connected
    student during each turn.
    """
    final_state = None
    for step_state in graph.stream(state, stream_mode="values"):
        final_state = step_state
    return final_state


# ---------------- CHAT HISTORY / THREAD PERSISTENCE ----------------
# Postgres-flavored official schema (UUID/JSONB/TEXT[]) doesn't map onto
# SQLite - adapted to TEXT columns (metadata/tags stored as JSON strings).
# Verified empirically against chainlit 2.11.1's SQLAlchemyDataLayer before
# committing to this approach.
#
# Lives under data/chats/ (not directly under data/) so it shares the
# mounted volume with the dev tool's JSON chat history without shadowing
# data/knowledge_bases/ and data/roster.txt, which are baked into the
# image at data/ directly - a volume mounted at data/ itself would hide
# them behind an empty volume on first boot.
DB_PATH = Path("data/chats/chainlit.db")

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    "id" TEXT PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" TEXT NOT NULL,
    "createdAt" TEXT
);

CREATE TABLE IF NOT EXISTS threads (
    "id" TEXT PRIMARY KEY,
    "createdAt" TEXT,
    "name" TEXT,
    "userId" TEXT,
    "userIdentifier" TEXT,
    "tags" TEXT,
    "metadata" TEXT,
    FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS steps (
    "id" TEXT PRIMARY KEY,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "threadId" TEXT NOT NULL,
    "parentId" TEXT,
    "streaming" BOOLEAN NOT NULL,
    "waitForAnswer" BOOLEAN,
    "isError" BOOLEAN,
    "metadata" TEXT,
    "tags" TEXT,
    "input" TEXT,
    "output" TEXT,
    "createdAt" TEXT,
    "command" TEXT,
    "start" TEXT,
    "end" TEXT,
    "generation" TEXT,
    "showInput" TEXT,
    "language" TEXT,
    "indent" INT,
    "defaultOpen" BOOLEAN,
    "autoCollapse" BOOLEAN,
    "icon" TEXT,
    "modes" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS elements (
    "id" TEXT PRIMARY KEY,
    "threadId" TEXT,
    "type" TEXT,
    "path" TEXT,
    "url" TEXT,
    "chainlitKey" TEXT,
    "name" TEXT NOT NULL,
    "display" TEXT,
    "objectKey" TEXT,
    "size" TEXT,
    "page" INT,
    "language" TEXT,
    "forId" TEXT,
    "mime" TEXT,
    "props" TEXT,
    "autoPlay" BOOLEAN,
    "playerConfig" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS feedbacks (
    "id" TEXT PRIMARY KEY,
    "forId" TEXT NOT NULL,
    "threadId" TEXT NOT NULL,
    "value" INT NOT NULL,
    "comment" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);
"""


def _ensure_sqlite_schema():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(_SQLITE_SCHEMA)
        conn.commit()
    finally:
        conn.close()


_ensure_sqlite_schema()


@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo=f"sqlite+aiosqlite:///{DB_PATH}")


# ---------------- LOGIN ----------------
# Reuses Chainlit's native two-field login form. The "password" field is
# repurposed to carry the student's display name - not a real secret, not
# checked against anything (see .chainlit/translations/pt-BR.json for the
# relabeled "Matrícula"/"Nome completo" fields). Matrícula is the actual
# identity, validated against the roster allowlist exactly like before.
@cl.password_auth_callback
def auth_callback(username: str, password: str) -> cl.User | None:
    matricula = username.strip()
    nome = password.strip()

    if not nome or not is_enrolled(matricula, ROSTER):
        return None

    return cl.User(identifier=matricula, display_name=nome, metadata={"name": nome})


# ---------------- CHAT LIFECYCLE ----------------
@cl.on_chat_start
async def on_chat_start():
    graph, tutor_prompt, _config = _get_pipeline()

    cl.user_session.set("graph", graph)
    cl.user_session.set(
        "state",
        TutorState(messages=[tutor_prompt], student_profile=StudentProfile()),  # type: ignore
    )

    user = cl.user_session.get("user")
    nome = (user.metadata or {}).get("name") if user else None
    greeting = f"Olá, {nome}! " if nome else "Olá! "

    await cl.Message(
        content=f"{greeting}Tire suas dúvidas sobre **{DISCIPLINE}**."
    ).send()


@cl.on_chat_resume
async def on_chat_resume(thread: cl.types.ThreadDict):
    """
    Rebuilds the LangGraph state from a resumed thread's message history.
    Same lossy-resume tradeoff as the old Streamlit app: only the message
    list is restored, student_profile/learning_state/teaching_state/evidence
    reset fresh - Chainlit's UI still replays the full step history for
    display regardless.
    """
    graph, tutor_prompt, _config = _get_pipeline()

    messages = [tutor_prompt]

    for step in thread.get("steps", []):
        if step.get("type") == "user_message":
            messages.append(HumanMessage(content=step.get("output", "")))
        elif step.get("type") == "assistant_message":
            messages.append(AIMessage(content=step.get("output", "")))

    cl.user_session.set("graph", graph)
    cl.user_session.set(
        "state",
        TutorState(messages=messages, student_profile=StudentProfile()),  # type: ignore
    )


def _short_ref_label(metadata: dict) -> str:
    """Compact chip label for a citation, e.g. '📄 SAIA-Chapter13.pdf, p. 2'."""
    file_name = metadata.get("file_path") or "material"
    start = metadata.get("page_start")
    end = metadata.get("page_end")

    if start and end and end != start:
        return f"📄 {file_name}, pp. {start}–{end}"
    if start:
        return f"📄 {file_name}, p. {start}"
    return f"📄 {file_name}"


def _linkify_citations(final_state, answer: str):
    """
    Replace each full citation string in the answer with a short label
    and return (rewritten_answer, elements).

    Chainlit turns any element whose `name` occurs in the message text
    into a clickable chip that opens the element, so each label names a
    side `cl.Pdf` that opens the source PDF at the cited page. The
    deterministic citation strings substitute_citation_markers() dropped
    into `answer` are what we match and swap out; citations whose source
    PDF wasn't saved with the KB keep their full text (highlight_citations
    still badges those).
    """
    docs = final_state.get("retrieved_docs", []) or []
    evidence = final_state.get("evidence", []) or []

    elements: list[cl.Pdf] = []
    label_by_target: dict[tuple, str] = {}

    for doc, ev in zip(docs, evidence):
        citation = getattr(ev, "citation", "")

        if not citation or citation not in answer:
            continue

        metadata = getattr(doc, "metadata", {}) or {}
        file_name = metadata.get("file_path", "")
        pdf_path = resolve_source_pdf(DISCIPLINE, file_name)

        if pdf_path is None:
            continue

        page_start = metadata.get("page_start")
        target = (file_name, page_start)

        label = label_by_target.get(target)
        if label is None:
            label = _short_ref_label(metadata)
            label_by_target[target] = label
            elements.append(
                cl.Pdf(
                    name=label,
                    path=str(pdf_path),
                    display="side",
                    page=page_start,
                )
            )

        answer = answer.replace(citation, label)

    return answer, elements


@cl.on_message
async def on_message(message: cl.Message):
    graph = cl.user_session.get("graph")
    state = cl.user_session.get("state")

    state["messages"].append(HumanMessage(content=message.content))

    async with cl.Step(name="Pensando...", type="run"):
        final_state = await asyncio.to_thread(_run_graph_sync, graph, state)

    cl.user_session.set("state", final_state)

    answer = final_state["messages"][-1].content
    answer, elements = _linkify_citations(final_state, answer)

    await cl.Message(
        content=highlight_citations(answer),
        elements=elements,
    ).send()
