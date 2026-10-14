import asyncio
import functools
import json
import logging
import os
import sqlite3
from pathlib import Path

import chainlit as cl
from chainlit.data import get_data_layer as _chainlit_data_layer
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from agent.chat_pipeline import load_pipeline
from agent.outline import cached_course_outline
from agent.graph import is_topics_request
from agent.nodes import latest_student_question
from agent.profiler import profile_student, worth_profiling
from agent.state import StudentProfile, TutorState
from agent.teaching import is_course_review
from agent.titler import generate_title
from rag.knowledge_base import (
    describe_course_materials,
    list_knowledge_bases,
    resolve_source_pdf,
)
from utils.chat_session import (
    NEW_THREAD_NAME,
    REVIEW_THREAD_NAME,
    TOPICS_THREAD_NAME,
    TUTOR_STATE_KEY,
    needs_title,
    restore_tutor_state,
    should_ask_style,
    truncate_for_edit,
    tutor_state_snapshot,
)
from utils.citations import find_ref_labels, highlight_citations, ref_label, replace_with_label
from utils.formatting import render_math
from utils.helpers import is_enrolled, load_roster
from utils.metrics import (
    ensure_metrics_schema,
    thread_trajectory,
    load_study_session,
    next_session_index,
    record_turn,
)
from utils.personalization import assign_arm
from utils.quick_replies import (
    ACTION_NAME,
    QUICK_REPLIES_KEY,
    STARTERS,
    STYLE_ACTION_NAME,
    STYLE_CHOICES,
    STYLE_LABELS,
    STYLE_QUESTION,
    STYLE_SKIP,
    QuickReply,
    load_quick_replies,
    save_quick_replies,
    quick_replies_for,
)
from utils.student_profile import (
    ensure_profile_schema,
    COURSE_METADATA_KEY,
    latest_other_thread,
    load_profile,
    save_profile,
    scope_to_course,
)

load_dotenv()

logger = logging.getLogger(__name__)

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
# knowledge base so it always matches the course actually shipped, with
# nothing hardcoded. Chainlit re-reads this file on every
# /project/settings request; the app language is pinned to pt-BR
# (.chainlit/config.toml), so it looks for chainlit_pt-BR.md and falls
# back to the static chainlit.md if writing the generated file fails.
#
# Topics are the course outline (agent/outline.py, the same list "Quais
# assuntos posso estudar?" gives) - the materials' file names, listed
# before, read like "TRAB 2A UNIDADE - 2022.1.pdf". Written at startup
# from the outline's cache, and again once the pipeline has built it
# (_get_pipeline) for a KB that had none yet.
README_PATH = Path("chainlit_pt-BR.md")


def _render_readme(topics: list[str]) -> str:
    lines = [
        f"# Tutor Virtual — {DISCIPLINE}",
        "",
        f"Este assistente ajuda você a estudar **{DISCIPLINE}**. Ele responde "
        "às suas perguntas **com base no material da disciplina** "
        "disponibilizado pelo professor — não busca informações fora desse "
        "conteúdo.",
        "",
        "Cada resposta traz links para as páginas do material em que se "
        "baseou: clique em um deles para abrir o PDF e conferir na fonte.",
    ]

    if topics:
        lines += ["", "## Assuntos da disciplina", ""]
        # Bullets: the readme dialog clips two-digit list numbers ("10." -> ".0").
        lines += [f"- {t}" for t in topics]
    else:
        try:
            materials = describe_course_materials(DISCIPLINE)
        except Exception:
            materials = []
        if materials:
            lines += ["", "## Material coberto", ""]
            lines += [f"- {m['file']}" for m in materials]

    lines += [
        "",
        "## Como usar",
        "",
        "- Pergunte sobre conceitos, definições e exemplos vistos na disciplina.",
        "- O tutor vai te guiando com perguntas para você chegar à resposta; "
        "se travar ou estiver com pressa (\"tenho prova amanhã\"), ele explica direto.",
        "- Use os botões abaixo das respostas para ver os assuntos, revisar "
        "para a prova ou pedir um exercício — ao resolver, mande sua resposta "
        "que ele corrige.",
        "- Use **Novo chat** para recomeçar do zero; as conversas anteriores "
        "continuam salvas na barra lateral.",
        "- Em computador compartilhado, saia da sua conta pelo menu do seu "
        "avatar, no canto superior direito.",
        "",
        "O tutor é um apoio ao estudo e pode cometer erros — confirme sempre as "
        "informações importantes no material e nas aulas.",
        "",
    ]

    return "\n".join(lines)


def _write_readme(topics: list[str]) -> None:
    try:
        README_PATH.write_text(_render_readme(topics), encoding="utf-8")
    except OSError:
        pass


_write_readme(cached_course_outline(DISCIPLINE))


@functools.lru_cache(maxsize=1)
def _get_pipeline():
    """Built once per container process (mirrors the old @st.cache_resource)."""
    pipeline = load_pipeline(DISCIPLINE)
    _write_readme(pipeline[2].course_outline)
    return pipeline


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

# Per-turn pedagogical metrics + personalization-study sessions - own
# SQLite file on the same volume, written from on_message (see
# utils/metrics.py).
ensure_metrics_schema()

# Per-student personalization profile - own SQLite file on the same
# volume, refreshed once per session (see _load_or_refresh_profile).
ensure_profile_schema()

# How long on_chat_start will wait for the profiler before falling back to
# the stored profile. The refresh keeps running in the background past
# this and its result is saved for next time.
PROFILE_REFRESH_TIMEOUT_S = 12


@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo=f"sqlite+aiosqlite:///{DB_PATH}")


# ---------------- LOGIN ----------------
# Chainlit's password login, fed by the app's own page (public/login.html,
# reached through public/app.js): the "password" field carries
# the student's display name - not a real secret, not checked against
# anything. Matrícula is the actual identity, validated against the roster
# allowlist.
@cl.password_auth_callback
def auth_callback(username: str, password: str) -> cl.User | None:
    matricula = username.strip()
    nome = password.strip()

    if not nome or not is_enrolled(matricula, ROSTER):
        return None

    return cl.User(identifier=matricula, display_name=nome, metadata={"name": nome})


# ---------------- STUDENT PROFILE ----------------
def _current_matricula() -> str | None:
    user = cl.user_session.get("user")
    return user.identifier if user else None


def _safe_thread_id() -> str | None:
    try:
        return cl.context.session.thread_id
    except Exception:  # noqa: BLE001 - only used to exclude the current thread
        return None


def _course_profile_sync(matricula: str) -> tuple[StudentProfile, str | None, bool]:
    """
    The stored profile scoped to this deployment's course: (profile,
    last_profiled_thread, switched). After a course switch the old
    course's topic lists and tutor note are dropped - persisted right away
    so they can't come back (utils/student_profile.scope_to_course).
    """
    stored, last_profiled = load_profile(matricula)
    profile, switched = scope_to_course(stored, DISCIPLINE)

    if profile is not stored:
        save_profile(matricula, profile, last_profiled)

    return profile, last_profiled, switched


def _refresh_profile_sync(matricula: str, current_thread_id: str | None) -> StudentProfile:
    """
    Option A catch-up: load the stored profile, and if the student has a
    finished conversation in THIS course we haven't profiled yet, fold it
    in and persist. Runs off the event loop (asyncio.to_thread). Never
    raises - profiling must not break a session.
    """
    profile, last_profiled, switched = _course_profile_sync(matricula)

    # Untagged threads predate course tagging: they belong to the course
    # the profile was built in, which after a switch is not this one.
    found = latest_other_thread(
        DB_PATH,
        matricula,
        current_thread_id,
        course=DISCIPLINE,
        include_untagged=not switched,
    )

    if found is None:
        return profile

    thread_id, transcript = found

    if thread_id == last_profiled:
        return profile

    if not worth_profiling(transcript):
        # Too short to learn from: mark it seen so it is not reconsidered,
        # without counting it as an observation.
        save_profile(matricula, profile, thread_id)
        return profile

    updated = profile_student(profile, transcript, trajectory=thread_trajectory(thread_id))

    # The profiler can outlive on_chat_start's wait (shielded); if the
    # student answered the style question meanwhile, keep their answer.
    current, _ = load_profile(matricula)
    if current.style_asked:
        updated = updated.model_copy(
            update={"stated_style": current.stated_style, "style_asked": True}
        )

    save_profile(matricula, updated, thread_id)

    return updated


async def _load_or_refresh_profile(matricula: str | None) -> StudentProfile:
    if not matricula:
        return StudentProfile()

    current_thread_id = _safe_thread_id()

    task = asyncio.create_task(
        asyncio.to_thread(_refresh_profile_sync, matricula, current_thread_id)
    )

    try:
        # shield: a slow profiler keeps running (and saves) past the wait.
        return await asyncio.wait_for(
            asyncio.shield(task), timeout=PROFILE_REFRESH_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        logger.info(
            "Profiler slow for %s; using stored profile, refresh continues",
            matricula,
        )
    except Exception:  # noqa: BLE001 - degrade to stored profile
        logger.warning("Profile refresh failed for %s", matricula, exc_info=True)

    return (await asyncio.to_thread(_course_profile_sync, matricula))[0]


# ---------------- PERSONALIZATION STUDY ----------------
async def _start_study_session(
    matricula: str | None, profile: StudentProfile, resumed_thread_id: str | None = None
) -> tuple[dict, StudentProfile]:
    """
    Decide this conversation's personalization arm (utils/personalization.py)
    and return (study session record, profile the TUTOR gets).

    A resumed conversation keeps the arm it was recorded with; a new one
    takes the student's next session number. In the "off" arm the tutor
    gets an empty profile, but the record keeps the real one - the
    profiler keeps learning from every conversation either way.
    """
    stored = (
        await asyncio.to_thread(load_study_session, resumed_thread_id)
        if resumed_thread_id
        else None
    )

    if stored is not None:
        session_index, arm = stored["session_index"], stored["arm"]
    else:
        session_index = (
            await asyncio.to_thread(next_session_index, matricula) if matricula else 0
        )
        arm = assign_arm(matricula, session_index)

    session = {
        "thread_id": resumed_thread_id,
        "matricula": matricula,
        "session_index": session_index,
        "arm": arm,
        "profile": profile,
    }

    return session, (profile if arm == "on" else StudentProfile())


# ---------------- CHAT LIFECYCLE ----------------
@cl.on_chat_start
async def on_chat_start():
    graph, tutor_prompt, _config = _get_pipeline()

    matricula = _current_matricula()
    profile = await _load_or_refresh_profile(matricula)
    study_session, tutor_profile = await _start_study_session(matricula, profile)

    cl.user_session.set("graph", graph)
    cl.user_session.set("study_session", study_session)
    cl.user_session.set(
        "state",
        TutorState(
            messages=[tutor_prompt],
            student_profile=tutor_profile,
            personalization_on=study_session["arm"] == "on",
        ),  # type: ignore
    )

    cl.user_session.set("titled", False)
    cl.user_session.set("course_tagged", False)

    # First conversations only: ask how they like explanations, so the
    # tutor can adapt before the profiler has observed anything. The usual
    # greeting waits for the answer (on_style_choice) - showing both at
    # once buries the question under the greeting.
    if matricula and should_ask_style(profile):
        question = cl.Message(
            content=STYLE_QUESTION,
            actions=[
                cl.Action(name=STYLE_ACTION_NAME, payload={"style": value}, label=label, icon=icon)
                for label, value, icon in STYLE_CHOICES
            ],
        )
        await question.send()
        cl.user_session.set("style_message", question)
    else:
        await _send_welcome()


async def _send_welcome():
    user = cl.user_session.get("user")
    nome = (user.metadata or {}).get("name") if user else None
    greeting = f"Olá, {nome}! " if nome else "Olá! "

    # Ready-made starters ride on the greeting itself: Chainlit's own
    # @cl.set_starters only renders while the thread has no messages, and
    # this greeting is a message.
    welcome = cl.Message(
        content=f"{greeting}Tire suas dúvidas sobre **{DISCIPLINE}**.",
        actions=_actions(STARTERS),
    )
    await welcome.send()
    cl.user_session.set("buttons_message", welcome)


@cl.on_chat_resume
async def on_chat_resume(thread: cl.types.ThreadDict):
    """
    Rebuilds the LangGraph state from a resumed thread's message history,
    plus the learning/teaching state saved after its last turn
    (utils/chat_session.restore_tutor_state; frustration reset). Evidence
    is not restored - it is per turn anyway. The persistent
    student_profile IS restored (from its own store), but a resume does
    not trigger a profiler refresh. The
    conversation keeps its personalization arm (utils/personalization.py).
    """
    graph, tutor_prompt, _config = _get_pipeline()

    messages = [tutor_prompt]

    for step in thread.get("steps", []):
        # Keep the Chainlit step id on each message: an edit of a resumed
        # message arrives with that id (see truncate_for_edit).
        if step.get("type") == "user_message":
            messages.append(HumanMessage(content=step.get("output", ""), id=step.get("id")))
        elif step.get("type") == "assistant_message":
            messages.append(AIMessage(content=step.get("output", ""), id=step.get("id")))

    matricula = _current_matricula()
    profile = (
        (await asyncio.to_thread(_course_profile_sync, matricula))[0]
        if matricula
        else StudentProfile()
    )

    study_session, tutor_profile = await _start_study_session(
        matricula, profile, resumed_thread_id=thread.get("id")
    )

    # Continue where the conversation stopped (topic, progress, place in
    # the guided arc) instead of restarting the arc on the same topic.
    # Threads from before this was saved resume fresh.
    restored = restore_tutor_state(thread.get("metadata"))

    cl.user_session.set("graph", graph)
    cl.user_session.set("study_session", study_session)
    cl.user_session.set(
        "state",
        TutorState(
            messages=messages,
            student_profile=tutor_profile,
            personalization_on=study_session["arm"] == "on",
            **restored,
        ),  # type: ignore
    )

    await _restore_chips_and_buttons(thread, cl.user_session.get("state"))

    # Threads from before smart titles (or opened with a greeting / a
    # button click) still carry a useless name - retitle on the next turn.
    cl.user_session.set("titled", not needs_title(thread.get("name")))
    # A resumed thread keeps whatever course tag it has (or none, if it
    # predates tagging) - never relabel an old conversation.
    cl.user_session.set("course_tagged", True)


async def _restore_chips_and_buttons(thread: cl.types.ThreadDict, state):
    """
    Chainlit restores a resumed thread's messages but neither the PDF
    elements behind the citation chips (never persisted: no blob storage
    configured) nor the action buttons (never persisted at all). Rebuild
    both: each chip's PDF from its label in the text (utils.citations.
    find_ref_labels), and the last reply's buttons from the thread
    metadata saved after each turn (_persist_turn_metadata).
    """
    chat_steps = [
        s for s in thread.get("steps", [])
        if s.get("type") in ("user_message", "assistant_message")
    ]

    # Into the thread dict itself, not sent: Chainlit runs on_chat_resume
    # BEFORE it sends the thread to the browser, which then replaces the
    # page's elements with the thread's - wiping any sent earlier. And
    # elements arriving that way must carry their url already.
    session = cl.context.session
    elements = thread.get("elements") or []

    for step in chat_steps:
        if step["type"] != "assistant_message":
            continue
        for label, file_name, page in find_ref_labels(step.get("output", "")):
            pdf_path = resolve_source_pdf(DISCIPLINE, file_name)
            if pdf_path is None:
                continue
            try:
                stored = await session.persist_file(
                    name=label, path=str(pdf_path), content=None, mime="application/pdf"
                )
            except Exception:  # noqa: BLE001 - a missing chip beats a failed resume
                logger.warning("Could not restore citation %s", label, exc_info=True)
                continue
            element = cl.Pdf(
                name=label,
                url=f"/project/file/{stored['id']}?session_id={session.id}",
                display="side",
                page=page,
                mime="application/pdf",
            )
            element.for_id = step["id"]
            elements.append(element.to_dict())

    thread["elements"] = elements

    if not chat_steps or chat_steps[-1]["type"] != "assistant_message":
        return

    metadata = thread.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except ValueError:
            metadata = {}

    buttons = load_quick_replies((metadata or {}).get(QUICK_REPLIES_KEY))
    if buttons is None:
        # Saved before buttons were - or the thread is only the greeting.
        if not any(s["type"] == "user_message" for s in chat_steps):
            buttons = STARTERS
        else:
            buttons = quick_replies_for(state) if state is not None else []

    if not buttons:
        return

    last = chat_steps[-1]
    message = cl.Message(content=last.get("output", ""), id=last["id"])
    message.actions = _actions(buttons)
    for action in message.actions:
        await action.send(for_id=last["id"])
    cl.user_session.set("buttons_message", message)


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
            label = ref_label(file_name, page_start, metadata.get("page_end"))
            label_by_target[target] = label
            elements.append(
                cl.Pdf(
                    name=label,
                    path=str(pdf_path),
                    display="side",
                    page=page_start,
                )
            )

        answer = replace_with_label(answer, citation, label)

    return answer, elements


# ---------------- QUICK REPLIES ----------------
def _actions(replies: list[QuickReply]) -> list[cl.Action]:
    return [
        cl.Action(
            name=ACTION_NAME,
            payload={"message": r.message},
            label=r.label,
            icon=r.icon,
        )
        for r in replies
    ]


async def _clear_previous_buttons():
    """Buttons only make sense under the latest tutor message."""
    previous = cl.user_session.get("buttons_message")
    cl.user_session.set("buttons_message", None)

    if previous is not None:
        try:
            await previous.remove_actions()
        except Exception:  # noqa: BLE001 - cosmetic; message may be gone (edit)
            pass


@cl.action_callback(ACTION_NAME)
async def on_quick_reply(action: cl.Action):
    """A button click is posted as the student's own message and answered."""
    if cl.user_session.get("busy"):
        return  # double click while the previous turn is still running

    user = cl.user_session.get("user")
    message = cl.Message(
        content=action.payload.get("message", action.label),
        type="user_message",
        author=user.identifier if user else "Aluno",
    )
    await message.send()

    # Same "task running" UI state a typed message gets (input disabled,
    # stop button) - an action call doesn't set it on its own. And the same
    # "on_message" run step Chainlit wraps a typed message in (@cl.on_message
    # registers a wrapped copy but returns the bare function): the waiting
    # dots only show inside that step.
    await cl.context.emitter.task_start()
    try:
        async with cl.Step(name="on_message", type="run", parent_id=message.id):
            await on_message(message)
    finally:
        await cl.context.emitter.task_end()


# ---------------- THREAD TITLE ----------------
async def _rename_thread(name: str):
    """Rename the current thread, live in the sidebar. Cosmetic - never raises."""
    thread_id = _safe_thread_id()
    data_layer = _chainlit_data_layer()

    if not thread_id or data_layer is None:
        return

    try:
        await data_layer.update_thread(thread_id=thread_id, name=name)
        # The sidebar re-fetches the thread list whenever this value
        # changes - re-emitting it with the new name refreshes it live.
        await cl.context.emitter.emit(
            "first_interaction", {"interaction": name, "thread_id": thread_id}
        )
    except Exception:  # noqa: BLE001 - a title is cosmetic
        logger.warning("Could not update thread title", exc_info=True)


async def _set_thread_title(final_state):
    """Background task: summarize the first real question into the title."""
    if is_course_review(final_state.get("learning_state")):
        await _rename_thread(REVIEW_THREAD_NAME)
        return

    question = latest_student_question(final_state["messages"])
    learning_state = final_state.get("learning_state")
    topic = " / ".join(
        t for t in (learning_state.topic, learning_state.subtopic) if t
    ) if learning_state else None

    title = await asyncio.to_thread(generate_title, question or "", topic)

    if title:
        await _rename_thread(title)


# ---------------- THREAD METADATA ----------------
async def _persist_turn_metadata(final_state, buttons: list[QuickReply]):
    """
    After each turn, save the tutor's state in the thread metadata (for
    on_chat_resume) and, on a new thread's first turn, its course (so the
    profiler only reads conversations from the current course - see
    utils/student_profile.latest_other_thread).

    One write per turn, awaited: Chainlit's update_thread rewrites the
    whole metadata, so concurrent writers would drop each other's keys.
    The snapshot also goes into cl.user_session, which Chainlit itself
    persists into the same metadata on disconnect - keeping both in sync
    so that write never puts back an older one.
    """
    metadata = {QUICK_REPLIES_KEY: save_quick_replies(buttons)}
    cl.user_session.set(QUICK_REPLIES_KEY, metadata[QUICK_REPLIES_KEY])

    snapshot = tutor_state_snapshot(final_state)
    if snapshot is not None:
        metadata[TUTOR_STATE_KEY] = snapshot
        cl.user_session.set(TUTOR_STATE_KEY, snapshot)

    if not cl.user_session.get("course_tagged"):
        metadata[COURSE_METADATA_KEY] = DISCIPLINE

    thread_id = _safe_thread_id()
    data_layer = _chainlit_data_layer()

    if not metadata or not thread_id or data_layer is None:
        return

    try:
        await data_layer.update_thread(thread_id=thread_id, metadata=metadata)
        cl.user_session.set("course_tagged", True)
    except Exception:  # noqa: BLE001 - worst case: resume fresh / profiler skips it
        logger.warning("Could not save thread metadata", exc_info=True)


# ---------------- STYLE QUESTION ----------------
@cl.action_callback(STYLE_ACTION_NAME)
async def on_style_choice(action: cl.Action):
    """
    The student's answer to the first-conversation style question: stored
    as StudentProfile.stated_style, applied from this turn on (in the
    "on" arm). Not a chat turn - nothing goes through the tutor.
    """
    matricula = _current_matricula()
    if not matricula:
        return

    choice = action.payload.get("style", STYLE_SKIP)

    def _save() -> StudentProfile:
        profile, last_profiled, _ = _course_profile_sync(matricula)
        update = {"style_asked": True}
        if choice != STYLE_SKIP:
            update["stated_style"] = choice
        profile = profile.model_copy(update=update)
        save_profile(matricula, profile, last_profiled)
        return profile

    profile = await asyncio.to_thread(_save)

    study_session = cl.user_session.get("study_session") or {}
    study_session["profile"] = profile
    state = cl.user_session.get("state")
    if state is not None and study_session.get("arm") == "on":
        state["student_profile"] = profile

    # Chainlit names a thread after its first interaction - here the
    # button's action name ("style_choice") would sit in the sidebar until
    # the first real question is titled. Show a neutral name meanwhile.
    if not cl.user_session.get("titled"):
        await _rename_thread(NEW_THREAD_NAME)

    question = cl.user_session.get("style_message")
    cl.user_session.set("style_message", None)
    if question is not None:
        try:
            await question.remove_actions()
            question.content = f"{STYLE_QUESTION}\n\n✓ {STYLE_LABELS.get(choice, choice)}"
            await question.update()
        except Exception:  # noqa: BLE001 - cosmetic
            pass

    # The greeting held back by on_chat_start - unless the student skipped
    # the buttons and already started chatting.
    if state is None or len(state["messages"]) <= 1:
        await _send_welcome()


# ---------------- TURN ----------------
@cl.on_message
async def on_message(message: cl.Message):
    graph = cl.user_session.get("graph")
    state = cl.user_session.get("state")

    cl.user_session.set("busy", True)
    try:
        await _clear_previous_buttons()

        if truncate_for_edit(state["messages"], message.id):
            logger.info("Message edited; history truncated to the edit point")

        # The Chainlit message id travels with the LangChain message (the
        # graph's add_messages reducer keeps it) so a later edit can find it.
        state["messages"].append(HumanMessage(content=message.content, id=message.id))

        try:
            async with cl.Step(name="Pensando...", type="run"):
                final_state = await asyncio.to_thread(_run_graph_sync, graph, state)
        except Exception:
            logger.exception("Graph execution failed for a turn")
            # Drop the unanswered question so the student's retry starts clean.
            if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
                state["messages"].pop()
            await cl.Message(
                content=(
                    "Desculpe, tive um problema para processar sua pergunta agora. "
                    "Pode tentar de novo?"
                )
            ).send()
            return

        cl.user_session.set("state", final_state)

        greeted = final_state.get("greeted", False)

        answer = final_state["messages"][-1].content
        answer, elements = _linkify_citations(final_state, answer)

        buttons = quick_replies_for(final_state)
        reply = cl.Message(
            content=render_math(highlight_citations(answer)),
            elements=elements,
            actions=_actions(buttons),
        )
        await reply.send()
        cl.user_session.set("buttons_message", reply)

        # Metrics: synchronous SQLite write, kept off the shared event loop
        # like _run_graph_sync. Never raises (see record_turn). After the
        # send so the row carries the reply's id - chainlit.db's feedbacks
        # point at it. Skipped for the greeting fast path - a "bom dia"
        # isn't a pedagogical turn and would only add noise (often stale,
        # from the prior real turn still in the persisted state).
        if not greeted:
            study_session = cl.user_session.get("study_session") or {}
            if not study_session.get("thread_id"):
                study_session["thread_id"] = _safe_thread_id()
            await asyncio.to_thread(
                record_turn,
                final_state,
                DISCIPLINE,
                study_session,
                reply.id,
            )

        await _persist_turn_metadata(final_state, buttons)

        # Title from the first real question, not from a greeting - in the
        # background so the student never waits on it.
        if not greeted and not cl.user_session.get("titled"):
            cl.user_session.set("titled", True)
            asyncio.create_task(_set_thread_title(final_state))
        elif greeted and not cl.user_session.get("titled") and is_topics_request(message.content):
            # Only the topic list so far: otherwise the sidebar shows the
            # button's action name ("quick_reply") until a real question.
            # Still untitled, so that question replaces this name.
            await _rename_thread(TOPICS_THREAD_NAME)
    finally:
        cl.user_session.set("busy", False)
