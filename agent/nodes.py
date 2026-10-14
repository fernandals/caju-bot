import functools
import json
import re
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import agent.prompts as prompts
from agent.grader import ChunkAssessment
from agent.state import (
    AnswerPlan,
    ChunkEvidence,
    LearningState,
    StudentProfile,
    TeachingState,
    TutorConfig,
    TutorState,
)
from agent.familiarity import topic_familiarity
from agent.plan_profile import apply_profile_to_plan, leads_with_example
from agent.profile_notes import select_profile_notes
from agent.teaching import (
    is_course_review,
    advance_teaching_state,
    effective_familiarity,
    personalization_trace,
)
from rag.citation import format_citation
from rag.models import ChunkMetadata


# update_tracking's prompt is incremental - it updates the existing
# LearningState from the latest turns, with that state as its baseline
# truth - so it doesn't need the whole history. Cap the transcript it sees
# so a long session doesn't re-pay for every earlier turn on every message
# (planning and generation already window their message slices).
TRACKING_WINDOW = 12

# A course-wide review retrieves per outline topic, keeping this many
# chunks from each - so the study guide can cite every part of the course.
REVIEW_CHUNKS_PER_TOPIC = 2
REVIEW_TOPIC_KEY = "review_topic"

# A bare topic name matches title-only slide chunks best ("AVL",
# "Inclusão em Árvore AVL") - nothing to ground a summary on. Fetch more
# candidates per topic and skip chunks with less text than this (after
# the "Document/Section/Pages" header rag/splitter.py prepends).
REVIEW_CANDIDATES_PER_TOPIC = 20
REVIEW_MIN_BODY_CHARS = 150


def _chunk_body(doc) -> str:
    return doc.page_content.split("\n\n", 1)[-1]


def node(label: str):
    """
    Wraps a graph node with the START banner + execution-time print every
    node was repeating by hand.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            print(f"\n========== START {label} ==========")
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                print(f"[{label}] Execution time: {elapsed:.2f} seconds")

        return wrapper

    return decorator


def latest_student_question(messages) -> str | None:
    """Content of the most recent HumanMessage, or None if there is none."""
    return next(
        (msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)),
        None,
    )


def dialogue_window(messages, size: int) -> list:
    """
    The last `size` student/tutor messages, for passing to a chat model.

    Excludes the app-level SYSTEM_PROMPT that lives at messages[0]: in a
    short conversation a raw `messages[-8:]` slice still contains it, and
    as the LAST system message it overrode the node's own per-turn prompt
    ("teach through guidance" beat "DIRECT MODE: explain now"), so the
    tutor followed different rules in the first few turns than later on.
    """
    return [m for m in messages if isinstance(m, (HumanMessage, AIMessage))][-size:]


def build_conversation_transcript(messages) -> str:
    """
    Renders the student/tutor exchange as a clean {turn, role, content}
    transcript for LLM prompts, instead of interpolating raw BaseMessage
    objects (which stringify with noisy internals like ids/
    additional_kwargs/response_metadata and waste tokens).

    Only HumanMessage / AIMessage are included: the tutor's own system
    prompt lives at messages[0], and rendering it here would show up as a
    "tutor" turn - the tracking and planning models would then read the
    entire system prompt back as if it were dialogue.
    """

    exchange = [
        msg
        for msg in messages
        if isinstance(msg, (HumanMessage, AIMessage))
    ]

    conversation = [
        {
            "turn": i,
            "role": "student" if isinstance(msg, HumanMessage) else "tutor",
            "content": msg.content,
        }
        for i, msg in enumerate(exchange)
    ]

    return json.dumps(conversation, indent=2, ensure_ascii=False)


@node("TRACKING")
def update_tracking(state: TutorState, model):
    """
    Infers and updates the student's current pedagogical learning state
    from the recent conversation context.
    """

    # -------------------------
    # Conversation context
    # -------------------------
    conversation_text = build_conversation_transcript(
        state["messages"][-TRACKING_WINDOW:]
    )

    # -------------------------
    # Previous state
    # -------------------------
    previous_state = state.get("learning_state")

    previous_state_text = (
        previous_state.model_dump_json(indent=2)
        if previous_state
        else "None"
    )

    # -------------------------
    # Build prompt
    # -------------------------

    prompt = f"""
{prompts.TRACKING_PROMPT}

---

Previous learning state:
{previous_state_text}

---

Recent conversation:
{conversation_text}
"""

    # -------------------------
    # Structured output
    # -------------------------
    structured_llm = model.with_structured_output(LearningState)

    new_learning_state = structured_llm.invoke(prompt)

    if new_learning_state is None:
        # Structured parse failed. Tracking is advisory and the prompt
        # treats the previous state as baseline truth anyway, so carry it
        # forward rather than crashing the turn.
        print("[TRACKING] No structured output; keeping previous state.")
        new_learning_state = previous_state or LearningState()

    return {
        "learning_state": new_learning_state,
    }


@node("GREETING")
def greet(state: TutorState, config: TutorConfig, model):
    """
    Fast path for a turn that isn't a question yet - a greeting or small
    talk. Skips planning/retrieval/assessment so a bare "oi" doesn't pay
    for the whole pipeline, and (because it never reaches plan_instruction)
    doesn't nudge the teaching arc forward.
    """

    system = SystemMessage(
        content=(
            f"You are a friendly tutor for {config.subject}. The student "
            f"only greeted you or made small talk - they haven't asked "
            f"anything yet. Reply in {config.answer_language} with ONE "
            f"short, warm sentence that invites them to ask about the "
            f"subject. Do not ask how they are, do not teach or introduce "
            f"a topic, do not mention these instructions."
        )
    )

    response = model.invoke([system] + dialogue_window(state["messages"], 4))

    return {"messages": [response], "greeted": True}


@node("TOPICS")
def list_topics(state: TutorState, config: TutorConfig):
    """
    Answers "what can I study?" with the course outline (agent/outline.py,
    routed by agent/graph.route_start). Fixed text, no model call. Flagged
    as `greeted` like the greeting fast path: not a pedagogical turn, so
    no metrics row and no thread title, and it doesn't touch the arc.
    """
    topics = "\n".join(f"{i}. {t}" for i, t in enumerate(config.course_outline, 1))

    reply = AIMessage(
        content=(
            f"Estes são os principais assuntos de **{config.subject}** que "
            f"podemos estudar juntos:\n\n{topics}\n\n"
            f"Por qual deles você quer começar? Pode também me perguntar "
            f"direto sobre uma dúvida específica."
        )
    )

    return {"messages": [reply], "greeted": True}


@node("PLANNING")
def plan_instruction(state: TutorState, config: TutorConfig, model):
    """
    Determines the pedagogical response strategy for the current interaction.
    The node does NOT generate the final answer.
    It only produces a structured instructional plan that downstream
    nodes will execute.
    """

    learning_state = state["learning_state"]
    previous_teaching_state = state.get("teaching_state") or TeachingState()
    student_profile = state.get("student_profile")

    # Is this topic in the profile's solid/shaky lists? Deterministic word
    # matching (agent/familiarity.py), no LLM call.
    familiarity = topic_familiarity(learning_state, student_profile, config.subject)

    # False in the study's "off" arm: no in-session adaptation either.
    adapt = state.get("personalization_on", True)

    proposed_teaching_state = advance_teaching_state(
        previous_teaching_state,
        learning_state,
        student_profile,
        familiarity,
        adapt_in_session=adapt,
    )

    personalization = personalization_trace(
        previous_teaching_state,
        learning_state,
        student_profile,
        proposed_teaching_state,
        familiarity,
        adapt_in_session=adapt,
    )
    personalization["familiarity"] = effective_familiarity(familiarity, student_profile)
    personalization["profile_in_prompt"] = (
        student_profile is not None and render_student_profile(student_profile) != NO_PROFILE_TEXT
    )

    planning_context = f"""
Current learning state:
{learning_state.model_dump_json(indent=2)}

Proposed teaching stage (already decided by the system; see STRATEGY
RULES above for how to use it):
{proposed_teaching_state.model_dump_json(indent=2)}
"""

    prompt = f"""
{prompts.PLANNING_PROMPT}

---

{planning_context}

Recent conversation:
{build_conversation_transcript(state["messages"][-6:])}
"""

    structured_model = model.with_structured_output(AnswerPlan)

    answer_plan = structured_model.invoke(prompt)

    if answer_plan is None:
        # Structured parse failed - fall back to guided teaching with
        # retrieval, the system's default pacing, rather than crashing the
        # turn (route_after_planning would hit .needs_retrieval on None).
        print("[PLANNING] No structured output; using default plan.")
        answer_plan = AnswerPlan(
            needs_retrieval=True,
            strategy="guided_teaching",
            rationale="planner returned no structured output; defaulted to guided teaching",
        )

    # The only planning-time override of the deterministic pacing: the LLM
    # judged the student explicitly wants directness even though the
    # proposed stage was "guided". Rather than answering directly on the
    # spot, give it one turn as "refer" - point the student at the source
    # material instead - and only actually skip to the full explanation if
    # the deterministic pacing has already reached "deepen"/"wrap_up" (i.e.
    # the student didn't engage with a prior "refer" nudge, or the guided
    # arc concluded on its own). Gated on config.allow_direct_answers: when
    # a course wants guidance enforced, this specific escape hatch is
    # disabled (the frustration/exam_prep escape valves in
    # advance_teaching_state still apply regardless, since those are about
    # the student's wellbeing, not about skipping effort).
    teaching_state = proposed_teaching_state

    if (
        answer_plan.strategy == "direct_answer"
        and teaching_state.mode == "guided"
        and teaching_state.stage not in ("deepen", "wrap_up")
        and config.allow_direct_answers
    ):
        teaching_state = teaching_state.model_copy(
            update={"stage": "refer", "turns_in_stage": 0}
        )

    # The study guide is built from the material (retrieve_documents).
    if is_course_review(learning_state) and config.course_outline:
        answer_plan = answer_plan.model_copy(update={"needs_retrieval": True})

    # The profile's explanation style / topic familiarity, applied to the
    # plan in code on explanation turns (agent/plan_profile.py).
    answer_plan, plan_levers = apply_profile_to_plan(
        answer_plan, teaching_state, learning_state, student_profile, familiarity
    )
    personalization["levers"] += plan_levers
    personalization["plan_changed"] = bool(plan_levers)

    # What the profile knows about THIS topic, for the generator's turn
    # reminder (agent/profile_notes.py).
    profile_notes, note_levers = select_profile_notes(
        student_profile, learning_state, teaching_state, config.subject
    )
    personalization["levers"] += note_levers

    result = {
        "answer_plan": answer_plan,
        "teaching_state": teaching_state,
        "personalization": personalization,
        "profile_notes": profile_notes,
        # Clear the greeting flag a previous small-talk turn may have left
        # in the persisted session state - this is a real teaching turn.
        "greeted": False,
    }

    # When retrieval is skipped, the graph jumps straight to
    # generate_answer, which reads state["evidence"]. langgraph does NOT
    # apply the TutorState field defaults (it's a TypedDict), so that key
    # is either missing (KeyError on the first turn) or, across turns in a
    # persisted session, still holds the previous turn's evidence and
    # citations. Reset both so the no-retrieval path starts clean.
    if not answer_plan.needs_retrieval:
        result["evidence"] = []
        result["retrieved_docs"] = []

    return result

@node("DOCUMENT RETRIEVAL")
def retrieve_documents(state: TutorState, retriever, config: TutorConfig):
    """
    Retrieves relevant instructional documents based on the current learning
    state and the student's question.
    """

    learning_state = state["learning_state"]

    question = latest_student_question(state["messages"])

    if question is None:
        raise ValueError("No HumanMessage found during document retrieval.")

    # "Help me review for the exam" names no topic, so one query on it
    # brings back a few arbitrary chunks. Query each topic of the course
    # outline instead, in course order, for material from all of it.
    # Each chunk is tagged with the topic it was retrieved for, so
    # assess_documents grades it against that topic.
    if is_course_review(learning_state) and config.course_outline:
        vectorstore = retriever.vectorstore
        vectors = vectorstore.embeddings.embed_documents(config.course_outline)
        docs, seen = [], set()

        for topic, vector in zip(config.course_outline, vectors):
            candidates = vectorstore.similarity_search_by_vector(
                vector, k=REVIEW_CANDIDATES_PER_TOPIC
            )
            picked = [
                doc
                for doc in candidates
                if len(_chunk_body(doc).strip()) >= REVIEW_MIN_BODY_CHARS
                and doc.page_content not in seen
            ][:REVIEW_CHUNKS_PER_TOPIC]

            for doc in picked:
                seen.add(doc.page_content)
                docs.append(
                    doc.model_copy(
                        update={"metadata": {**doc.metadata, REVIEW_TOPIC_KEY: topic}}
                    )
                )

        return {
            "retrieved_docs": docs,
        }

    # Augment with topic/subtopic, which are genuinely part of what's being
    # asked about. Previously this also injected intent/strategy-derived
    # phrases (e.g. "exercises examples practice problems") into the
    # embedding query as a relevance "boost" - removed: there was no
    # evidence it improved retrieval, and stuffing unrelated keywords into
    # a semantic search query risks diluting it away from the actual
    # question rather than sharpening it.
    query_parts = [question]

    if learning_state.topic:
        query_parts.append(learning_state.topic)

    if learning_state.subtopic:
        query_parts.append(learning_state.subtopic)

    query = " ".join(query_parts) # type: ignore

    docs = retriever.invoke(query)

    return {
        "retrieved_docs": docs,
    }


@node("DOCUMENT ASSESSMENT")
def assess_documents(state: TutorState, config: TutorConfig, model):
    """
    Single pass over each retrieved chunk that BOTH grades its relevance
    and extracts grounded evidence - previously two separate LLM batches
    (grade_documents then extract_evidence) over the same chunks on every
    turn.

    Filters the chunks by relevance (same thresholds and fallback as
    before), then returns the survivors together with their evidence,
    kept in the same order so that retrieved_docs[i] lines up with
    evidence[i] for every downstream consumer (generate_answer, the
    Chainlit citation linkifier, the metrics recorder).
    """

    retrieved_docs = state.get("retrieved_docs", [])

    if not retrieved_docs:
        return {
            "retrieved_docs": [],
            "evidence": [],
        }

    question = latest_student_question(state["messages"])

    if question is None:
        raise ValueError("No HumanMessage found during document assessment.")

    learning_state = state["learning_state"]
    course_review = is_course_review(learning_state)

    # The review question itself ("help me review the main points")
    # matches no chunk in particular - graded against it, every chunk came
    # out "partially useful" and nearly all were dropped. Grade each one
    # against the course topic it was retrieved for instead.
    def review_note(doc) -> str:
        topic = doc.metadata.get(REVIEW_TOPIC_KEY)
        if not topic:
            return ""
        return (
            f"\nThe student is reviewing the whole course for an exam; this "
            f"chunk was retrieved for the course topic \"{topic}\". Score "
            f"how useful it is for reviewing THAT topic.\n"
        )

    assessor = model.with_structured_output(ChunkAssessment)

    system = SystemMessage(
        content=prompts.ASSESS_PROMPT.format(
            domain=config.subject
        )
    )

    inputs = [
        [
            system,
            HumanMessage(
                content=f"""
Student question:
{question}
{review_note(doc)}
Learning state:
{learning_state.model_dump_json(indent=2)}

Retrieved chunk:
{doc.page_content}
"""
            )
        ]
        for doc in retrieved_docs
    ]

    results: list[ChunkAssessment] = assessor.batch(inputs)

    # A chunk whose structured assessment failed to parse comes back as
    # None - drop it rather than let .relevance_score blow up the turn.
    scored_docs = sorted(
        (
            (doc, assessment)
            for doc, assessment in zip(retrieved_docs, results)
            if assessment is not None
        ),
        key=lambda pair: pair[1].relevance_score,
        reverse=True,
    )

    # threshold - lower for a course review: a "partially useful" chunk
    # (it introduces the topic) is still the right place to send the
    # student to study it, which is what that reply's citations are for.
    threshold = 0.4 if course_review else 0.5
    kept = [
        (doc, assessment)
        for doc, assessment in scored_docs
        if assessment.relevance_score >= threshold
    ]

    # A course review walks the topics in course order (the order they
    # were retrieved in), not by score.
    if course_review:
        order = {id(doc): i for i, doc in enumerate(retrieved_docs)}
        kept.sort(key=lambda pair: order[id(pair[0])])

    # fallback: only when the top scores are at least weakly relevant.
    # Below this floor, the chunks are clearly irrelevant and shouldn't be
    # forced into the answer just to have "something" to cite.
    if not kept:
        kept = [
            (doc, assessment)
            for doc, assessment in scored_docs[:3]
            if assessment.relevance_score >= 0.2
        ]

    filtered_docs = []
    evidence = []

    for i, (doc, assessment) in enumerate(kept, start=1):

        metadata = ChunkMetadata.model_validate(doc.metadata) # type: ignore

        filtered_docs.append(doc)
        evidence.append(
            ChunkEvidence(
                doc_id=f"DOC_{i}",
                citation=format_citation(metadata),
                evidence=assessment.evidence,
            )
        )

        print(
            f"[DOCUMENT ASSESSMENT] DOC_{i}: "
            f"score={assessment.relevance_score:.2f}, "
            f"evidence_items={len(assessment.evidence)}"
        )

    return {
        "retrieved_docs": filtered_docs,
        "evidence": evidence,
    }


@node("ANSWER GENERATION")
def generate_answer(state: TutorState, config: TutorConfig, model):
    """
    Generates the final answer to the student's question based on the
    current learning state, the instructional plan, and the retrieved
    documents.
    """

    question = latest_student_question(state["messages"])

    if question is None:
        raise ValueError("No HumanMessage found during answer generation.")

    learning_state = state["learning_state"]
    answer_plan = state["answer_plan"]
    teaching_state = state.get("teaching_state") or TeachingState()
    student_profile = state.get("student_profile") or StudentProfile()

    teaching_instructions = resolve_teaching_instructions(
        teaching_state,
        answer_plan.strategy,
        example_first=leads_with_example(answer_plan, teaching_state, student_profile),
        course_review=is_course_review(learning_state),
        course_outline=config.course_outline,
    )

    context = ""
    citations_by_doc_id = {}

    for ev in state.get("evidence", []):

        citations_by_doc_id[ev.doc_id] = ev.citation

        context += f"""
=======================================

SOURCE
{ev.doc_id}

CITE_AS
[[CITE:{ev.doc_id}]]

EVIDENCE
{chr(10).join("- "+e for e in ev.evidence)}
"""

    system_prompt = SystemMessage(
        content=prompts.GENERATE_PROMPT.format(
            domain=config.subject,
            question=question,
            learning_state=learning_state.model_dump_json(indent=2),
            student_profile=render_student_profile(student_profile),
            # needs_retrieval / confidence are routing signals - noise to
            # the generator, which only needs the "how to answer" fields.
            answer_plan=answer_plan.model_dump_json(
                indent=2,
                exclude={"needs_retrieval", "confidence"},
            ),
            teaching_instructions=teaching_instructions,
            context=context,
            answer_language=config.answer_language,
            max_sentences=config.max_sentences,
        )
    )

    conversation_window = dialogue_window(state["messages"], 8)

    # Repeat this turn's stage instructions right before the reply. Buried
    # mid-way through the long system prompt they lose to the pattern of
    # the recent dialogue: a single earlier tutor turn ending in a question
    # was enough for the model to keep asking questions in DIRECT MODE.
    reminder = (
        "Instructions for THIS reply (they take precedence over the "
        "pattern of earlier turns):\n" + teaching_instructions
    )

    # What the profile knows about this topic (picked in plan_instruction).
    # Here, at the end of the context, rather than in the long system
    # prompt - that is where the model actually follows instructions.
    profile_notes = state.get("profile_notes") or []
    if profile_notes:
        reminder += "\n\nAbout this student:\n" + "\n".join(f"- {n}" for n in profile_notes)

    turn_reminder = SystemMessage(content=reminder)

    response = model.invoke(
        [system_prompt]
        + conversation_window
        + [turn_reminder]
    )

    if isinstance(response.content, str):
        response = response.model_copy(
            update={
                "content": substitute_citation_markers(
                    response.content,
                    citations_by_doc_id,
                )
            }
        )

    return {
        "messages": [response]
    }


NO_PROFILE_TEXT = "(No profile yet - this is one of the first sessions with this student.)"


_STYLE_PHRASING = {
    "concise": "Tends to prefer a concise answer.",
    "detailed": "Tends to prefer a thorough, detailed explanation.",
    "example_first": "Grasps ideas better from a concrete example before the theory.",
    "step_by_step": "Tends to prefer reasoning laid out step by step.",
}


def render_student_profile(profile: StudentProfile) -> str:
    """Compact, prompt-ready rendering - only the parts we actually know."""
    parts: list[str] = []

    if profile.tutor_note:
        parts.append(profile.tutor_note)

    if profile.explanation_style in _STYLE_PHRASING:
        parts.append(_STYLE_PHRASING[profile.explanation_style])
    elif profile.stated_style in _STYLE_PHRASING:
        parts.append(f"Said so themselves: {_STYLE_PHRASING[profile.stated_style]}")

    if profile.prior_knowledge:
        parts.append(f"Knows from outside this course: {'; '.join(profile.prior_knowledge)}.")

    if profile.responds_to_guiding_questions == "poorly":
        parts.append("Does not respond well to being asked guiding questions - be more direct.")
    elif profile.responds_to_guiding_questions == "well":
        parts.append("Engages well with guiding questions.")

    if profile.solid_topics:
        parts.append(f"Already solid on: {', '.join(profile.solid_topics)}.")

    if profile.shaky_topics:
        parts.append(f"Has repeatedly struggled with: {', '.join(profile.shaky_topics)}.")

    if not parts:
        return NO_PROFILE_TEXT

    return " ".join(parts)


def resolve_teaching_instructions(
    teaching_state: TeachingState,
    strategy: str,
    example_first: bool = False,
    course_review: bool = False,
    course_outline: list[str] | None = None,
) -> str:
    """
    Picks the instruction block for generate_answer.

    Base: teaching_state decides pacing - "direct" mode, otherwise the
    current stage of the guided arc. On top of that, the planner's
    `strategy` can further shape the response:
      - exercise_first / hint_only replace the stage block entirely (they
        change what kind of reply this is), unless we're already in direct
        mode;
      - step_by_step just appends a formatting note.
    `example_first` (the student's profile, see agent/plan_profile.py)
    appends a note to open with the example.

    A course-wide review (agent/teaching.is_course_review) is a study guide
    over the course outline instead - it is always direct (exam_prep).
    """
    if course_review and course_outline:
        outline = "\n".join(f"{i}. {t}" for i, t in enumerate(course_outline, 1))
        return prompts.COURSE_REVIEW_INSTRUCTIONS.format(outline=outline)

    if teaching_state.mode == "direct":
        base = prompts.DIRECT_MODE_INSTRUCTIONS
    elif strategy in prompts.STRATEGY_OVERRIDE_INSTRUCTIONS:
        base = prompts.STRATEGY_OVERRIDE_INSTRUCTIONS[strategy]
    else:
        base = prompts.TEACHING_STAGE_INSTRUCTIONS[teaching_state.stage]

    if strategy == "step_by_step":
        base = f"{base}\n\n{prompts.STRATEGY_STEP_NOTE}"

    if example_first:
        base = f"{base}\n\n{prompts.EXAMPLE_FIRST_NOTE}"

    return base


# The generation model is told to emit citation markers in exactly the
# [[CITE:DOC_1]] form, but a smaller model routinely degrades that to
# [DOC_1], [CITE: DOC_1], (DOC_1) or a bare DOC_1. Match all of those so
# an internal id can never leak into the student-facing answer.
CITATION_MARKER_REGEX = re.compile(
    r"[\[\(]{1,2}\s*(?:CITE\s*[:=]?\s*)?DOC[ _-]?(\d+)\s*[\]\)]{1,2}"
    r"|(?<![\w-])DOC[ _-]?(\d+)(?![\w-])",
    re.IGNORECASE,
)

# Fenced blocks (``` ... ```, unclosed at the end included) and inline
# `code` spans - left untouched by substitute_citation_markers' tidying.
CODE_SPAN_REGEX = re.compile(r"(```.*?(?:```|$)|`[^`\n]+`)", re.DOTALL)


def substitute_citation_markers(
    text: str,
    citations_by_doc_id: dict[str, str],
) -> str:
    """
    Replaces the citation markers left by the generation model with the
    real, deterministically-built citation for that source.

    The model is deliberately never shown the human-readable citation text,
    so it can't accidentally translate, paraphrase, or otherwise alter it
    while writing the answer. A marker that points at no known source (a
    hallucinated DOC_9, or one whose chunk was filtered out) is removed.
    """

    def replace(match: re.Match) -> str:
        digits = match.group(1) or match.group(2)
        return citations_by_doc_id.get(f"DOC_{digits}", "")

    text = CITATION_MARKER_REGEX.sub(replace, text)

    # Tidy the gap left where a marker resolved to nothing (or sat
    # mid-sentence): "resposta ." -> "resposta.", collapse double spaces.
    # Prose only - inside code, spacing is content (indentation, `x .y`) -
    # and never a line's leading indentation (nested Markdown lists).
    def tidy(prose: str) -> str:
        prose = re.sub(r"[ \t]+([.,;:!?])", r"\1", prose)
        return re.sub(r"(?<=\S)[ \t]{2,}", " ", prose)

    parts = CODE_SPAN_REGEX.split(text)

    # split() with one capture group alternates prose, code, prose, ...
    return "".join(
        part if i % 2 else tidy(part) for i, part in enumerate(parts)
    )