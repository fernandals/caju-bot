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
from agent.teaching import advance_teaching_state
from rag.citation import format_citation
from rag.models import ChunkMetadata


# update_tracking's prompt is incremental - it updates the existing
# LearningState from the latest turns, with that state as its baseline
# truth - so it doesn't need the whole history. Cap the transcript it sees
# so a long session doesn't re-pay for every earlier turn on every message
# (planning and generation already window their message slices).
TRACKING_WINDOW = 12


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


@node("PLANNING")
def plan_instruction(state: TutorState, config: TutorConfig, model):
    """
    Determines the pedagogical response strategy for the current interaction.
    The node does NOT generate the final answer.
    It only produces a structured instructional plan that downstream
    nodes will execute.
    """

    learning_state = state["learning_state"]

    proposed_teaching_state = advance_teaching_state(
        state.get("teaching_state") or TeachingState(),
        learning_state,
        state.get("student_profile"),
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
    # proposed stage was "guided". Keep topic_anchor/stage intact so a
    # guided arc can resume next turn if the student re-engages. Gated on
    # config.allow_direct_answers: when a course wants guidance enforced,
    # this specific escape hatch is disabled (the frustration/exam_prep
    # escape valves in advance_teaching_state still apply regardless, since
    # those are about the student's wellbeing, not about skipping effort).
    teaching_state = proposed_teaching_state

    if (
        answer_plan.strategy == "direct_answer"
        and teaching_state.mode == "guided"
        and config.allow_direct_answers
    ):
        teaching_state = teaching_state.model_copy(update={"mode": "direct"})

    result = {
        "answer_plan": answer_plan,
        "teaching_state": teaching_state,
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
def retrieve_documents(state: TutorState, retriever):
    """
    Retrieves relevant instructional documents based on the current learning
    state and the student's question.
    """

    learning_state = state["learning_state"]

    question = latest_student_question(state["messages"])

    if question is None:
        raise ValueError("No HumanMessage found during document retrieval.")

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

    # threshold
    kept = [
        (doc, assessment)
        for doc, assessment in scored_docs
        if assessment.relevance_score >= 0.5
    ]

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
        teaching_state, answer_plan.strategy
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

    conversation_window = state["messages"][-8:]

    response = model.invoke(
        [system_prompt]
        + conversation_window
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

    if profile.responds_to_guiding_questions == "poorly":
        parts.append("Does not respond well to being asked guiding questions - be more direct.")
    elif profile.responds_to_guiding_questions == "well":
        parts.append("Engages well with guiding questions.")

    if profile.solid_topics:
        parts.append(f"Already solid on: {', '.join(profile.solid_topics)}.")

    if profile.shaky_topics:
        parts.append(f"Has repeatedly struggled with: {', '.join(profile.shaky_topics)}.")

    if not parts:
        return "(No profile yet - this is one of the first sessions with this student.)"

    return " ".join(parts)


def resolve_teaching_instructions(teaching_state: TeachingState, strategy: str) -> str:
    """
    Picks the instruction block for generate_answer.

    Base: teaching_state decides pacing - "direct" mode, otherwise the
    current stage of the guided arc. On top of that, the planner's
    `strategy` can further shape the response:
      - exercise_first / hint_only replace the stage block entirely (they
        change what kind of reply this is), unless we're already in direct
        mode;
      - step_by_step just appends a formatting note.
    """
    if teaching_state.mode == "direct":
        base = prompts.DIRECT_MODE_INSTRUCTIONS
    elif strategy in prompts.STRATEGY_OVERRIDE_INSTRUCTIONS:
        base = prompts.STRATEGY_OVERRIDE_INSTRUCTIONS[strategy]
    else:
        base = prompts.TEACHING_STAGE_INSTRUCTIONS[teaching_state.stage]

    if strategy == "step_by_step":
        return f"{base}\n\n{prompts.STRATEGY_STEP_NOTE}"

    return base


CITATION_MARKER_REGEX = re.compile(r"\[\[CITE:(DOC_\d+)\]\]")

def substitute_citation_markers(
    text: str,
    citations_by_doc_id: dict[str, str],
) -> str:
    """
    Replaces opaque [[CITE:DOC_n]] markers left by the generation model
    with the real, deterministically-built citation for that source.

    The model is deliberately never shown the human-readable citation text,
    so it can't accidentally translate, paraphrase, or otherwise alter it
    while writing the answer.
    """

    def replace(match: re.Match) -> str:
        return citations_by_doc_id.get(match.group(1), "")

    return CITATION_MARKER_REGEX.sub(replace, text)