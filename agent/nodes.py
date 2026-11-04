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
    ReplyVerdict,
    StudentProfile,
    TeachingState,
    TutorConfig,
    TutorState,
)
from agent.familiarity import core_words, topic_familiarity
from agent.plan_profile import apply_profile_to_plan, turn_style
from agent.style_tracking import arc_outcome, record_arc
from agent.profile_notes import select_profile_notes
from agent.teaching import (
    close_arc,
    continue_exercise,
    exercise_turn,
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

# Messages the judge reads (judge_reply): enough to see the student's
# earlier replies on the topic, not just the last one.
JUDGE_WINDOW = 8

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


# Minimum relevance_score (agent/grader.py) for a chunk's evidence to reach
# the answer. Tied to how the grading model scores: gpt-4.1-nano scored
# in coarse steps (about 0.4 or 0.7) and, at 0.5, dropped 29% of the
# chunks the retrieval eval's judge found useful; gpt-4.1-mini at 0.6
# keeps 88% of them with fewer useless ones (74% vs 66% precision), and
# reading the disagreements, its extra keeps were mostly right.
RELEVANCE_THRESHOLD = 0.6

# Lower for a course review: a "partially useful" chunk (it introduces
# the topic) is still the right place to send the student to study it,
# which is what that reply's citations are for.
REVIEW_RELEVANCE_THRESHOLD = 0.5


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


def without_course_topic(learning_state: LearningState, subject: str) -> LearningState:
    """
    The tracker used to name the course itself as the topic, with the
    real one in subtopic ("Estruturas de Dados Básicas II" / "Heap") - or
    with none at all, hiding that the student had not named a topic yet.
    A topic that is only the course's name (agent/familiarity.core_words:
    nothing left once its words are dropped) yields to the subtopic, or
    to None.
    """
    topic, subtopic = learning_state.topic, learning_state.subtopic

    while topic and not core_words(topic, subject):
        topic, subtopic = subtopic, None

    if (topic, subtopic) == (learning_state.topic, learning_state.subtopic):
        return learning_state

    return learning_state.model_copy(update={"topic": topic, "subtopic": subtopic})


@node("TRACKING")
def update_tracking(state: TutorState, model, config: TutorConfig | None = None):
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

    # The attempt at an exercise is often the first message naming the
    # topic - not a topic change for learning_progress (TRACKING_PROMPT).
    teaching_state = state.get("teaching_state")
    exercise_note = (
        "\nThe tutor's last message gave the student an exercise: their "
        "latest message is probably an attempt at it - judge it as one.\n"
        if teaching_state is not None and teaching_state.exercise_pending
        else ""
    )

    # The tutor asked which topic the exercise should be on: a bare "heap"
    # is the answer, and the topic.
    if teaching_state is not None and teaching_state.exercise_topic_pending:
        exercise_note = (
            "\nThe tutor's last message asked which topic the student wants "
            "an exercise on: their latest message is probably the answer - "
            "the topic (intent: practice).\n"
        )

    # Without the course's name, the tracker took it for the topic.
    course_text = ""
    if config is not None:
        course_text = f"\nCourse (never the topic itself): {config.subject}\n"
        if config.course_outline:
            course_text += "Course topic list:\n" + "\n".join(
                f"- {t}" for t in config.course_outline
            ) + "\n"

    prompt = f"""
{prompts.TRACKING_PROMPT}

---
{course_text}
Previous learning state:
{previous_state_text}
{exercise_note}
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

    if config is not None:
        new_learning_state = without_course_topic(new_learning_state, config.subject)

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
            f"You are {prompts.TUTOR_NAME}, a friendly tutor for {config.subject}. The student "
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


@node("ABOUT TUTOR")
def about_tutor(state: TutorState, config: TutorConfig):
    """
    Answers "who are you?" / "what does CAJU mean?" (routed by
    agent/graph.route_start). Fixed text, no model call - flagged `greeted`
    like list_topics: not a pedagogical turn.
    """
    reply = AIMessage(
        content=(
            f"Eu sou o **{prompts.TUTOR_NAME}** — **C**hat de **A**poio à "
            f"**J**ornada **U**niversitária! Estou aqui para te ajudar a "
            f"estudar **{config.subject}** com base no material da "
            f"disciplina: em vez de entregar a resposta pronta, vou te "
            f"guiando com perguntas, e cada explicação traz o link para a "
            f"página do material de onde ela veio.\n\n"
            f"Por onde você quer começar?"
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

    proposed_teaching_state = continue_exercise(
        previous_teaching_state,
        advance_teaching_state(
            previous_teaching_state,
            learning_state,
            student_profile,
            familiarity,
            adapt_in_session=adapt,
        ),
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

    # An explicit request to practise gets an exercise - on a topic the
    # student names: with none yet, the tutor asks which (teaching.exercise_turn),
    # which needs nothing from the material.
    exercise = exercise_turn(
        latest_student_question(state["messages"]), learning_state, previous_teaching_state
    )
    if exercise == "exercise":
        answer_plan = answer_plan.model_copy(update={"strategy": "exercise_first"})
    elif exercise == "ask_topic":
        answer_plan = answer_plan.model_copy(
            update={"strategy": "guided_teaching", "needs_retrieval": False}
        )

    if (
        answer_plan.strategy == "direct_answer"
        and teaching_state.mode == "guided"
        and teaching_state.stage not in ("deepen", "wrap_up", "concluded")
        and config.allow_direct_answers
    ):
        teaching_state = teaching_state.model_copy(
            update={"stage": "refer", "turns_in_stage": 0}
        )

    # The study guide is built from the material (retrieve_documents).
    if is_course_review(learning_state) and config.course_outline:
        answer_plan = answer_plan.model_copy(update={"needs_retrieval": True})

    # A guided arc that ended this turn counts for the style used in it;
    # enough stuck ones in a row switch the style (agent/style_tracking.py).
    tracker = state.get("style_tracker")
    outcome = arc_outcome(previous_teaching_state, teaching_state)
    if tracker is not None and outcome is not None:
        tracker, switched_to = record_arc(tracker, outcome)
        if switched_to:
            personalization["levers"].append(f"style_switch_{switched_to}")

    # The student's style / the profile's topic familiarity, applied to the
    # plan in code on explanation turns (agent/plan_profile.py).
    answer_plan, plan_levers = apply_profile_to_plan(
        answer_plan, teaching_state, learning_state, student_profile, familiarity, tracker
    )
    personalization["levers"] += plan_levers
    personalization["plan_changed"] = bool(plan_levers)

    # The style block the reply gets (agent/prompts.STYLE_INSTRUCTIONS).
    style = turn_style(answer_plan, teaching_state, tracker)
    if style:
        personalization["levers"].append(f"style_note_{style[0]}")

    # What the profile knows about THIS topic, for the generator's turn
    # reminder (agent/profile_notes.py).
    profile_notes, note_levers = select_profile_notes(
        student_profile, learning_state, teaching_state, config.subject
    )
    personalization["levers"] += note_levers

    # Remembered for the next turn, whose reply then checks the attempt.
    # An "exercises" student gets a task in every styled reply - but not
    # in the one asking which topic to practise.
    asks_topic = exercise == "ask_topic"
    teaching_state = teaching_state.model_copy(
        update={
            "exercise_pending": not asks_topic
            and (
                answer_plan.strategy == "exercise_first"
                or (style is not None and style[0] == "exercises")
            ),
            "exercise_requested": answer_plan.strategy == "exercise_first",
            "exercise_topic_pending": asks_topic,
        }
    )

    result = {
        "answer_plan": answer_plan,
        "teaching_state": teaching_state,
        "new_arc": (
            teaching_state.topic_anchor != previous_teaching_state.topic_anchor
            or previous_teaching_state.stage == "not_covered"
        ),
        "personalization": personalization,
        "profile_notes": profile_notes,
        "style_tracker": tracker,
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

    threshold = REVIEW_RELEVANCE_THRESHOLD if course_review else RELEVANCE_THRESHOLD
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
def generate_answer(state: TutorState, config: TutorConfig, model, review_model=None):
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

    # A new topic with nothing in the material is not taught.
    not_covered = is_not_covered(state)
    if not_covered:
        teaching_state = teaching_state.model_copy(
            update={
                "mode": "guided",
                "stage": "not_covered",
                "turns_in_stage": 0,
                "may_close": False,
                "exercise_pending": False,
                "exercise_requested": False,
            }
        )
        state = {**state, "teaching_state": teaching_state}

    # Judging the student's reply takes the stronger review model
    # (agent/models.py), like checking an exercise attempt below.
    verdict = judge_reply(state, context, review_model or model)

    teaching_instructions = resolve_teaching_instructions(
        teaching_state,
        answer_plan.strategy,
        style=turn_style(answer_plan, teaching_state, state.get("style_tracker")),
        course_review=is_course_review(learning_state),
        course_outline=config.course_outline,
        reviewing_attempt=teaching_state.reviewing_attempt,
        verdict=verdict,
    )

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

    reminder += "\n\n" + prompts.MATH_FORMAT_REMINDER

    turn_reminder = SystemMessage(content=reminder)

    # Checking an exercise attempt takes a stronger model (agent/models.py).
    if teaching_state.reviewing_attempt and review_model is not None:
        model = review_model

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

    updates = apply_verdict(state, verdict)
    if not_covered:
        updates["teaching_state"] = teaching_state

    return {"messages": [response], **updates}


def is_not_covered(state: TutorState) -> bool:
    """
    A topic starting this turn that the material has nothing on: retrieval
    was asked for and no chunk passed assessment. Only at the start of a
    topic - mid-topic, a "não sei" or an aside can retrieve nothing on a
    topic that is covered. Not for course reviews, replies without a topic
    yet, or an attempt at an exercise.
    """
    teaching_state = state.get("teaching_state")
    learning_state = state.get("learning_state")
    plan = state.get("answer_plan")

    return bool(
        state.get("new_arc")
        and plan is not None
        and plan.needs_retrieval
        and not state.get("evidence")
        and learning_state is not None
        and learning_state.topic
        and not is_course_review(learning_state)
        and teaching_state is not None
        and not teaching_state.reviewing_attempt
    )


def is_judged_turn(teaching_state: TeachingState) -> bool:
    """A guided "check" turn: the student is replying to a question - not
    asking for an exercise the tutor first asks the topic of."""
    return (
        teaching_state.mode == "guided"
        and teaching_state.stage == "check"
        and not teaching_state.exercise_topic_pending
    )


def judge_reply(state: TutorState, context: str, model) -> ReplyVerdict | None:
    """
    The tutor's verdict on the student's reply (prompts.JUDGE_PROMPT), on
    judged turns only; None elsewhere or if the call fails - the reply is
    then written as before, and nothing closes.
    """
    teaching_state = state.get("teaching_state")
    if teaching_state is None or not is_judged_turn(teaching_state):
        return None

    dialogue = dialogue_window(state["messages"], JUDGE_WINDOW)
    if len(dialogue) < 2 or not isinstance(dialogue[-2], AIMessage):
        return None

    learning_state = state.get("learning_state")
    topic = " - ".join(t for t in (learning_state.topic, learning_state.subtopic) if t) if learning_state else ""

    try:
        judged = model.with_structured_output(ReplyVerdict).invoke(
            prompts.JUDGE_PROMPT.format(
                topic=topic or "the current topic",
                dialogue=build_conversation_transcript(dialogue),
                context=context or "(none retrieved)",
            )
        )
    except Exception as exc:  # noqa: BLE001 - an unjudged reply is still answered
        print(f"[ANSWER GENERATION] Could not judge the reply: {exc}")
        return None

    if judged is not None:
        print(f"[ANSWER GENERATION] Verdict: {judged.verdict}, core idea: {judged.shows_core_idea} - {judged.reason}")

    return judged


def closes_topic(teaching_state: TeachingState, verdict: ReplyVerdict | None) -> bool:
    """
    A reply concludes the topic only when it is right AND, with the earlier
    replies, shows the central idea - a right answer to one small step does
    not - and the arc allows closing (TeachingState.may_close).
    """
    return (
        verdict is not None
        and is_judged_turn(teaching_state)
        and teaching_state.may_close
        and verdict.verdict == "correct"
        and verdict.shows_core_idea
    )


# The tracker's label, corrected by the tutor's verdict when they disagree.
_PROGRESS_FOR_VERDICT = {"correct": "improving", "partial": "improving", "wrong": "stable"}


def apply_verdict(state: TutorState, verdict: ReplyVerdict | None) -> dict:
    """
    State updates from the verdict on a "check" reply (judge_reply): a reply
    that closes the topic (closes_topic) concludes it (teaching.close_arc)
    and counts as a mastered arc for the student's style; otherwise the
    tracker's learning_progress - judged without the material - yields to
    the verdict. No verdict (other stages, or the call failed): nothing.
    """
    teaching_state = state.get("teaching_state")
    learning_state = state.get("learning_state")

    if verdict is None or verdict.verdict == "none" or teaching_state is None or not is_judged_turn(teaching_state):
        return {}

    if closes_topic(teaching_state, verdict):
        updates = {
            "teaching_state": close_arc(teaching_state),
            "learning_state": learning_state.model_copy(update={"learning_progress": "mastered"}),
        }
        tracker = state.get("style_tracker")
        if tracker is not None:
            updates["style_tracker"], _ = record_arc(tracker, "mastered")
        return updates

    progress = _PROGRESS_FOR_VERDICT[verdict.verdict]
    if learning_state.learning_progress == "stuck" and verdict.verdict == "wrong":
        progress = "stuck"
    return {"learning_state": learning_state.model_copy(update={"learning_progress": progress})}


NO_PROFILE_TEXT = "(No profile yet - this is one of the first sessions with this student.)"


def render_student_profile(profile: StudentProfile) -> str:
    """Compact, prompt-ready rendering - only the parts we actually know."""
    parts: list[str] = []

    if profile.tutor_note:
        parts.append(profile.tutor_note)

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


# Styles whose first question on a topic replaces the "introduce" stage's
# name cue (prompts.INTRODUCE_STYLED); step_by_step keeps it.
_CUE_STYLES = {"examples", "exercises"}


def _with_outline(instructions: str, course_outline: list[str] | None) -> str:
    """The course topics, for a reply that suggests what to study next."""
    if not course_outline:
        return instructions
    outline = "\n".join(f"- {t}" for t in course_outline)
    return f"{instructions}\n\nCourse topics:\n{outline}"


def resolve_teaching_instructions(
    teaching_state: TeachingState,
    strategy: str,
    style: tuple[str, str] | None = None,
    course_review: bool = False,
    course_outline: list[str] | None = None,
    reviewing_attempt: bool = False,
    verdict: ReplyVerdict | None = None,
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
    `style` - (explanation style, "guiding" / "explaining"), see
    agent/plan_profile.turn_style - appends that style's block.

    A course-wide review (agent/teaching.is_course_review) is a study guide
    over the course outline instead - it is always direct (exam_prep).
    """
    if course_review and course_outline:
        outline = "\n".join(f"{i}. {t}" for i, t in enumerate(course_outline, 1))
        return prompts.COURSE_REVIEW_INSTRUCTIONS.format(outline=outline)

    # Nothing in the material (is_not_covered): whatever else was planned.
    if teaching_state.stage == "not_covered":
        return _with_outline(prompts.TEACHING_STAGE_INSTRUCTIONS["not_covered"], course_outline)

    # An exercise asked for with no topic named (teaching.exercise_turn).
    if teaching_state.exercise_topic_pending:
        return _with_outline(prompts.ASK_EXERCISE_TOPIC_INSTRUCTIONS, course_outline)

    # A correct reply where closing is allowed: the topic is done.
    if closes_topic(teaching_state, verdict):
        return _with_outline(prompts.CLOSE_TOPIC_INSTRUCTIONS, course_outline)

    if teaching_state.mode == "direct":
        base = prompts.DIRECT_MODE_INSTRUCTIONS
    elif strategy in prompts.STRATEGY_OVERRIDE_INSTRUCTIONS:
        base = prompts.STRATEGY_OVERRIDE_INSTRUCTIONS[strategy]
    elif teaching_state.stage == "introduce" and style and style[0] in _CUE_STYLES:
        base = prompts.INTRODUCE_STYLED
    else:
        base = prompts.TEACHING_STAGE_INSTRUCTIONS[teaching_state.stage]

    if strategy == "step_by_step":
        base = f"{base}\n\n{prompts.STRATEGY_STEP_NOTE}"

    if style:
        base = f"{base}\n\n{prompts.STYLE_INSTRUCTIONS[style[0]][style[1]]}"

    if teaching_state.stage == "concluded":
        base = _with_outline(base, course_outline)

    if is_judged_turn(teaching_state) and verdict is not None:
        base = f"{base}\n\n{prompts.VERDICT_NOTES[verdict.verdict]}"

    # First, so it frames whatever the stage asks for next.
    if reviewing_attempt:
        base = f"{prompts.ATTEMPT_REVIEW_NOTE}\n\n{base}"

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