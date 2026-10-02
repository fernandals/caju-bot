"""
LLM eval for the pedagogical state the teaching arc runs on.

Unlike tests/ (pure logic, no API key), this calls the real tracking and
planning models over short scripted conversations and checks the labels
they produce against what agent/teaching.py needs to see. Every case is
run several times, since the labels are sampled.

    python -m evals.tracking_eval                 # default models
    TRACKING_MODEL=gpt-4.1-nano python -m evals.tracking_eval
    RUNS=5 ONLY=progress python -m evals.tracking_eval

Costs a few cents per run.
"""

import collections
import contextlib
import io
import os
from dataclasses import dataclass
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agent.nodes import plan_instruction, update_tracking
from agent.state import LearningState, TeachingState, TutorConfig

H, A = HumanMessage, AIMessage

PF = dict(topic="Arquitetura de Software", subtopic="Pipe-Filter")
CS = dict(topic="Arquitetura de Software", subtopic="Cliente-Servidor")


@dataclass
class Case:
    group: str
    name: str
    previous: LearningState
    messages: list
    # field -> (description, predicate over the produced value)
    checks: dict[str, tuple[str, Callable]]


def is_(*allowed):
    return (" | ".join(map(str, allowed)), lambda v: v in allowed)


def not_(*banned):
    return ("not " + " | ".join(map(str, banned)), lambda v: v not in banned)


def gt(x):
    return (f"> {x}", lambda v: v > x)


def lt(x):
    return (f"< {x}", lambda v: v < x)


def same_as(value):
    return (f"== {value!r} (unchanged)", lambda v: v == value)


def changed_from(value):
    return (f"!= {value!r}", lambda v: v != value)


NONE = ("None", lambda v: v is None)
SET = ("set", lambda v: bool(v))


INTRO_PF = [
    H("o que é o estilo pipe-filter?"),
    A("Pelo nome, o que você acha que fazem os 'filters' e os 'pipes' num sistema?"),
]

CASES = [
    # ---------------- learning_progress ----------------
    Case("progress", "first question is stable, not stuck",
         LearningState(),
         [H("o que é o estilo pipe-filter?")],
         {"learning_progress": is_("stable"), "comprehension_level": is_("medium")}),

    Case("progress", "one 'não sei' is not stuck yet",
         LearningState(**PF),
         INTRO_PF + [H("não sei")],
         {"learning_progress": not_("stuck", "mastered", "improving")}),

    Case("progress", "partial answer -> improving",
         LearningState(**PF),
         INTRO_PF + [H("acho que os filtros processam os dados? os pipes não sei")],
         {"learning_progress": is_("improving")}),

    Case("progress", "single wrong answer -> not stuck/mastered",
         LearningState(**PF, learning_progress="improving"),
         INTRO_PF + [
             H("filtros transformam e pipes levam os dados"),
             A("Isso! E um filtro precisa conhecer o filtro anterior ou o próximo?"),
             H("sim, eles compartilham uma memória global pra saber o estado dos outros"),
         ],
         {"learning_progress": is_("stable"), "current_difficulty": SET}),

    Case("progress", "corrects own mistake after hint -> improving",
         LearningState(**PF, learning_progress="stable",
                       current_difficulty="acha que filtros compartilham estado"),
         INTRO_PF + [
             H("filtros transformam e pipes levam os dados"),
             A("Isso! E um filtro precisa conhecer o filtro anterior ou o próximo?"),
             H("sim, eles compartilham memória"),
             A("Pense no `cat arquivo | grep erro` do Unix: o grep sabe que quem mandou os dados foi o cat?"),
             H("ah não, ele só lê o que chega na entrada. então os filtros são independentes"),
         ],
         {"learning_progress": is_("improving", "mastered")}),

    Case("progress", "re-asks instead of engaging -> stuck",
         LearningState(**PF),
         INTRO_PF + [
             H("mas o que é pipe-filter?"),
             A("Tenta um chute: se 'filter' filtra algo, o que ele faria com os dados que recebe?"),
             H("não sei, só me explica o que é pipe-filter"),
         ],
         {"learning_progress": is_("stuck")}),

    Case("progress", "explicitly lost -> stuck",
         LearningState(**PF, learning_progress="stable"),
         INTRO_PF + [
             H("não sei"),
             A("Tudo bem! Pensa numa linha de montagem: cada estação faz uma coisa e passa adiante. O que seria a estação?"),
             H("continuo perdido, não entendi nada disso"),
         ],
         {"learning_progress": is_("stuck"), "comprehension_level": is_("low")}),

    Case("progress", "explains concept back -> mastered",
         LearningState(**PF, learning_progress="improving"),
         INTRO_PF + [
             H("filtros transformam os dados e pipes levam de um filtro pro outro"),
             A("Exato! E um filtro precisa saber quem está antes ou depois dele?"),
             H("não, os filtros são independentes e não compartilham estado, por isso dá pra trocar ou reordenar eles, tipo cat | grep | sort no unix"),
         ],
         {"learning_progress": is_("mastered"), "comprehension_level": is_("high"),
          "current_difficulty": NONE}),

    Case("progress", "three short correct answers -> mastered",
         LearningState(**CS, learning_progress="improving"),
         [
             H("me explica cliente-servidor"),
             A("Quem você acha que inicia a comunicação: o cliente ou o servidor?"),
             H("o cliente, ele faz a requisição"),
             A("Isso. E o servidor conhece os clientes de antemão?"),
             H("não, ele só espera requisições e responde"),
             A("Certo! E se muitos clientes chegarem ao mesmo tempo, onde fica o gargalo?"),
             H("no servidor, que é centralizado. dá pra escalar replicando servidores"),
         ],
         {"learning_progress": is_("mastered")}),

    Case("progress", "difficulty cleared once past it",
         LearningState(**PF, learning_progress="improving",
                       current_difficulty="acha que o pipe armazena os dados"),
         INTRO_PF + [
             H("o pipe guarda os dados?"),
             A("Pensa num cano de água: ele guarda a água ou só leva de um lugar pro outro?"),
             H("ah, só leva. então o pipe só transporta os dados de um filtro pro outro, não armazena"),
         ],
         {"current_difficulty": NONE}),

    # ---------------- frustration ----------------
    Case("frustration", "calm engaged student stays low",
         LearningState(**PF),
         INTRO_PF + [H("hmm acho que os filtros processam alguma coisa, né?")],
         {"frustration_level": lt(0.3)}),

    Case("frustration", "impatient repeat crosses escape threshold (0.6)",
         LearningState(**PF, frustration_level=0.3, learning_progress="stable"),
         INTRO_PF + [
             H("não sei, me explica"),
             A("Tenta um chute: o que um filtro faria com os dados?"),
             H("já falei que não sei!! só me diz logo o que é, to perguntando isso pela terceira vez"),
         ],
         {"frustration_level": gt(0.6)}),

    Case("frustration", "drops once student gets it",
         LearningState(**PF, frustration_level=0.7, learning_progress="stuck"),
         INTRO_PF + [
             H("não entendo nada disso"),
             A("Pensa no `cat log | grep erro` do Unix. O grep transforma o que recebe?"),
             H("ahh entendi! o grep é um filtro, ele recebe o texto e deixa passar só as linhas com erro, e o | é o pipe"),
         ],
         {"frustration_level": lt(0.7), "learning_progress": not_("stuck")}),

    # ---------------- intent ----------------
    Case("intent", "exam tomorrow -> exam_prep",
         LearningState(),
         [H("tenho prova amanhã, me resume rapidinho o estilo pipe-filter")],
         {"intent": is_("exam_prep")}),

    Case("intent", "asks for exercise -> practice",
         LearningState(**CS),
         [H("me passa um exercício sobre cliente-servidor pra eu treinar")],
         {"intent": is_("practice")}),

    Case("intent", "names a misconception -> debug_confusion",
         LearningState(),
         [H("eu achava que pipe e filter eram a mesma coisa, não entendi a diferença")],
         {"intent": is_("debug_confusion")}),

    Case("intent", "plain question -> learn",
         LearningState(),
         [H("o que é a arquitetura blackboard?")],
         {"intent": is_("learn")}),

    # ---------------- topic anchor ----------------
    # agent/teaching.py restarts the arc whenever (topic, subtopic) changes,
    # so ANY rewording of these on a same-topic turn resets the student to
    # "introduce" and throws away turns_in_stage.
    Case("topic", "follow-up on same concept keeps anchor",
         LearningState(**PF, learning_progress="improving"),
         INTRO_PF + [
             H("filtros transformam os dados e pipes levam"),
             A("Isso! E quais você acha que são as desvantagens desse estilo?"),
             H("talvez ficar lento se tiver muitos filtros?"),
         ],
         {"topic": same_as(PF["topic"]), "subtopic": same_as(PF["subtopic"])}),

    Case("topic", "answer mentioning a sub-aspect keeps anchor",
         LearningState(**PF, learning_progress="improving"),
         INTRO_PF + [
             H("filtros transformam e pipes transportam"),
             A("Isso! E um filtro precisa conhecer os vizinhos?"),
             H("não, os filtros são independentes e não compartilham estado entre si"),
         ],
         {"topic": same_as(PF["topic"]), "subtopic": same_as(PF["subtopic"])}),

    Case("topic", "student question on a detail keeps anchor",
         LearningState(**CS, learning_progress="stable"),
         [
             H("me explica cliente-servidor"),
             A("Quem você acha que inicia a comunicação?"),
             H("o cliente. mas e se o servidor cair, o que acontece?"),
         ],
         {"topic": same_as(CS["topic"]), "subtopic": same_as(CS["subtopic"])}),

    Case("topic", "explicit switch changes anchor and resets progress",
         LearningState(**PF, learning_progress="mastered", comprehension_level="high"),
         INTRO_PF + [
             H("filtros transformam e pipes transportam, e são independentes"),
             A("Perfeito! Resumindo: filtros independentes ligados por pipes."),
             H("show. agora quero aprender sobre blackboard"),
         ],
         {"subtopic": changed_from(PF["subtopic"]), "learning_progress": is_("stable"),
          "current_difficulty": NONE}),
]


# Planner: the only LLM-driven override of the arc. "Just tell me" should be
# planned as direct_answer, which plan_instruction turns into a "refer" turn;
# a normal question must not be.
PLANNER_CASES = [
    ("asks to just be told -> refer",
     LearningState(**PF),
     TeachingState(topic_anchor=(PF["topic"], PF["subtopic"]), stage="introduce"),
     INTRO_PF + [H("não quero adivinhar, só me fala direto o que é")],
     "refer"),
    ("answers guiding question -> stays guided",
     LearningState(**PF, learning_progress="improving"),
     TeachingState(topic_anchor=(PF["topic"], PF["subtopic"]), stage="introduce"),
     INTRO_PF + [H("acho que os filtros processam os dados e os pipes levam pro próximo")],
     "check"),
]


def main():
    runs = int(os.getenv("RUNS", 3))
    only = os.getenv("ONLY")
    tracking_model = os.getenv("TRACKING_MODEL", "gpt-4.1-mini")
    planning_model = os.getenv("PLANNING_MODEL", "gpt-4.1-mini")
    tracker = ChatOpenAI(model=tracking_model, temperature=0, timeout=60, max_retries=2)
    planner = ChatOpenAI(model=planning_model, temperature=0, timeout=60, max_retries=2)

    print(f"tracking={tracking_model} planning={planning_model} runs={runs}\n")

    total = passed = 0
    failures = []

    for case in CASES:
        if only and case.group != only:
            continue
        values = collections.defaultdict(list)
        for _ in range(runs):
            with contextlib.redirect_stdout(io.StringIO()):
                out = update_tracking(
                    {"messages": case.messages, "learning_state": case.previous},
                    model=tracker,
                )
            ls = out["learning_state"]
            for field in case.checks:
                values[field].append(getattr(ls, field))

        for field, (desc, ok) in case.checks.items():
            hits = sum(ok(v) for v in values[field])
            total += runs
            passed += hits
            mark = "PASS" if hits == runs else ("FLAKY" if hits else "FAIL")
            print(f"{mark:5} [{case.group}] {case.name} :: {field} {desc}  ({hits}/{runs})")
            if hits < runs:
                failures.append((case, field, desc, values[field]))

    if not only or only == "planner":
        config = TutorConfig(subject="Software Architecture")
        for name, ls, ts, msgs, expected_stage in PLANNER_CASES:
            stages = []
            for _ in range(runs):
                with contextlib.redirect_stdout(io.StringIO()):
                    out = plan_instruction(
                        {"messages": msgs, "learning_state": ls, "teaching_state": ts},
                        config=config, model=planner,
                    )
                stages.append(out["teaching_state"].stage)
            hits = stages.count(expected_stage)
            total += runs
            passed += hits
            mark = "PASS" if hits == runs else ("FLAKY" if hits else "FAIL")
            print(f"{mark:5} [planner] {name} :: stage == {expected_stage}  ({hits}/{runs})")
            if hits < runs:
                failures.append((name, "stage", expected_stage, stages))

    print(f"\n{passed}/{total} checks passed")
    if failures:
        print("\nFailures (expected -> got):")
        for case, field, desc, got in failures:
            label = case.name if isinstance(case, Case) else case
            got_fmt = [round(g, 2) if isinstance(g, float) else g for g in got]
            print(f"  - {label} :: {field}: {desc} -> {got_fmt}")


if __name__ == "__main__":
    main()
