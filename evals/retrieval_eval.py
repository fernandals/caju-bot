"""
Retrieval eval for any course's knowledge base - how often the app's
retriever brings back the material that answers a student's question.

Two steps:

    # 1. Once per course: questions written by a model from the PDFs -
    #    "specific" ones from sampled pages, tagged with that (file, page),
    #    and one "broad" one per file about its main topic ("me explica
    #    árvores AVL"), tagged with the file.
    python -m evals.retrieval_eval generate <pdf_folder> data/evals/<course>.json

    # 2. Score one or more KB folders against that set - e.g. the current
    #    KB and one rebuilt with an ingestion change, side by side.
    python -m evals.retrieval_eval run data/evals/<course>.json <kb_dir> [<kb_dir> ...]

The questions come from pages, not from chunks, so the same set scores KBs
chunked in different ways. Metrics, over the app's own retriever
(rag/knowledge_base.open_retriever, top 5):

- supported@1 / supported@5: a judge model says the first / any of the
  retrieved chunks holds what answers the question. Independent of chunk
  boundaries - the main metric.
- useful/5: how many of the 5 retrieved chunks the judge counts as
  supporting - retrieval slots not wasted on title-only or off-topic
  chunks. Broad questions are where this matters.
- page-hit@5 / MRR: a retrieved chunk spans the question's source page
  (for a broad question: any chunk from its file).
  Cheap and deterministic, but favours chunks spanning more pages, and
  misses when the page's content sits on another page (a repeated slide).
- chunk stats: retrieved chunk size and pages, and the share of near-empty
  chunks (body under 30 tokens) among them.

Costs a few cents per run (gpt-4.1-mini; override with EVAL_MODEL).
"""

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import fitz
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from rag.knowledge_base import open_retriever
from rag.splitter import token_length
from utils.helpers import detect_pdf_type

MIN_PAGE_WORDS = 25
TINY_CHUNK_TOKENS = 30


def _model() -> ChatOpenAI:
    return ChatOpenAI(model=os.getenv("EVAL_MODEL", "gpt-4.1-mini"), temperature=0)


# ==========================================================
# Generate
# ==========================================================

class StudentQuestion(BaseModel):
    question: str | None = Field(
        description="The student's question, or null if the page has no substantive content."
    )


QUESTION_PROMPT = """
Below is one page of a course's material (file "{file}", page {page}).

Write ONE question a student of this course might ask the tutor, which
this page answers. Rules:
- In the same language as the page.
- In your own words, the way a student would phrase it - do not copy
  distinctive phrases or titles from the page.
- About the content itself; never mention "the slide", "the page" or
  "the material".
- If the page has no substantive content (a cover, a section title, an
  agenda, only numbers or names), return null.

Page:
{text}
"""


BROAD_PROMPT = """
Below are the first pages of one file of a course's material (file
"{file}").

Write ONE broad question a student might ask the tutor about this file's
MAIN topic - the kind asked before studying it, e.g. "como funciona X?",
"o que preciso saber sobre X?", "me explica X". Rules:
- In the same language as the material.
- Name the topic in your own words; no details only an expert would know.
- If the file is administrative (schedule, course plan, grading) or only
  an exercise list, return null.

Material:
{text}
"""


def _broad_text(pdf: fitz.Document) -> str:
    """The file's first substantive pages, enough to tell its topic."""
    texts = [p.get_text("text") for p in pdf]
    return "\n".join(t for t in texts if len(t.split()) >= MIN_PAGE_WORDS)[:4000]


def generate(pdf_folder: Path, out: Path, total: int, seed: int) -> None:
    rng = random.Random(seed)
    candidates: dict[str, list[tuple[int, str, str]]] = {}
    broad_inputs: list[tuple[str, str, str]] = []

    for path in sorted(pdf_folder.glob("**/*.pdf")):
        pdf = fitz.open(path)
        doc_type = detect_pdf_type(pdf).value
        broad_inputs.append((path.name, _broad_text(pdf), doc_type))
        pages = [
            (number, page.get_text("text"), doc_type)
            for number, page in enumerate(pdf, start=1)
            if len(page.get_text("text").split()) >= MIN_PAGE_WORDS
        ]
        if pages:
            rng.shuffle(pages)
            candidates[path.name] = pages

    # Round-robin over files, so every file gets questions before any gets
    # many - a course is many files, not its longest deck. Oversample: the
    # model skips content-free pages.
    picks = []
    while len(picks) < total * 2 and any(candidates.values()):
        for name in sorted(candidates):
            if candidates[name]:
                number, text, doc_type = candidates[name].pop()
                picks.append((name, number, text, doc_type))

    writer = _model().with_structured_output(StudentQuestion)
    results = writer.batch(
        [QUESTION_PROMPT.format(file=n, page=p, text=t[:4000]) for n, p, t, _ in picks],
        config={"max_concurrency": 8},
    )

    questions = [
        {"kind": "specific", "question": r.question.strip(), "file": n, "page": p, "doc_type": d}
        for (n, p, _, d), r in zip(picks, results)
        if r is not None and r.question and r.question.strip()
    ][:total]

    broad = writer.batch(
        [BROAD_PROMPT.format(file=n, text=t) for n, t, _ in broad_inputs if t],
        config={"max_concurrency": 8},
    )
    questions += [
        {"kind": "broad", "question": r.question.strip(), "file": n, "page": None, "doc_type": d}
        for (n, t, d), r in zip([b for b in broad_inputs if b[1]], broad)
        if r is not None and r.question and r.question.strip()
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    kinds = {k: sum(q["kind"] == k for q in questions) for k in ("specific", "broad")}
    print(f"{kinds} questions from {len({q['file'] for q in questions})} files -> {out}")


# ==========================================================
# Run
# ==========================================================

class Support(BaseModel):
    supporting: list[int] = Field(
        description="Numbers of the excerpts that contain what answers the question (empty if none)."
    )


JUDGE_PROMPT = """
A student asked: "{question}"

Which of the numbered excerpts below contain information that answers
that question, or a substantial part of it? An excerpt that only
mentions the topic, or is just a title, does not count.

{excerpts}
"""


def _body(text: str) -> str:
    return text.split("\n\n", 1)[-1]


def _page_hit(doc, q) -> bool:
    meta = doc.metadata
    if meta.get("file_path") != q["file"]:
        return False
    if q.get("page") is None:  # broad: the file is enough
        return True
    return meta.get("page_start", 0) <= q["page"] <= meta.get("page_end", 0)


def run(questions_path: Path, kb_dirs: list[Path], save: Path | None = None) -> None:
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    judge = _model().with_structured_output(Support)

    print(f"{len(questions)} questions ({questions_path})\n")
    saved = {}

    for kb_dir in kb_dirs:
        retriever = open_retriever(kb_dir)
        retrieved = retriever.batch([q["question"] for q in questions], config={"max_concurrency": 8})

        verdicts = judge.batch(
            [
                JUDGE_PROMPT.format(
                    question=q["question"],
                    excerpts="\n\n".join(
                        f"[{i}]\n{_body(d.page_content)}" for i, d in enumerate(docs, 1)
                    ),
                )
                for q, docs in zip(questions, retrieved)
            ],
            config={"max_concurrency": 8},
        )

        rows = []
        for q, docs, v in zip(questions, retrieved, verdicts):
            supporting = set(v.supporting) if v else set()
            first_hit = next((i for i, d in enumerate(docs, 1) if _page_hit(d, q)), None)
            rows.append(
                {
                    "question": q["question"],
                    "retrieved": [
                        f"{d.metadata.get('file_path')} p.{d.metadata.get('page_start')}-{d.metadata.get('page_end')}"
                        for d in docs
                    ],
                    "supporting": sorted(supporting),
                    "group": q.get("kind", "specific"),
                    "useful": len(supporting & set(range(1, len(docs) + 1))),
                    "supported@1": 1 in supporting,
                    "supported@5": bool(supporting),
                    "page-hit@5": first_hit is not None,
                    "rr": 1 / first_hit if first_hit else 0.0,
                }
            )

        saved[str(kb_dir)] = rows

        chunk_tokens = [token_length(_body(d.page_content)) for docs in retrieved for d in docs]
        chunk_pages = [
            d.metadata.get("page_end", 0) - d.metadata.get("page_start", 0) + 1
            for docs in retrieved
            for d in docs
        ]

        with open(kb_dir / "metadata.json", encoding="utf-8") as f:
            meta = json.load(f)

        print(f"== {kb_dir}  ({meta.get('chunks')} chunks, parser {meta.get('parser_version')})")

        groups = [("all", rows)] + [
            (g, [r for r in rows if r["group"] == g]) for g in ("specific", "broad")
        ]
        print(f"   {'':8} {'n':>4} {'supp@1':>7} {'supp@5':>7} {'useful/5':>9} {'page@5':>7} {'MRR':>6}")
        for name, group in groups:
            if not group:
                continue
            n = len(group)
            print(
                f"   {name:8} {n:4d}"
                f" {sum(r['supported@1'] for r in group) / n:7.0%}"
                f" {sum(r['supported@5'] for r in group) / n:7.0%}"
                f" {sum(r['useful'] for r in group) / n:9.1f}"
                f" {sum(r['page-hit@5'] for r in group) / n:7.0%}"
                f" {sum(r['rr'] for r in group) / n:6.2f}"
            )
        print(
            f"   retrieved chunks: median {statistics.median(chunk_tokens):.0f} tokens,"
            f" {sum(t < TINY_CHUNK_TOKENS for t in chunk_tokens) / len(chunk_tokens):.0%} near-empty,"
            f" {statistics.mean(chunk_pages):.1f} pages on average\n"
        )

    if save:
        save.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"per-question results -> {save}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write a question set from a folder of course PDFs")
    gen.add_argument("pdf_folder", type=Path)
    gen.add_argument("out", type=Path)
    gen.add_argument("--total", type=int, default=60)
    gen.add_argument("--seed", type=int, default=0)

    run_cmd = sub.add_parser("run", help="score KB folders against a question set")
    run_cmd.add_argument("questions", type=Path)
    run_cmd.add_argument("kb_dirs", type=Path, nargs="+")
    run_cmd.add_argument("--save", type=Path, help="write per-question results (JSON) here")

    args = parser.parse_args()

    if args.command == "generate":
        if not args.pdf_folder.is_dir():
            sys.exit(f"Error: '{args.pdf_folder}' is not a folder.")
        generate(args.pdf_folder, args.out, args.total, args.seed)
    else:
        for kb_dir in args.kb_dirs:
            if not (kb_dir / "metadata.json").is_file():
                sys.exit(f"Error: '{kb_dir}' is not a knowledge base folder.")
        run(args.questions, args.kb_dirs, args.save)


if __name__ == "__main__":
    main()
