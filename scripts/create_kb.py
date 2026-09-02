"""
Standalone CLI to build a course knowledge base from a folder of PDFs,
without going through the Streamlit dev tool (app.py).

Intended for the teacher-facing pipeline described in DEPLOY.md: point
this at a folder of course PDFs, get a data/knowledge_bases/<kb_id>/
folder back, then build the Docker image.

Usage (run from the repo root, so `rag`/`utils` resolve as packages):
    python -m scripts.create_kb <pdf_folder> "<discipline name>"

Pass --roster <sigaa_grade_sheet.xls> to also generate data/roster.txt
from the sheet's "Matrícula" column in the same run.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag.knowledge_base import create_and_save_knowledge_base, list_knowledge_bases
from utils.helpers import parse_sigaa_roster, write_roster

ROSTER_PATH = Path("data/roster.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "pdf_folder",
        type=Path,
        help="Folder containing the course's PDF files (searched recursively).",
    )
    parser.add_argument(
        "discipline",
        help="Discipline/course name to tag this knowledge base with.",
    )
    parser.add_argument(
        "--roster",
        type=Path,
        help=(
            "SIGAA grade sheet (.xls) to build data/roster.txt from - the "
            "'Matrícula' column becomes the enrollment allowlist."
        ),
    )

    return parser.parse_args()


def main() -> None:
    load_dotenv()

    args = parse_args()

    if not args.pdf_folder.is_dir():
        sys.exit(f"Error: '{args.pdf_folder}' is not a folder.")

    if not any(args.pdf_folder.glob("**/*.pdf")):
        sys.exit(f"Error: no PDF files found under '{args.pdf_folder}'.")

    student_ids = None
    if args.roster is not None:
        if not args.roster.is_file():
            sys.exit(f"Error: '{args.roster}' is not a file.")
        try:
            student_ids = parse_sigaa_roster(args.roster)
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"Error reading roster '{args.roster}': {exc}")
        if not student_ids:
            sys.exit(f"Error: no enrollment IDs found in '{args.roster}'.")

    create_and_save_knowledge_base(args.pdf_folder, args.discipline)

    if student_ids is not None:
        write_roster(student_ids, ROSTER_PATH, header=args.discipline)
        print(f"Wrote {len(student_ids)} enrollment ID(s) to {ROSTER_PATH}")

    print("\nKnowledge bases now on disk:")
    for kb in list_knowledge_bases():
        print(f"  {kb['name']} (ID: {kb['id']})")


if __name__ == "__main__":
    main()
