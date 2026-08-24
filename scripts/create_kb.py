"""
Standalone CLI to build a course knowledge base from a folder of PDFs,
without going through the Streamlit dev tool (app.py).

Intended for the teacher-facing pipeline described in DEPLOY.md: point
this at a folder of course PDFs, get a data/knowledge_bases/<kb_id>/
folder back, then build the Docker image.

Usage (run from the repo root, so `rag`/`utils` resolve as packages):
    python -m scripts.create_kb <pdf_folder> "<discipline name>"
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag.knowledge_base import create_and_save_knowledge_base, list_knowledge_bases


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

    return parser.parse_args()


def main() -> None:
    load_dotenv()

    args = parse_args()

    if not args.pdf_folder.is_dir():
        sys.exit(f"Error: '{args.pdf_folder}' is not a folder.")

    if not any(args.pdf_folder.glob("**/*.pdf")):
        sys.exit(f"Error: no PDF files found under '{args.pdf_folder}'.")

    create_and_save_knowledge_base(args.pdf_folder, args.discipline)

    print("\nKnowledge bases now on disk:")
    for kb in list_knowledge_bases():
        print(f"  {kb['name']} (ID: {kb['id']})")


if __name__ == "__main__":
    main()
