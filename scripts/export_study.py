"""
Export the personalization study (utils/study_analysis.py) to CSV, for
analysis outside the Streamlit panel (spreadsheet, R, a notebook).

Pull the databases first (scripts/pull_metrics.py, with chainlit.db for
the 👍/👎), then:

    python -m scripts.export_study [--metrics PATH] [--chainlit PATH] [--out DIR] [--discipline NAME]

Writes to DIR (default data/study/):
    sessions.csv  one row per conversation, with its outcomes
    arms.csv      mean outcome per arm (on / off), all conversations
    paired.csv    per student: their "on" mean minus their "off" mean
"""

import argparse
from pathlib import Path

from utils.study_analysis import build_sessions, compare_arms, paired_by_student


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--metrics", type=Path, default=Path("data/chats/metrics.db"))
    parser.add_argument("--chainlit", type=Path, default=Path("data/chats/chainlit.db"))
    parser.add_argument("--out", type=Path, default=Path("data/study"))
    parser.add_argument("--discipline", help="only this course's conversations (default: all)")
    args = parser.parse_args()

    if not args.metrics.is_file():
        raise SystemExit(f"metrics.db not found at {args.metrics}")

    sessions = build_sessions(args.metrics, args.chainlit, args.discipline)

    args.out.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(args.out / "sessions.csv", index=False)
    compare_arms(sessions).to_csv(args.out / "arms.csv")
    paired_by_student(sessions).to_csv(args.out / "paired.csv")

    print(f"{len(sessions)} conversations, {sessions['matricula'].nunique()} students -> {args.out}/")


if __name__ == "__main__":
    main()
