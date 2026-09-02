"""
Pull the live metrics.db (and optionally chainlit.db) off the Railway
volume so you can open it in the local Streamlit panel.

The student app writes one anonymous row per conversation turn into
metrics.db on the mounted volume (/app/data/chats/). This script copies
that file down to data/chats/metrics.db - exactly where the "Métricas da
turma" tab in app.py looks by default.

Usage (run from the repo root, with the Railway project already linked -
see DEPLOY.md):
    python -m scripts.pull_metrics

Options:
    --volume NAME   Railway volume name (default: langchain-rag-volume)
    --dest PATH     Local file to write (default: data/chats/metrics.db)
    --chainlit      Also pull chainlit.db (chat history) next to it
    --open          Launch `streamlit run app.py` once the download is done

Requires the Railway CLI (`railway`) on PATH and a linked project. If you
have no SSH key registered with Railway, run `railway ssh keys add` once,
or use the dashboard's volume file browser instead.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_VOLUME = "langchain-rag-volume"
DEFAULT_DEST = Path("data/chats/metrics.db")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--volume",
        default=DEFAULT_VOLUME,
        help=f"Railway volume name (default: {DEFAULT_VOLUME}).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Local path to write metrics.db to (default: {DEFAULT_DEST}).",
    )
    parser.add_argument(
        "--chainlit",
        action="store_true",
        help="Also download chainlit.db (chat history) into the same folder.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Run `streamlit run app.py` after downloading.",
    )
    return parser.parse_args()


def _download(volume: str, remote_name: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "railway",
        "volume",
        "files",
        "--volume",
        volume,
        "download",
        f"/{remote_name}",
        str(dest),
        "--overwrite",
    ]
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(
            f"Error: `railway volume files download` failed for "
            f"/{remote_name}. Is the project linked (`railway status`) and "
            f"an SSH key registered (`railway ssh keys add`)?"
        )
    print(f"Wrote {dest} ({dest.stat().st_size} bytes)")


def main() -> None:
    args = parse_args()

    if shutil.which("railway") is None:
        sys.exit(
            "Error: the Railway CLI (`railway`) is not on PATH. Install it "
            "from https://docs.railway.com/guides/cli and `railway link` "
            "this project first."
        )

    _download(args.volume, "metrics.db", args.dest)

    if args.chainlit:
        _download(args.volume, "chainlit.db", args.dest.parent / "chainlit.db")

    if args.open:
        print("\n$ streamlit run app.py")
        subprocess.run(["streamlit", "run", "app.py"])
    else:
        print(
            "\nDone. Open the panel with:\n"
            "    streamlit run app.py\n"
            "and go to the '📊 Métricas da turma' tab "
            f"(it defaults to {DEFAULT_DEST})."
        )


if __name__ == "__main__":
    main()
