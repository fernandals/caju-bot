import math
import uuid
import json
import fitz

from typing import Any
from pprint import pprint
from pathlib import Path

from rag.models import DocumentType, RawDocument

def detect_pdf_type(page: fitz.Page) -> DocumentType:
  width = page.mediabox_size[0]
  height = page.mediabox_size[1]

  return (
    DocumentType.SLIDES
    if width / height > 1.1
    else DocumentType.PDF
  )

def extract_slide_structure(text: str) -> tuple[str, str]:
  lines = [l.strip() for l in text.splitlines() if l.strip()]  # noqa: E741

  if not lines:
    return "", ""
 
  title = lines[0]
  body = "\n".join(lines[1:])

  return title, body

def softmax(scores):
    exp_scores = {k: math.exp(v) for k, v in scores.items()}
    total = sum(exp_scores.values())
    return {k: v / total for k, v in exp_scores.items()}

def generate_kb_id():
    return f"kb_{uuid.uuid4().hex[:8]}"

def save_chat(chat_id, chat, discipline=None):
    folder = Path("data/chats")
    folder.mkdir(parents=True, exist_ok=True)

    data = {
        "chat_id": chat_id,
        "discipline": discipline,
        "messages": chat
    }

    with open(folder / f"{chat_id}.json", "w") as f:
        json.dump(data, f, indent=2)

def load_chats():
    folder = Path("data/chats")

    if not folder.exists():
        return []

    chats = []
    for file in folder.glob("*.json"):
        with open(file) as f:
            chats.append(json.load(f))

    return chats

def print_tutor_state(
    state: Any,
    title: str = "TUTOR STATE",
    show_messages: bool = True,
    show_retrieved_docs: bool = True,
    max_doc_chars: int = 500,
):
    """
    Pretty debug visualization for the tutor state.

    Parameters
    ----------
    state : Any
        Current TutorState object or dict-like structure.

    title : str
        Section title printed at the top.

    show_messages : bool
        Whether to display conversation messages.

    show_retrieved_docs : bool
        Whether to display retrieved documents.

    max_doc_chars : int
        Maximum number of characters shown per retrieved document.
    """

    separator = "=" * 80

    print(f"\n{separator}")
    print(f"{title:^80}")
    print(separator)

    # --------------------------------------------------
    # Helper
    # --------------------------------------------------

    def section(name: str):
        print(f"\n{'-' * 30}")
        print(name.upper())
        print(f"{'-' * 30}")

    # --------------------------------------------------
    # Student Profile
    # --------------------------------------------------

    if "student_profile" in state:
        section("Student Profile")

        profile = state["student_profile"]

        if hasattr(profile, "model_dump"):
            pprint(profile.model_dump())
        else:
            pprint(profile)

    # --------------------------------------------------
    # Learning State
    # --------------------------------------------------

    if "learning_state" in state:
        section("Learning State")

        learning_state = state["learning_state"]

        if hasattr(learning_state, "model_dump"):
            pprint(learning_state.model_dump())
        else:
            pprint(learning_state)

    # --------------------------------------------------
    # Answer Plan
    # --------------------------------------------------

    if "answer_plan" in state:
        section("Answer Plan")

        answer_plan = state["answer_plan"]

        if hasattr(answer_plan, "model_dump"):
            pprint(answer_plan.model_dump())
        else:
            pprint(answer_plan)

    # --------------------------------------------------
    # Retrieved Documents
    # --------------------------------------------------

    if show_retrieved_docs and "retrieved_docs" in state:
        section("Retrieved Documents")

        docs = state.get("retrieved_docs", [])

        if not docs:
            print("No retrieved documents.")

        else:
            print(f"{len(docs)} document(s) retrieved.\n")

            for i, doc in enumerate(docs, 1):

                print("=" * 80)
                print(f"[DOCUMENT {i}]")
                print("=" * 80)

                # ------------------------------------------
                # LangChain Document support
                # ------------------------------------------

                if hasattr(doc, "page_content"):
                    content = doc.page_content
                    metadata = getattr(doc, "metadata", {})

                # ------------------------------------------
                # Pydantic / dict fallback
                # ------------------------------------------

                else:
                    if hasattr(doc, "model_dump"):
                        doc_data = doc.model_dump()
                    else:
                        doc_data = doc

                    content = doc_data.get("content", "")
                    metadata = {
                        k: v
                        for k, v in doc_data.items()
                        if k != "content"
                    }

                # ------------------------------------------
                # Content
                # ------------------------------------------

                truncated_content = (
                    content[:max_doc_chars] + "..."
                    if len(content) > max_doc_chars
                    else content
                )

                print("\nCONTENT:\n")
                print(truncated_content)

                # ------------------------------------------
                # Metadata
                # ------------------------------------------

                print("\nMETADATA:\n")

                if metadata:
                    pprint(metadata)
                else:
                    print("No metadata available.")

                print("\n")

    # --------------------------------------------------
    # Messages
    # --------------------------------------------------

    if show_messages and "messages" in state:
        section("Messages")

        messages = state.get("messages", [])

        if not messages:
            print("No messages.")
        else:
            for i, msg in enumerate(messages, 1):
                role = msg.__class__.__name__

                content = getattr(msg, "content", str(msg))

                print(f"\n[{i}] {role}")
                print(content)

    print(f"\n{separator}\n")

def offset_to_page(
    offset: int,
    doc: RawDocument,
) -> int:
    """
    Maps a character offset back to its page number.
    """

    pages = doc.pages

    for i, page in enumerate(pages):

        if i == len(pages) - 1:
            return page.number

        next_page = pages[i + 1]

        if page.start_offset <= offset < next_page.start_offset:
            return page.number

    return pages[-1].number