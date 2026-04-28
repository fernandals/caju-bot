import fitz
from rag.models import DocumentType
import math
import uuid
import json
from pathlib import Path

def detect_pdf_type(page: fitz.Page) -> DocumentType:
  width = page.mediabox_size[0]
  height = page.mediabox_size[1]

  return (
    DocumentType.SLIDES
    if width / height > 1.1
    else DocumentType.PDF
  )

def extract_slide_structure(text: str) -> tuple[str, str]:
  lines = [l.strip() for l in text.splitlines() if l.strip()]

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
    folder = Path(f"data/chats")

    if not folder.exists():
        return []

    chats = []
    for file in folder.glob("*.json"):
        with open(file) as f:
            chats.append(json.load(f))

    return chats