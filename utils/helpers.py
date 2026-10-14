import math
import re
import statistics
import unicodedata
import uuid
from pathlib import Path
from pprint import pprint
from typing import Any

import fitz

from rag.models import DocumentType, RawDocument

# Words a course's material uses for its numbered parts, in Portuguese and
# English - "Chapter" alone missed every Portuguese deck.
_CHAPTER_WORDS = (
    r"(?:chapter|cap[ií]tulo|cap\.|aula|unidade|lecture|module|m[óo]dulo|unit)"
)

# A running header like "Chapter 2. Viewpoints" / "Aula 3 - Árvores", at
# the start of the line or after a separator ("Book – Part I – Chapter 2.
# Title") - never mid-sentence ("como vimos na aula 3 - árvores").
CHAPTER_HEADER_REGEX = re.compile(
    rf"(?:^\s*|[–—|-]\s*){_CHAPTER_WORDS}\s+(\d+)\s*[.:–-]\s*(.+)",
    re.IGNORECASE,
)

# A file name like "SAIA-Chapter12.pdf" / "Aula_03.pdf".
CHAPTER_FILENAME_REGEX = re.compile(
    rf"{_CHAPTER_WORDS}[\s_-]?(\d+)",
    re.IGNORECASE,
)

def _page_is_landscape(page: fitz.Page) -> bool:
  width, height = page.mediabox_size

  return height > 0 and width / height > 1.1

# Slides vs documents, measured on two courses' material (EDB2 slides and
# lists, Software Architecture book chapters and a deck): body text of
# slides is 2.4-5.6% of the page width, of documents 1.6-2.0%; slides
# carry 18-79 words per page, documents 110-477.
SLIDE_MIN_FONT_RATIO = 0.022
SLIDE_MAX_WORDS_PER_PAGE = 90


def _spans(page: fitz.Page) -> list[tuple[float, str]]:
  """(font size, text) of every non-blank text line on the page."""
  lines = []

  for block in page.get_text("dict")["blocks"]:
    for line in block.get("lines", []):
      text = "".join(span["text"] for span in line["spans"]).strip()
      if text:
        lines.append((max(span["size"] for span in line["spans"]), text))

  return lines


def _body_font_size(lines: list[tuple[float, str]]) -> float:
  """Font size most of the page's characters are set in."""
  sizes = [size for size, text in lines for _ in text]
  return statistics.median(sizes) if sizes else 0.0


def detect_pdf_type(pdf: fitz.Document, sample_pages: int = 5) -> DocumentType:
  """
  Classifies a PDF as SLIDES or PDF over its first `sample_pages` pages,
  by a vote of three signals - any two make it slides:

  - landscape pages;
  - large body text relative to the page width;
  - few words per page.

  The page shape alone misfiled decks exported in portrait (a
  PowerPoint "handout" with one slide per page), which then went through
  the book parser as a single section.
  """
  n = min(sample_pages, pdf.page_count)

  if n == 0:
    return DocumentType.UNKNOWN

  pages = [pdf[i] for i in range(n)]
  landscape = sum(_page_is_landscape(p) for p in pages) * 2 > n

  lines = [line for p in pages for line in _spans(p)]
  large_font = (
    _body_font_size(lines) / statistics.median(p.rect.width for p in pages)
    >= SLIDE_MIN_FONT_RATIO
  )

  sparse = (
    statistics.median(len(p.get_text("text").split()) for p in pages)
    < SLIDE_MAX_WORDS_PER_PAGE
  )

  return DocumentType.SLIDES if landscape + large_font + sparse >= 2 else DocumentType.PDF


# A heading is set at least this much larger than the page's body text.
HEADING_MIN_SCALE = 1.2

# Longer than this, the "largest text" is not a title (a page set in one
# big font, a quote).
HEADING_MAX_CHARS = 150


def page_heading_lines(page: fitz.Page) -> list[str]:
  """
  The page's title lines: those set in its largest font, when that font
  clearly stands out from the body text, in reading order. A title that
  wraps ("Árvores" / "AVL") comes back whole, which the first line of the
  page text did not give. Empty when nothing stands out.
  """
  # Lines without words ("+", a bullet, a big arrow) are decoration - a
  # deck set its "+" section markers larger than the titles.
  lines = [(size, text) for size, text in _spans(page) if _letters(text) >= 1]

  if not lines:
    return []

  top = max(size for size, _ in lines)
  heading = [text for size, text in lines if size >= top * 0.9]
  rest = [(size, text) for size, text in lines if size < top * 0.9]

  # A page with nothing else on it (a cover: "Árvores" / "AVL") is all
  # title; otherwise the title must stand out from the rest.
  if rest and top < _body_font_size(rest) * HEADING_MIN_SCALE:
    return []

  if len(" ".join(heading)) > HEADING_MAX_CHARS or _letters("".join(heading)) < 3:
    return []

  return heading


def _letters(text: str) -> int:
  return sum(c.isalpha() for c in text)


def clean_title(text: str | None) -> str:
  """
  A title as a single-spaced line: extracted text keeps layout runs of
  spaces, symbol-font glyphs (private-use characters, e.g. a bullet
  "\uf0e5") and leading markers ("+Viewpoints", "• Título").
  """
  text = "".join(c for c in (text or "") if unicodedata.category(c) != "Co")
  text = " ".join(text.split())

  while text and not (text[0].isalnum() or text[0] in "(\"'¿¡"):
    text = text[1:].lstrip()

  return text

def extract_slide_structure(text: str, heading_lines: list[str] | None = None) -> tuple[str, str]:
  lines = [l.strip() for l in text.splitlines() if l.strip()]

  if not lines:
    return "", ""

  # The title from the slide's largest font (page_heading_lines), when it
  # has one; the body is the rest of the slide.
  if heading_lines:
    body = list(lines)
    for heading in heading_lines:
      if heading.strip() in body:
        body.remove(heading.strip())
    return clean_title(" ".join(heading_lines)), "\n".join(body)

  # Skip boilerplate leading lines: a repeated running header (e.g. "Book –
  # Part I – Chapter 2. Title") and bare transition markers ("+") used
  # between sections in some slide decks, so the slide's own heading is
  # used as the title instead.
  while len(lines) > 1 and (
      CHAPTER_HEADER_REGEX.search(lines[0]) or lines[0] == "+"
  ):
    lines = lines[1:]

  title = clean_title(lines[0])
  body = "\n".join(lines[1:])

  return title, body

def softmax(scores):
    exp_scores = {k: math.exp(v) for k, v in scores.items()}
    total = sum(exp_scores.values())
    return {k: v / total for k, v in exp_scores.items()}

def generate_kb_id():
    return f"kb_{uuid.uuid4().hex[:8]}"

def detect_chapter_header(text: str) -> tuple[str | None, str | None]:
    """
    Looks for a "Chapter N. Title" running header on the FIRST LINE of the
    given text only (e.g. a page or a section's opening page).

    Deliberately does not scan the full body: chapters are often referenced
    in passing elsewhere in the text (e.g. "see Chapter 9 for tactics..."),
    which would misattribute chunks to the wrong chapter if matched anywhere.
    """
    if not text:
        return None, None

    first_line = text.strip().splitlines()[0] if text.strip() else ""

    match = CHAPTER_HEADER_REGEX.search(first_line)

    if not match:
        return None, None

    return match.group(1), match.group(2).strip()

def detect_chapter_from_filename(file_path: str) -> str | None:
    """
    Fallback chapter number detection based on filename convention,
    e.g. 'SAIA-Chapter12.pdf' -> '12'.
    """
    match = CHAPTER_FILENAME_REGEX.search(file_path)

    if not match:
        return None

    return match.group(1)

def load_roster(path: Path) -> set[str]:
    """
    Loads a plain-text roster of valid enrollment IDs: one per line,
    blank lines and '#' comments ignored. Not authentication - just an
    allowlist gate, prepared by the teacher ahead of time.
    """
    if not path.exists():
        return set()

    ids = set()

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            ids.add(line)

    return ids

def is_enrolled(student_id: str, roster: set[str]) -> bool:
    return student_id.strip() in roster


def _strip_accents(text: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def parse_sigaa_roster(source: Any) -> list[str]:
    """
    Extracts the enrollment IDs (matrículas) from a SIGAA "Planilha de
    notas" .xls export - the file a teacher downloads from SIGAA to enter
    grades. The sheet has some header/instruction rows, then a row whose
    first data column is labelled "Matrícula", then one student per line.

    We locate that "Matrícula" header cell and read its column downward,
    keeping every value that is all digits. Returns the IDs in sheet
    order, de-duplicated.

    `source` is a path, a file-like object, or the raw bytes of the .xls
    (e.g. a Streamlit upload's .getvalue()).
    """
    import io

    import pandas as pd

    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)

    df = pd.read_excel(source, header=None, dtype=str, engine="xlrd")

    header_col = None
    header_row = None
    for row_idx, row in df.iterrows():
        for col_idx, value in enumerate(row):
            if (
                isinstance(value, str)
                and _strip_accents(value).strip().lower() == "matricula"
            ):
                header_row, header_col = row_idx, col_idx
                break
        if header_col is not None:
            break

    if header_col is None:
        raise ValueError(
            "Não encontrei a coluna 'Matrícula' na planilha. "
            "Confira se este é o arquivo .xls exportado do SIGAA."
        )

    ids: list[str] = []
    seen: set[str] = set()
    for value in df.iloc[header_row + 1 :, header_col]:
        if not isinstance(value, str):
            continue
        student_id = value.strip()
        if student_id.isdigit() and student_id not in seen:
            seen.add(student_id)
            ids.append(student_id)

    return ids


def write_roster(
    ids: list[str],
    path: Path,
    header: str | None = None,
) -> None:
    """
    Writes a plain-text roster in the format load_roster expects: one ID
    per line, with an optional leading '# ' comment line.
    """
    lines = [f"# {header}"] if header else []
    lines.extend(ids)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

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