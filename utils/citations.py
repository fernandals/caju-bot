import re

# Wraps our own citation markers - e.g. "[SAIA-Chapter13.pdf, Chapter 13,
# Section 13.1 ...]" - in backticks so they render as a distinct inline-code
# "badge" instead of blending into plain text. Matched narrowly (must
# contain ".pdf") so it can't accidentally wrap unrelated bracketed text.
CITATION_PATTERN = re.compile(r"(\[[^\[\]]*?\.pdf[^\[\]]*?\])")


def ref_label(file_name: str | None, page_start=None, page_end=None) -> str:
    """Compact chip label for a citation, e.g. '📄 SAIA-Chapter13.pdf, p. 2'."""
    file_name = file_name or "material"

    if page_start and page_end and page_end != page_start:
        return f"📄 {file_name}, pp. {page_start}–{page_end}"
    if page_start:
        return f"📄 {file_name}, p. {page_start}"
    return f"📄 {file_name}"


# ref_label's output, read back from a saved message: Chainlit doesn't
# persist the PDF elements behind the chips (no blob storage configured),
# so a resumed conversation rebuilds them from the labels in the text.
_REF_LABEL_REGEX = re.compile(
    r"📄 (?P<file>[^\n📄]+?\.pdf)(?:, (?:p\. (?P<page>\d+)|pp\. (?P<start>\d+)–\d+))?"
)


def find_ref_labels(text: str) -> list[tuple[str, str, int | None]]:
    """Distinct (label, file name, first page) of the chips in a message."""
    found = {}

    for match in _REF_LABEL_REGEX.finditer(text or ""):
        page = match.group("page") or match.group("start")
        found.setdefault(match.group(0), (match.group("file"), int(page) if page else None))

    return [(label, file, page) for label, (file, page) in found.items()]


def replace_with_label(text: str, citation: str, label: str) -> str:
    """
    Swap a full citation for its chip label, never glued to the previous
    character. Chainlit turns the label into a Markdown link, so right
    after a "!" ("fez![label](...)") it became an image - a broken one.
    """
    text = text.replace(citation, label)
    return re.sub(rf"(?<=[^\s(\[]){re.escape(label)}", f" {label}", text)


def highlight_citations(text: str) -> str:
    return CITATION_PATTERN.sub(r"`\1`", text)
