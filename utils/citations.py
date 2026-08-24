import re

# Wraps our own citation markers - e.g. "[SAIA-Chapter13.pdf, Chapter 13,
# Section 13.1 ...]" - in backticks so they render as a distinct inline-code
# "badge" instead of blending into plain text. Matched narrowly (must
# contain ".pdf") so it can't accidentally wrap unrelated bracketed text.
CITATION_PATTERN = re.compile(r"(\[[^\[\]]*?\.pdf[^\[\]]*?\])")


def highlight_citations(text: str) -> str:
    return CITATION_PATTERN.sub(r"`\1`", text)
