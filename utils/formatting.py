"""
Math delimiters for the chat's LaTeX rendering (.chainlit/config.toml,
latex = true).

Chainlit renders any $...$ as math, so a "$" in ordinary text - "R$ 10",
"US$ 5" - would start a formula and garble the message. The tutor is told
to write math as \\( \\) / \\[ \\] instead (agent/prompts.GENERATE_PROMPT);
here every other "$" is escaped to a literal one, then those delimiters
are turned into the $ / $$ Chainlit understands. Code is left untouched.
"""

import re

from agent.nodes import CODE_SPAN_REGEX

_UNESCAPED_DOLLAR = re.compile(r"(?<!\\)\$")
_DISPLAY_MATH = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_INLINE_MATH = re.compile(r"\\\((.+?)\\\)", re.DOTALL)


def _prose(text: str) -> str:
    text = _UNESCAPED_DOLLAR.sub(r"\\$", text)
    text = _DISPLAY_MATH.sub(lambda m: f"$${m.group(1)}$$", text)
    return _INLINE_MATH.sub(lambda m: f"${m.group(1).strip()}$", text)


def render_math(text: str) -> str:
    """Tutor answer -> Chainlit Markdown: \\( \\) / \\[ \\] math, literal "$"."""
    parts = CODE_SPAN_REGEX.split(text)

    # split() with one capture group alternates prose, code, prose, ...
    return "".join(part if i % 2 else _prose(part) for i, part in enumerate(parts))
