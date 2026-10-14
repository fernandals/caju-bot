"""
The course's topic list, for "what can I study?" requests (the TOPICS
starter button, or the same question typed - agent/graph.is_topics_request).

Without it the tutor has no idea what the course covers: the request went
through the guided-teaching arc, which answered with a question back, and
retrieval on such a vague query brings back arbitrary chunks.

Built once per knowledge base with one LLM call over the material file
names and section titles (rag/knowledge_base.describe_course_materials /
list_material_sections - chapter titles are often missing, so neither is
usable raw), then cached as outline.json in the KB folder. A rebuilt KB
gets a new folder, so the cache can't go stale.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from rag.knowledge_base import (
    describe_course_materials,
    knowledge_base_dir,
    list_material_sections,
)

logger = logging.getLogger(__name__)

OUTLINE_FILE = "outline.json"

# Section titles per file shown to the model - enough to tell what a
# generically named file ("Chapter12.pdf") covers, without the noise of
# every slide heading.
SECTIONS_PER_FILE = 12


class CourseOutline(BaseModel):
    topics: list[str] = Field(
        description=(
            "The course's main study topics, in teaching order, each a "
            "short name (2-6 words)."
        )
    )


OUTLINE_PROMPT = """
You are given the materials of the course "{subject}": each source file
name followed by some of its section headings (extracted automatically -
some are noise, such as sentence fragments or names).

List the course's main study topics, the way a syllabus would: in
teaching order, 5 to 15 items, each a short name (2-6 words) in
{language}. Merge files on the same subject into one topic. Leave out
administrative material (schedule, course plan, grading, contact) and
exercise lists or assignments - they are not topics themselves.

Materials:
{materials}
"""


def _materials_text(subject: str) -> str:
    sections = defaultdict(list)

    for section in list_material_sections(subject):
        title = (section["section_title"] or "").strip()
        if title and title not in sections[section["file"]]:
            sections[section["file"]].append(title)

    lines = []
    for material in describe_course_materials(subject):
        name = Path(material["file"]).stem
        if material["title"]:
            name += f" ({material['title']})"
        lines.append(f"- {name}")
        lines.extend(f"    - {t}" for t in sections[material["file"]][:SECTIONS_PER_FILE])

    return "\n".join(lines)


def load_course_outline(subject: str, language: str, model) -> list[str]:
    """
    The cached topic list, built on first use. Empty if the KB has no
    materials or the model call fails - topic requests then go through
    the normal pipeline (agent/graph.route_start).
    """
    kb_dir = knowledge_base_dir(subject)
    if kb_dir is None:
        return []

    cache = kb_dir / OUTLINE_FILE
    if cache.is_file():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))["topics"]
        except (OSError, ValueError, KeyError):
            logger.warning("Unreadable %s; rebuilding it", cache)

    materials = _materials_text(subject)
    if not materials:
        return []

    try:
        outline = model.with_structured_output(CourseOutline).invoke(
            OUTLINE_PROMPT.format(subject=subject, language=language, materials=materials)
        )
    except Exception:  # noqa: BLE001 - the tutor works without it
        logger.warning("Could not build the course outline", exc_info=True)
        return []

    topics = [t.strip() for t in (outline.topics if outline else []) if t.strip()]

    if topics:
        try:
            cache.write_text(
                json.dumps({"topics": topics}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            logger.warning("Could not cache the course outline", exc_info=True)

    return topics
