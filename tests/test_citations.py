"""
Regression tests for agent/nodes.py::substitute_citation_markers.

PROJECT_LOG.md (2026-09-09) documents a real bug here: the generation
model degrades the intended `[[CITE:DOC_1]]` marker to variants like
`[DOC_1]`, `(DOC_1)`, or a bare `DOC_1`, and any of those leaking to the
student looks broken/unprofessional. The regex was made tolerant to
catch all known variants - this file is what keeps that fix from
regressing silently.
"""

import pytest

from agent.nodes import substitute_citation_markers

CITATIONS = {"DOC_1": "[Chapter 2, Pages 4-5]", "DOC_2": "[Chapter 3, Page 8]"}


@pytest.mark.parametrize(
    "marker",
    [
        "[[CITE:DOC_1]]",
        "[DOC_1]",
        "[CITE: DOC_1]",
        "[CITE:DOC_1]",
        "(DOC_1)",
        "DOC-1",
        "DOC_1",
        "doc_1",
    ],
)
def test_known_marker_variants_are_replaced(marker):
    text = f"Loops repetem um bloco {marker} de código."
    result = substitute_citation_markers(text, CITATIONS)

    assert marker not in result
    assert "[Chapter 2, Pages 4-5]" in result


def test_multiple_distinct_markers_resolve_independently():
    text = "Primeiro veja [[CITE:DOC_1]], depois [[CITE:DOC_2]]."
    result = substitute_citation_markers(text, CITATIONS)

    assert "[Chapter 2, Pages 4-5]" in result
    assert "[Chapter 3, Page 8]" in result


def test_hallucinated_doc_id_is_removed_not_leaked():
    text = "Isso vem daqui [[CITE:DOC_9]]."
    result = substitute_citation_markers(text, CITATIONS)

    assert "DOC_9" not in result
    assert "DOC" not in result


def test_filtered_out_doc_id_is_removed():
    # DOC_3 was never in citations_by_doc_id (e.g. its chunk was filtered
    # by assess_documents) - must not leak the raw marker.
    text = "Conforme [DOC_3] descreve."
    result = substitute_citation_markers(text, CITATIONS)

    assert "DOC_3" not in result
    assert "[" not in result or "Chapter" in result


def test_removed_marker_does_not_leave_dangling_punctuation_gap():
    text = "A resposta é essa [[CITE:DOC_9]] ."
    result = substitute_citation_markers(text, CITATIONS)

    assert " ." not in result
    assert "  " not in result


def test_text_without_any_marker_is_unchanged():
    text = "Essa resposta não cita nenhuma fonte."
    assert substitute_citation_markers(text, CITATIONS) == text


def test_empty_citations_map_strips_all_markers():
    text = "Veja [[CITE:DOC_1]] e [DOC_2]."
    result = substitute_citation_markers(text, {})

    assert "DOC_1" not in result
    assert "DOC_2" not in result
