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
from rag.citation import format_citation
from rag.models import ChunkMetadata
from utils.citations import find_ref_labels, ref_label, replace_with_label

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


def test_code_block_indentation_is_kept():
    text = (
        "Pseudocódigo [[CITE:DOC_1]]:\n\n"
        "```\n"
        "função insere(nó, valor):\n"
        "    se nó é nulo:\n"
        "        retorna novo nó\n"
        "```\n"
    )

    result = substitute_citation_markers(text, CITATIONS)

    assert "    se nó é nulo:\n        retorna novo nó" in result


def test_inline_code_spacing_is_kept():
    text = "Use `a  =  b ;` aqui."

    assert substitute_citation_markers(text, CITATIONS) == text


def test_nested_list_indentation_is_kept():
    text = "- item\n    - subitem"

    assert substitute_citation_markers(text, CITATIONS) == text


LABEL = "📄 EDB2 09 AVL.pdf, p. 27"


@pytest.mark.parametrize("before", ["fez!", "fez.", "fez"])
def test_label_is_never_glued_to_the_previous_character(before):
    result = replace_with_label(f"{before}[EDB2 09 AVL.pdf, Slide 27]", "[EDB2 09 AVL.pdf, Slide 27]", LABEL)

    assert result == f"{before} {LABEL}"


def test_label_after_a_space_is_unchanged():
    result = replace_with_label("fez! [EDB2 09 AVL.pdf, Slide 27]", "[EDB2 09 AVL.pdf, Slide 27]", LABEL)

    assert result == f"fez! {LABEL}"


def test_citation_survives_marker_substitution_with_spaced_title():
    metadata = ChunkMetadata(
        source="EDB2 09 AVL.pdf",
        file_path="EDB2 09 AVL.pdf",
        doc_type="slides",
        page_start=77,
        page_end=77,
        section_id="77",
        section_title="Exercícios de  Árvore AVL",
        chunk_index=0,
        total_chunks=1,
        start_offset=0,
        end_offset=1,
    )
    citation = format_citation(metadata)

    result = substitute_citation_markers("Pratique [[CITE:DOC_1]].", {"DOC_1": citation})

    assert citation in result


class TestRefLabels:
    def test_labels_round_trip(self):
        text = (
            f"Uma AVL {ref_label('EDB2 09 AVL.pdf', 27, 27)}. Rotações "
            f"{ref_label('EDB2 09 AVL.pdf', 30, 32)} e de novo "
            f"{ref_label('EDB2 09 AVL.pdf', 27, 27)}; ver {ref_label('Plano de Curso 2022.1 - DIM0117.pdf')}"
        )

        assert find_ref_labels(text) == [
            ("📄 EDB2 09 AVL.pdf, p. 27", "EDB2 09 AVL.pdf", 27),
            ("📄 EDB2 09 AVL.pdf, pp. 30–32", "EDB2 09 AVL.pdf", 30),
            ("📄 Plano de Curso 2022.1 - DIM0117.pdf", "Plano de Curso 2022.1 - DIM0117.pdf", None),
        ]

    def test_no_labels(self):
        assert find_ref_labels("sem citações [algo.pdf]") == []
        assert find_ref_labels(None) == []
