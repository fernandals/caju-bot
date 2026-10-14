"""
The answer text on its way to the student: citation markers swapped for
real citations (agent/nodes.substitute_citation_markers), citations
turned into PDF chips and read back on resume (utils/citations.py), and
math / dollar signs for Chainlit's LaTeX (utils/formatting.py).
"""

import pytest

from agent.nodes import substitute_citation_markers
from rag.citation import format_citation
from rag.models import ChunkMetadata
from utils.citations import find_ref_labels, ref_label, replace_with_label
from utils.formatting import render_math

CITATIONS = {"DOC_1": "[Chapter 2, Pages 4-5]", "DOC_2": "[Chapter 3, Page 8]"}


# ---------------- citation markers ----------------


@pytest.mark.parametrize("marker", ["[[CITE:DOC_1]]", "[DOC_1]", "[CITE: DOC_1]", "(DOC_1)", "DOC-1", "DOC_1", "doc_1"])
def test_every_marker_variant_the_model_writes_is_replaced(marker):
    result = substitute_citation_markers(f"Loops repetem um bloco {marker} de código.", CITATIONS)
    assert marker not in result and "[Chapter 2, Pages 4-5]" in result


def test_unknown_markers_never_leak():
    # A hallucinated id, or one whose chunk the grader dropped.
    result = substitute_citation_markers("Veja [[CITE:DOC_9]] e [DOC_3] .", CITATIONS)
    assert "DOC" not in result
    assert " ." not in result and "  " not in result

    assert "DOC" not in substitute_citation_markers("Veja [[CITE:DOC_1]] e [DOC_2].", {})


def test_text_and_code_spacing_is_kept():
    for text in ("Essa resposta não cita nenhuma fonte.", "Use `a  =  b ;` aqui.", "- item\n    - subitem"):
        assert substitute_citation_markers(text, CITATIONS) == text

    code = "Pseudocódigo [[CITE:DOC_1]]:\n\n```\nfunção insere(nó):\n    se nó é nulo:\n        retorna novo nó\n```\n"
    assert "    se nó é nulo:\n        retorna novo nó" in substitute_citation_markers(code, CITATIONS)


def test_citation_with_a_spaced_title_still_matches():
    # Runs of spaces in an extracted title once broke the chip link.
    citation = format_citation(
        ChunkMetadata(
            source="a.pdf", file_path="EDB2 09 AVL.pdf", doc_type="slides", page_start=77, page_end=77,
            section_id="77", section_title="Exercícios de  Árvore AVL",
            chunk_index=0, total_chunks=1, start_offset=0, end_offset=1,
        )
    )
    assert citation in substitute_citation_markers("Pratique [[CITE:DOC_1]].", {"DOC_1": citation})


# ---------------- chips ----------------

LABEL = "📄 EDB2 09 AVL.pdf, p. 27"
CITATION = "[EDB2 09 AVL.pdf, Slide 27]"


@pytest.mark.parametrize("before", ["fez!", "fez.", "fez", "fez! "])
def test_chip_is_never_glued_to_the_previous_character(before):
    # Right after "!", Chainlit's link became Markdown image syntax.
    assert replace_with_label(f"{before}{CITATION}", CITATION, LABEL) == f"{before.strip()} {LABEL}"


def test_chips_read_back_from_a_saved_message():
    text = (
        f"Uma AVL {ref_label('EDB2 09 AVL.pdf', 27, 27)}. Rotações {ref_label('EDB2 09 AVL.pdf', 30, 32)} "
        f"e de novo {ref_label('EDB2 09 AVL.pdf', 27, 27)}; ver {ref_label('Plano de Curso 2022.1 - DIM0117.pdf')}"
    )
    assert find_ref_labels(text) == [
        ("📄 EDB2 09 AVL.pdf, p. 27", "EDB2 09 AVL.pdf", 27),
        ("📄 EDB2 09 AVL.pdf, pp. 30–32", "EDB2 09 AVL.pdf", 30),
        ("📄 Plano de Curso 2022.1 - DIM0117.pdf", "Plano de Curso 2022.1 - DIM0117.pdf", None),
    ]
    assert find_ref_labels("sem citações [algo.pdf]") == [] and find_ref_labels(None) == []


# ---------------- math ----------------


@pytest.mark.parametrize(
    "text, rendered",
    [
        (r"custa \(O(n \log n)\) no pior caso", r"custa $O(n \log n)$ no pior caso"),
        (r"\[T(n) = 2T(n/2) + n\]", "$$T(n) = 2T(n/2) + n$$"),
        ("custa R$ 10 ou US$ 2", r"custa R\$ 10 ou US\$ 2"),
        (r"R$ 5 por \(n\) itens", r"R\$ 5 por $n$ itens"),
        ("$n$", r"\$n\$"),  # a stray $ is shown, never a formula
        (r"R\$ 10", r"R\$ 10"),
        ("use `echo $HOME`:\n\n```sh\nprice=$1\n```\n", "use `echo $HOME`:\n\n```sh\nprice=$1\n```\n"),
    ],
)
def test_math_delimiters_and_literal_dollars(text, rendered):
    assert render_math(text) == rendered
