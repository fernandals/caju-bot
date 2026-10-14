"""
Building a knowledge base from any course's PDFs (rag/, utils/helpers.py):
slides vs documents, titles from font sizes, letterhead kept out of
titles, chapter markers, slides grouped into chunks, and the keyword
search. PDFs are built in memory with PyMuPDF.
"""

import fitz
import pytest
from langchain_core.documents import Document

from rag.models import DocumentType, Page, RawDocument, RawDocumentMetadata, Section, SemanticBlock
from rag.parser import SLIDE_GROUP_TOKENS, drop_slide_builds, extract_document_title, find_boilerplate_lines, group_slides
from rag.retrieval import BM25, reciprocal_rank_fusion, tokenize
from rag.splitter import token_length
from utils.helpers import clean_title, detect_chapter_from_filename, detect_chapter_header, detect_pdf_type, page_heading_lines

LANDSCAPE, PORTRAIT = (720, 540), (595, 842)
PROSE = (
    "Uma árvore binária de busca guarda chaves menores à esquerda e maiores à direita de "
    "cada nó, o que permite buscar, inserir e remover em tempo proporcional à altura. "
)


def pdf(pages: int, size, lines: list[tuple[float, str]]) -> fitz.Document:
    """Each page gets `lines`, written top to bottom at their font sizes."""
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page(width=size[0], height=size[1])
        y = 40
        for fontsize, text in lines:
            page.insert_textbox(fitz.Rect(30, y, size[0] - 30, size[1] - 20), text, fontsize=fontsize)
            y += fontsize * 1.6 * (1 + len(text) * fontsize * 0.5 // (size[0] - 60))
    return doc


# ---------------- slides or document ----------------


@pytest.mark.parametrize(
    "doc, kind",
    [
        (pdf(5, LANDSCAPE, [(32, "Árvores AVL"), (20, "balanceada pela altura")]), DocumentType.SLIDES),
        # A deck exported in portrait (one slide per page).
        (pdf(5, PORTRAIT, [(40, "Complexidade Média"), (24, "busca em lista ordenada")]), DocumentType.SLIDES),
        (pdf(3, PORTRAIT, [(10, PROSE * 6)]), DocumentType.PDF),
    ],
)
def test_slides_or_document(doc, kind):
    assert detect_pdf_type(doc) == kind


# ---------------- titles ----------------


@pytest.mark.parametrize(
    "lines, heading",
    [
        ([(32, "Inserção em AVL"), (18, "rotação simples"), (18, "rotação dupla")], ["Inserção em AVL"]),
        # A cover: all title, wrapped over two lines.
        ([(48, "Árvores"), (48, "B")], ["Árvores", "B"]),
        # Decoration without letters set larger than the title.
        ([(54, "+"), (40, "Viewpoints"), (18, "a set of conventions")], ["Viewpoints"]),
        ([(10, PROSE * 3)], []),
    ],
)
def test_page_title_is_the_largest_font(lines, heading):
    size = PORTRAIT if lines[0][0] == 10 else LANDSCAPE
    assert page_heading_lines(pdf(1, size, lines)[0]) == heading


@pytest.mark.parametrize(
    "raw, clean",
    [
        ("Exercícios de  Árvore AVL", "Exercícios de Árvore AVL"),
        (" Construção", "Construção"),
        ("+Viewpoints", "Viewpoints"),
        ("• Heapsort", "Heapsort"),
        ("(a) Altura", "(a) Altura"),
    ],
)
def test_titles_are_cleaned(raw, clean):
    assert clean_title(raw) == clean


def raw(file: str, first_page: str, heading: list[str] | None = None) -> RawDocument:
    return RawDocument(
        content=first_page,
        pages=[Page(number=1, text=first_page, start_offset=0, heading_lines=heading or [])],
        metadata=RawDocumentMetadata(source=file, file_path=file, num_pages=1, doc_type=DocumentType.PDF),
    )


LETTERHEAD = "UNIVERSIDADE FEDERAL DO RIO GRANDE DO NORTE\nInstituto Metrópole Digital\n"


def test_document_title():
    assert extract_document_title(raw("EDB2 09 AVL.pdf", "Árvores\nAVL", ["Árvores", "AVL"])) == "Árvores AVL"
    # No title on page 1: the file name.
    assert extract_document_title(raw("Lista_de_Exercicios 2.pdf", LETTERHEAD + "1) Escreva")) == "Lista de Exercicios 2"

    # A line opening many of the course's files is letterhead, not a title...
    docs = [raw(f"lista{i}.pdf", LETTERHEAD + f"Lista {i}") for i in range(4)]
    docs.append(raw("aula.pdf", "Heaps", ["UNIVERSIDADE FEDERAL DO RIO GRANDE DO NORTE", "Heaps"]))
    boilerplate = find_boilerplate_lines(docs)
    assert "universidade federal do rio grande do norte" in boilerplate
    assert extract_document_title(docs[-1], boilerplate) == "Heaps"

    # ...but a title a few decks share is not.
    shared = [raw(f"arvore{i}.pdf", "Árvores", ["Árvores"]) for i in range(3)] + [raw(f"o{i}.pdf", f"T {i}", [f"T {i}"]) for i in range(10)]
    assert "árvores" not in find_boilerplate_lines(shared)


@pytest.mark.parametrize(
    "line, number",
    [
        ("Chapter 2. Viewpoints", "2"),
        ("Aula 3 - Árvores AVL", "3"),
        ("Capítulo 12: Heaps", "12"),
        ("Unidade 2 – Grafos", "2"),
        ("Software Architecture in Action – Part I – Chapter 2. Viewpoints", "2"),
        ("Como vimos na aula 3 - árvores", None),  # a mention, not a header
    ],
)
def test_chapter_header(line, number):
    assert detect_chapter_header(f"{line}\nconteúdo")[0] == number


@pytest.mark.parametrize("name, number", [("SAIA-Chapter12.pdf", "12"), ("Aula_03.pdf", "03"), ("EDB2 09 AVL.pdf", None)])
def test_chapter_from_file_name(name, number):
    assert detect_chapter_from_filename(name) == number


# ---------------- slides into chunks ----------------

LONG = "uma árvore AVL é uma árvore binária de busca balanceada pela altura"


def slide(page: int, title: str, body: str, chapter: str | None = None) -> Section:
    offset = page * 1000
    blocks = [SemanticBlock(content=body, page_start=page, page_end=page, start_offset=offset, end_offset=offset + len(body))] if body else []
    return Section(
        id=str(page), title=title, page_start=page, page_end=page, start_offset=offset,
        end_offset=offset + len(body), content=body, blocks=blocks, chapter_number=chapter,
    )


def test_animation_steps_are_dropped():
    step, full = slide(1, "Inserção", "30\n20"), slide(2, "Inserção", "30\n20\n70")
    assert drop_slide_builds([step, full]) == [full]

    different = [slide(1, "AVL", LONG), slide(2, "Rotações", "rotação simples à direita")]
    assert drop_slide_builds(different) == different


def test_slides_are_grouped_with_their_titles():
    [group] = group_slides([slide(1, "AVL", LONG), slide(2, "Rotações", LONG)])

    assert (group.page_start, group.page_end, group.title) == (1, 2, "AVL")
    assert [b.content for b in group.blocks] == ["AVL", LONG, "Rotações", LONG]


@pytest.mark.parametrize(
    "slides, pages",
    [
        # A divider (a title alone) starts a new group, with what it introduces.
        ([slide(1, "Heap", LONG), slide(2, "Árvores AVL", ""), slide(3, "Definição", LONG)], [(1, 1), (2, 3)]),
        ([slide(1, "A", LONG, "1"), slide(2, "B", LONG, "2")], [(1, 1), (2, 2)]),  # chapter change
        ([slide(1, "AVL", LONG)], [(1, 1)]),
    ],
)
def test_slide_group_boundaries(slides, pages):
    assert [(g.page_start, g.page_end) for g in group_slides(slides)] == pages


def test_slide_groups_stay_under_the_budget():
    body = " ".join(["palavra"] * (SLIDE_GROUP_TOKENS // 3))
    groups = group_slides([slide(p, f"S{p}", body) for p in range(1, 8)])

    assert len(groups) > 1 and all(token_length(g.content) <= SLIDE_GROUP_TOKENS for g in groups)


# ---------------- keyword search ----------------


def chunk(i: int) -> Document:
    return Document(page_content=f"chunk {i}", metadata={"file_path": "a.pdf", "chunk_index": i, "start_offset": i})


def test_bm25_and_rank_fusion():
    assert tokenize("Árvore AVL: inserção!") == ["arvore", "avl", "insercao"]

    bm25 = BM25(["uma lista de prioridades guarda elementos", "o heapsort ordena usando um heap", "árvores binárias de busca"])
    assert bm25.top("como funciona o heapsort?", 3)[0] == 1
    assert BM25(["árvores AVL", "heap"]).top("grafos", 5) == []

    fused = reciprocal_rank_fusion([[chunk(1), chunk(2), chunk(3)], [chunk(3), chunk(4)]], 2)
    assert [d.metadata["chunk_index"] for d in fused] == [3, 1]
