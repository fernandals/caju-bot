"""
Slide/document detection, page titles from font sizes, document titles
and chapter markers (utils/helpers.py, rag/parser.py) - on PDFs built in
memory with PyMuPDF.
"""

import fitz
import pytest

from rag.models import DocumentType, Page, RawDocument, RawDocumentMetadata
from rag.parser import extract_document_title, find_boilerplate_lines
from utils.helpers import (
    clean_title,
    detect_chapter_from_filename,
    detect_chapter_header,
    detect_pdf_type,
    page_heading_lines,
)

LANDSCAPE = (720, 540)
PORTRAIT = (595, 842)
PROSE = (
    "Uma árvore binária de busca guarda chaves menores à esquerda e maiores "
    "à direita de cada nó, o que permite buscar, inserir e remover em tempo "
    "proporcional à altura da árvore. "
)


def pdf(pages: int, size, lines: list[tuple[float, str]]) -> fitz.Document:
    """Each page gets `lines`, written top to bottom at their font sizes."""
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page(width=size[0], height=size[1])
        y = 40
        for fontsize, text in lines:
            box = fitz.Rect(30, y, size[0] - 30, size[1] - 20)
            page.insert_textbox(box, text, fontsize=fontsize)
            y += fontsize * 1.6 * (1 + len(text) * fontsize * 0.5 // (size[0] - 60))
    return doc


class TestDetectPdfType:
    def test_landscape_deck(self):
        assert detect_pdf_type(pdf(5, LANDSCAPE, [(32, "Árvores AVL"), (20, "balanceada pela altura")])) == DocumentType.SLIDES

    def test_deck_exported_in_portrait(self):
        doc = pdf(5, PORTRAIT, [(40, "Complexidade Média"), (24, "busca em lista ordenada")])
        assert detect_pdf_type(doc) == DocumentType.SLIDES

    def test_dense_portrait_document(self):
        assert detect_pdf_type(pdf(3, PORTRAIT, [(10, PROSE * 6)])) == DocumentType.PDF


class TestPageHeadingLines:
    def test_title_larger_than_body(self):
        page = pdf(1, LANDSCAPE, [(32, "Inserção em AVL"), (18, "rotação simples"), (18, "rotação dupla")])[0]
        assert page_heading_lines(page) == ["Inserção em AVL"]

    def test_cover_with_only_a_title(self):
        page = pdf(1, LANDSCAPE, [(48, "Árvores"), (48, "B")])[0]
        assert page_heading_lines(page) == ["Árvores", "B"]

    def test_decoration_bigger_than_the_title_is_ignored(self):
        page = pdf(1, LANDSCAPE, [(54, "+"), (40, "Viewpoints"), (18, "a set of conventions")])[0]
        assert page_heading_lines(page) == ["Viewpoints"]

    def test_uniform_font_has_no_title(self):
        assert page_heading_lines(pdf(1, PORTRAIT, [(10, PROSE * 3)])[0]) == []


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
def test_clean_title(raw, clean):
    assert clean_title(raw) == clean


class TestChapters:
    @pytest.mark.parametrize(
        "line, number",
        [
            ("Chapter 2. Viewpoints", "2"),
            ("Aula 3 - Árvores AVL", "3"),
            ("Capítulo 12: Heaps", "12"),
            ("Unidade 2 – Grafos", "2"),
        ],
    )
    def test_running_header(self, line, number):
        assert detect_chapter_header(f"{line}\nconteúdo")[0] == number

    def test_a_passing_mention_is_not_a_header(self):
        assert detect_chapter_header("Como vimos na aula 3 - árvores")[0] is None

    @pytest.mark.parametrize("name, number", [("SAIA-Chapter12.pdf", "12"), ("Aula_03.pdf", "03"), ("EDB2 09 AVL.pdf", None)])
    def test_file_name(self, name, number):
        assert detect_chapter_from_filename(name) == number


def raw(file: str, first_page: str, heading: list[str] | None = None) -> RawDocument:
    return RawDocument(
        content=first_page,
        pages=[Page(number=1, text=first_page, start_offset=0, heading_lines=heading or [])],
        metadata=RawDocumentMetadata(source=file, file_path=file, num_pages=1, doc_type=DocumentType.PDF),
    )


LETTERHEAD = "UNIVERSIDADE FEDERAL DO RIO GRANDE DO NORTE\nInstituto Metrópole Digital\n"


class TestDocumentTitle:
    def test_largest_font_title(self):
        doc = raw("EDB2 09 AVL.pdf", "Árvores\nAVL", ["Árvores", "AVL"])
        assert extract_document_title(doc) == "Árvores AVL"

    def test_without_a_title_the_file_name(self):
        doc = raw("Lista_de_Exercicios 2.pdf", LETTERHEAD + "1) Escreva as funções")
        assert extract_document_title(doc) == "Lista de Exercicios 2"

    def test_letterhead_is_boilerplate(self):
        docs = [raw(f"lista{i}.pdf", LETTERHEAD + f"Lista {i}") for i in range(4)]
        docs.append(raw("aula.pdf", "Heaps", ["UNIVERSIDADE FEDERAL DO RIO GRANDE DO NORTE", "Heaps"]))

        boilerplate = find_boilerplate_lines(docs)

        assert "universidade federal do rio grande do norte" in boilerplate
        assert extract_document_title(docs[-1], boilerplate) == "Heaps"

    def test_a_title_shared_by_a_few_files_is_not_boilerplate(self):
        docs = [raw(f"arvore{i}.pdf", "Árvores", ["Árvores"]) for i in range(3)]
        docs += [raw(f"outro{i}.pdf", f"Tema {i}", [f"Tema {i}"]) for i in range(10)]

        assert "árvores" not in find_boilerplate_lines(docs)


def test_running_header_after_a_separator():
    header = "Software Architecture in Action – Part I – Chapter 2. Viewpoints"
    assert detect_chapter_header(f"{header}\nslide") == ("2", "Viewpoints")
