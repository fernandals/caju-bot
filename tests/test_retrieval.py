"""rag/retrieval.py: BM25 and reciprocal rank fusion (no embeddings)."""

from langchain_core.documents import Document

from rag.retrieval import BM25, reciprocal_rank_fusion, tokenize


def doc(i: int) -> Document:
    return Document(page_content=f"chunk {i}", metadata={"file_path": "a.pdf", "chunk_index": i, "start_offset": i})


def test_tokenize_ignores_case_and_accents():
    assert tokenize("Árvore AVL: inserção!") == ["arvore", "avl", "insercao"]


def test_bm25_ranks_exact_terms_first():
    bm25 = BM25(
        [
            "uma lista de prioridades guarda elementos",
            "o heapsort ordena usando um heap",
            "árvores binárias de busca",
        ]
    )

    assert bm25.top("como funciona o heapsort?", 3)[0] == 1


def test_bm25_returns_only_matching_documents():
    assert BM25(["árvores AVL", "heap"]).top("grafos", 5) == []


def test_fusion_rewards_documents_in_both_rankings():
    dense = [doc(1), doc(2), doc(3)]
    sparse = [doc(3), doc(4)]

    assert [d.metadata["chunk_index"] for d in reciprocal_rank_fusion([dense, sparse], 2)] == [3, 1]
