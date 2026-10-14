"""
How the student app searches a course's knowledge base.

Besides plain embedding search, a hybrid search fuses it with keyword
search (BM25): embeddings match meaning, keywords match the exact terms
course material is full of - "heapsort", "Método Mestre", a function's
name - which embeddings can rank below a vaguely related paragraph.
The two rankings are merged by reciprocal rank fusion (RRF).

BM25 is implemented here, over the chunks already stored in the vector
store, rather than adding a dependency for a few lines.
"""

import math
import re
import unicodedata
from collections import Counter

from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

# Standard BM25 parameters, and the usual RRF constant.
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60

# Each ranking contributes this many candidates to the fusion.
HYBRID_CANDIDATES = 20


def tokenize(text: str) -> list[str]:
    """Lowercase words without accents: "Árvore" and "arvore" match."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in re.findall(r"\w+", text) if len(t) > 1]


class BM25:
    def __init__(self, texts: list[str]):
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0

        frequency = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        scores = []

        for doc, length in zip(self.docs, self.lengths):
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    norm = BM25_K1 * (1 - BM25_B + BM25_B * length / (self.avg_length or 1))
                    score += self.idf[term] * tf * (BM25_K1 + 1) / (tf + norm)
            scores.append(score)

        return scores

    def top(self, query: str, n: int) -> list[int]:
        """Indexes of the n best-scoring documents with any matching term."""
        scores = self.scores(query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in ranked[:n] if scores[i] > 0]


def _key(doc: Document) -> tuple:
    meta = doc.metadata
    return (meta.get("file_path"), meta.get("chunk_index"), meta.get("start_offset"))


def reciprocal_rank_fusion(rankings: list[list[Document]], k: int) -> list[Document]:
    """The k documents with the best summed 1 / (RRF_K + rank) over the rankings."""
    scores: dict[tuple, float] = {}
    docs: dict[tuple, Document] = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            key = _key(doc)
            scores[key] = scores.get(key, 0.0) + 1 / (RRF_K + rank)
            docs.setdefault(key, doc)

    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [docs[key] for key in best]


class HybridRetriever(BaseRetriever):
    """Embedding search and BM25 over the same chunks, fused by RRF."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: Chroma
    k: int
    chunks: list[Document]
    bm25: BM25

    @classmethod
    def from_vectorstore(cls, vectorstore: Chroma, k: int) -> "HybridRetriever":
        stored = vectorstore.get(include=["documents", "metadatas"])
        chunks = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(stored["documents"], stored["metadatas"])
        ]
        return cls(vectorstore=vectorstore, k=k, chunks=chunks, bm25=BM25([c.page_content for c in chunks]))

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        dense = self.vectorstore.similarity_search(query, k=HYBRID_CANDIDATES)
        sparse = [self.chunks[i] for i in self.bm25.top(query, HYBRID_CANDIDATES)]
        return reciprocal_rank_fusion([dense, sparse], self.k)


def build_retriever(vectorstore: Chroma, search: str, k: int):
    """search: "mmr", "similarity" or "hybrid"."""
    if search == "hybrid":
        return HybridRetriever.from_vectorstore(vectorstore, k)

    if search == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": max(20, 4 * k), "lambda_mult": 0.7},
        )

    if search == "similarity":
        return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

    raise ValueError(f"Unknown search {search!r}")
