"""RAG retriever over 5 fixed sources using ChromaDB."""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedDocument:
    text: str
    source: str
    score: float
    metadata: dict


class RAGRetriever:
    """Retrieves context from 5 fixed sources using ChromaDB vector store."""

    SOURCES = ["past_responses", "changelogs", "faq", "issue_spec", "similar_responses"]

    def __init__(
        self,
        chroma_path: str = "data/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.encoder = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(path=chroma_path, settings=Settings(anonymized_telemetry=False))
        self._collections: dict[str, chromadb.Collection] = {}

    def _get_collection(self, source: str) -> chromadb.Collection:
        if source not in self._collections:
            self._collections[source] = self.client.get_or_create_collection(
                name=source, metadata={"hnsw:space": "cosine"}
            )
        return self._collections[source]

    def index_source(self, source: str, documents: list[dict]) -> None:
        """Index documents into a source collection.

        Args:
            documents: list of {"id": str, "text": str, "metadata": dict}
        """
        collection = self._get_collection(source)
        texts = [d["text"] for d in documents]
        embeddings = self.encoder.encode(texts).tolist()
        collection.add(
            ids=[d["id"] for d in documents],
            embeddings=embeddings,
            documents=texts,
            metadatas=[d.get("metadata", {}) for d in documents],
        )

    def retrieve(
        self,
        query: str,
        sources: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        """Retrieve relevant documents across specified sources."""
        sources = sources or self.SOURCES
        query_embedding = self.encoder.encode([query]).tolist()

        all_results = []
        for source in sources:
            try:
                collection = self._get_collection(source)
                if collection.count() == 0:
                    continue
                results = collection.query(
                    query_embeddings=query_embedding,
                    n_results=min(top_k, collection.count()),
                )
                for i in range(len(results["documents"][0])):
                    all_results.append(
                        RetrievedDocument(
                            text=results["documents"][0][i],
                            source=source,
                            score=1.0 - (results["distances"][0][i] if results["distances"] else 0),
                            metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        )
                    )
            except Exception:
                continue

        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]
