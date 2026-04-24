"""Tests for RAGRetriever — indexing and retrieval from ChromaDB."""

import pytest
from unittest.mock import MagicMock, patch

from src.stage4b.rag_retriever import RAGRetriever, RetrievedDocument


# ============================================================
# RetrievedDocument Tests
# ============================================================

class TestRetrievedDocument:

    def test_creation(self):
        doc = RetrievedDocument(text="hello", source="faq", score=0.9, metadata={"app": "test"})
        assert doc.text == "hello"
        assert doc.source == "faq"
        assert doc.score == 0.9
        assert doc.metadata == {"app": "test"}

    def test_default_metadata(self):
        doc = RetrievedDocument(text="hi", source="faq", score=0.5, metadata={})
        assert doc.metadata == {}


# ============================================================
# RAGRetriever Tests (mocked ChromaDB + SentenceTransformer)
# ============================================================

class TestRAGRetriever:

    @pytest.fixture
    def mock_retriever(self):
        with patch("src.stage4b.rag_retriever.SentenceTransformer") as mock_st, \
             patch("src.stage4b.rag_retriever.chromadb") as mock_chroma:
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = MagicMock(tolist=lambda: [[0.1, 0.2, 0.3]])
            mock_st.return_value = mock_encoder

            mock_client = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client

            retriever = RAGRetriever(chroma_path="/tmp/test_chroma", embedding_model="test-model")
            retriever._mock_client = mock_client
            retriever._mock_encoder = mock_encoder
            yield retriever

    def test_sources_defined(self):
        assert "past_responses" in RAGRetriever.SOURCES
        assert "changelogs" in RAGRetriever.SOURCES
        assert "faq" in RAGRetriever.SOURCES
        assert "issue_spec" in RAGRetriever.SOURCES
        assert "similar_responses" in RAGRetriever.SOURCES
        assert len(RAGRetriever.SOURCES) == 5

    def test_get_collection_creates_once(self, mock_retriever):
        mock_col = MagicMock()
        mock_retriever._mock_client.get_or_create_collection.return_value = mock_col

        col1 = mock_retriever._get_collection("faq")
        col2 = mock_retriever._get_collection("faq")
        assert col1 is col2
        mock_retriever._mock_client.get_or_create_collection.assert_called_once()

    def test_get_collection_different_sources(self, mock_retriever):
        mock_retriever._mock_client.get_or_create_collection.side_effect = [MagicMock(), MagicMock()]
        col1 = mock_retriever._get_collection("faq")
        col2 = mock_retriever._get_collection("changelogs")
        assert col1 is not col2

    def test_index_source_calls_add(self, mock_retriever):
        mock_col = MagicMock()
        mock_retriever._mock_client.get_or_create_collection.return_value = mock_col
        mock_retriever._mock_encoder.encode.return_value = MagicMock(
            tolist=lambda: [[0.1, 0.2], [0.3, 0.4]]
        )

        docs = [
            {"id": "d1", "text": "hello", "metadata": {"app": "test"}},
            {"id": "d2", "text": "world", "metadata": {"app": "test"}},
        ]
        mock_retriever.index_source("faq", docs)

        mock_col.add.assert_called_once()
        call_kwargs = mock_col.add.call_args.kwargs
        assert call_kwargs["ids"] == ["d1", "d2"]
        assert call_kwargs["documents"] == ["hello", "world"]

    def test_index_source_default_metadata(self, mock_retriever):
        mock_col = MagicMock()
        mock_retriever._mock_client.get_or_create_collection.return_value = mock_col
        mock_retriever._mock_encoder.encode.return_value = MagicMock(
            tolist=lambda: [[0.1, 0.2]]
        )

        docs = [{"id": "d1", "text": "hello"}]
        mock_retriever.index_source("faq", docs)

        call_kwargs = mock_col.add.call_args.kwargs
        assert call_kwargs["metadatas"] == [{}]

    def test_retrieve_returns_documents(self, mock_retriever):
        mock_col = MagicMock()
        mock_col.count.return_value = 2
        mock_col.query.return_value = {
            "documents": [["doc1 text", "doc2 text"]],
            "distances": [[0.1, 0.3]],
            "metadatas": [[{"app": "a"}, {"app": "b"}]],
        }
        mock_retriever._mock_client.get_or_create_collection.return_value = mock_col

        results = mock_retriever.retrieve("test query", sources=["faq"], top_k=5)
        assert len(results) == 2
        assert all(isinstance(r, RetrievedDocument) for r in results)

    def test_retrieve_scores_calculated(self, mock_retriever):
        mock_col = MagicMock()
        mock_col.count.return_value = 1
        mock_col.query.return_value = {
            "documents": [["doc text"]],
            "distances": [[0.2]],
            "metadatas": [[{}]],
        }
        mock_retriever._mock_client.get_or_create_collection.return_value = mock_col

        results = mock_retriever.retrieve("test", sources=["faq"], top_k=5)
        assert results[0].score == pytest.approx(0.8)  # 1.0 - 0.2

    def test_retrieve_sorted_by_score_descending(self, mock_retriever):
        mock_col = MagicMock()
        mock_col.count.return_value = 3
        mock_col.query.return_value = {
            "documents": [["low", "mid", "high"]],
            "distances": [[0.9, 0.5, 0.1]],
            "metadatas": [[{}, {}, {}]],
        }
        mock_retriever._mock_client.get_or_create_collection.return_value = mock_col

        results = mock_retriever.retrieve("test", sources=["faq"], top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_respects_top_k(self, mock_retriever):
        mock_col = MagicMock()
        mock_col.count.return_value = 5
        mock_col.query.return_value = {
            "documents": [["a", "b", "c", "d", "e"]],
            "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
            "metadatas": [[{}, {}, {}, {}, {}]],
        }
        mock_retriever._mock_client.get_or_create_collection.return_value = mock_col

        results = mock_retriever.retrieve("test", sources=["faq"], top_k=3)
        assert len(results) == 3

    def test_retrieve_skips_empty_collection(self, mock_retriever):
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_retriever._mock_client.get_or_create_collection.return_value = mock_col

        results = mock_retriever.retrieve("test", sources=["faq"], top_k=5)
        assert results == []

    def test_retrieve_handles_exception_gracefully(self, mock_retriever):
        mock_col = MagicMock()
        mock_col.count.side_effect = Exception("DB error")
        mock_retriever._mock_client.get_or_create_collection.return_value = mock_col

        results = mock_retriever.retrieve("test", sources=["faq"], top_k=5)
        assert results == []

    def test_retrieve_uses_all_sources_by_default(self, mock_retriever):
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_retriever._mock_client.get_or_create_collection.return_value = mock_col

        mock_retriever.retrieve("test")
        # Should attempt to get collection for all 5 sources
        assert mock_retriever._mock_client.get_or_create_collection.call_count == 5

    def test_retrieve_sets_source_on_documents(self, mock_retriever):
        mock_col = MagicMock()
        mock_col.count.return_value = 1
        mock_col.query.return_value = {
            "documents": [["test doc"]],
            "distances": [[0.1]],
            "metadatas": [[{}]],
        }
        mock_retriever._mock_client.get_or_create_collection.return_value = mock_col

        results = mock_retriever.retrieve("test", sources=["changelogs"], top_k=5)
        assert results[0].source == "changelogs"
