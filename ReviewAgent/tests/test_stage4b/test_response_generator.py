"""Tests for ResponseGenerator — context building and response generation."""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from src.common.schemas import ReviewObject, IssueSpec, GeneratedResponse, ExtractedEntities
from src.stage4b.response_generator import ResponseGenerator, SYSTEM_PROMPT


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate = AsyncMock(return_value="Thank you for your feedback. We're aware of the crash on login and our team is working on a fix. Please try updating to the latest version.")
    return llm


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    doc = MagicMock()
    doc.text = "We fixed the login crash in v3.3."
    doc.source = "changelogs"
    doc.score = 0.8
    retriever.retrieve.return_value = [doc]
    return retriever


@pytest.fixture
def review():
    return ReviewObject(
        review_id="rev-001",
        text="App crashes every time I try to login!",
        rating=1,
        app_id="com.test.app",
        timestamp=datetime(2026, 3, 15),
        labels=["bug_report"],
    )


@pytest.fixture
def issue_spec():
    return IssueSpec(
        issue_id="ISS-ABC",
        cluster_id="CLU-001",
        title="Login crash on Android 14",
        issue_type="bug_report",
        description="App crashes during login",
        steps_to_reproduce=["Open app", "Tap login", "App crashes"],
        actual_behavior="App crashes with no error",
        severity="P1",
        affected_component="auth_service",
        priority_score=0.85,
    )


# ============================================================
# Build Context Tests
# ============================================================

class TestBuildContext:

    def test_includes_review_text(self, mock_llm, review):
        gen = ResponseGenerator(mock_llm)
        ctx = gen._build_context(review, None, False)
        assert "App crashes every time I try to login!" in ctx

    def test_includes_rating(self, mock_llm, review):
        gen = ResponseGenerator(mock_llm)
        ctx = gen._build_context(review, None, False)
        assert "1/5" in ctx

    def test_includes_issue_spec(self, mock_llm, review, issue_spec):
        gen = ResponseGenerator(mock_llm)
        ctx = gen._build_context(review, issue_spec, False)
        assert "Login crash on Android 14" in ctx
        assert "bug_report" in ctx
        assert "P1" in ctx
        assert "auth_service" in ctx

    def test_includes_actual_behavior(self, mock_llm, review, issue_spec):
        gen = ResponseGenerator(mock_llm)
        ctx = gen._build_context(review, issue_spec, False)
        assert "App crashes with no error" in ctx

    def test_includes_steps_to_reproduce(self, mock_llm, review, issue_spec):
        gen = ResponseGenerator(mock_llm)
        ctx = gen._build_context(review, issue_spec, False)
        assert "Open app" in ctx

    def test_includes_priority(self, mock_llm, review, issue_spec):
        gen = ResponseGenerator(mock_llm)
        ctx = gen._build_context(review, issue_spec, False)
        assert "85%" in ctx

    def test_no_issue_spec_section_when_none(self, mock_llm, review):
        gen = ResponseGenerator(mock_llm)
        ctx = gen._build_context(review, None, False)
        assert "Structured Issue Analysis" not in ctx

    def test_includes_rag_context(self, mock_llm, mock_retriever, review):
        gen = ResponseGenerator(mock_llm, mock_retriever)
        ctx = gen._build_context(review, None, True)
        assert "Reference Information" in ctx
        assert "login crash" in ctx.lower() or "changelogs" in ctx

    def test_no_rag_when_disabled(self, mock_llm, mock_retriever, review):
        gen = ResponseGenerator(mock_llm, mock_retriever)
        ctx = gen._build_context(review, None, False)
        assert "Reference Information" not in ctx

    def test_no_rag_when_no_retriever(self, mock_llm, review):
        gen = ResponseGenerator(mock_llm, None)
        ctx = gen._build_context(review, None, True)
        assert "Reference Information" not in ctx

    def test_rag_query_includes_issue_title(self, mock_llm, mock_retriever, review, issue_spec):
        gen = ResponseGenerator(mock_llm, mock_retriever)
        gen._build_context(review, issue_spec, True)
        query_arg = mock_retriever.retrieve.call_args[0][0]
        assert "Login crash on Android 14" in query_arg


# ============================================================
# Generate Tests
# ============================================================

class TestGenerate:

    def test_returns_generated_response(self, mock_llm, review):
        gen = ResponseGenerator(mock_llm)
        resp = asyncio.run(gen.generate(review, include_rag=False))
        assert isinstance(resp, GeneratedResponse)

    def test_response_has_review_id(self, mock_llm, review):
        gen = ResponseGenerator(mock_llm)
        resp = asyncio.run(gen.generate(review, include_rag=False))
        assert resp.review_id == "rev-001"

    def test_response_has_issue_id_when_spec(self, mock_llm, review, issue_spec):
        gen = ResponseGenerator(mock_llm)
        resp = asyncio.run(gen.generate(review, issue_spec, include_rag=False))
        assert resp.issue_id == "ISS-ABC"

    def test_response_empty_issue_id_without_spec(self, mock_llm, review):
        gen = ResponseGenerator(mock_llm)
        resp = asyncio.run(gen.generate(review, None, include_rag=False))
        assert resp.issue_id == ""

    def test_response_has_text(self, mock_llm, review):
        gen = ResponseGenerator(mock_llm)
        resp = asyncio.run(gen.generate(review, include_rag=False))
        assert len(resp.text) > 0

    def test_response_has_response_id(self, mock_llm, review):
        gen = ResponseGenerator(mock_llm)
        resp = asyncio.run(gen.generate(review, include_rag=False))
        assert len(resp.response_id) > 0

    def test_refinement_iterations_zero(self, mock_llm, review):
        gen = ResponseGenerator(mock_llm)
        resp = asyncio.run(gen.generate(review, include_rag=False))
        assert resp.refinement_iterations == 0

    def test_rag_sources_when_enabled(self, mock_llm, mock_retriever, review):
        gen = ResponseGenerator(mock_llm, mock_retriever)
        resp = asyncio.run(gen.generate(review, include_rag=True))
        assert len(resp.rag_sources_used) > 0

    def test_rag_sources_include_issue_spec(self, mock_llm, review, issue_spec):
        gen = ResponseGenerator(mock_llm)
        resp = asyncio.run(gen.generate(review, issue_spec, include_rag=False))
        assert "issue_spec" in resp.rag_sources_used

    def test_llm_called_with_system_prompt(self, mock_llm, review):
        gen = ResponseGenerator(mock_llm)
        asyncio.run(gen.generate(review, include_rag=False))
        call_kwargs = mock_llm.generate.call_args.kwargs
        assert call_kwargs["system_prompt"] == SYSTEM_PROMPT

    def test_llm_called_with_correct_params(self, mock_llm, review):
        gen = ResponseGenerator(mock_llm)
        asyncio.run(gen.generate(review, include_rag=False))
        call_kwargs = mock_llm.generate.call_args.kwargs
        assert call_kwargs["temperature"] == 0.4
        assert call_kwargs["max_tokens"] == 512


# ============================================================
# Generate Batch Tests
# ============================================================

class TestGenerateBatch:

    def test_batch_returns_correct_count(self, mock_llm):
        gen = ResponseGenerator(mock_llm)
        reviews = [
            ReviewObject(review_id="r1", text="Bad app", rating=1, app_id="a", timestamp=datetime(2026, 1, 1)),
            ReviewObject(review_id="r2", text="Great app", rating=5, app_id="a", timestamp=datetime(2026, 1, 1)),
        ]
        specs = [None, None]
        results = asyncio.run(gen.generate_batch(reviews, specs, include_rag=False))
        assert len(results) == 2

    def test_batch_preserves_review_ids(self, mock_llm):
        gen = ResponseGenerator(mock_llm)
        reviews = [
            ReviewObject(review_id="r1", text="Bad", rating=1, app_id="a", timestamp=datetime(2026, 1, 1)),
            ReviewObject(review_id="r2", text="Good", rating=5, app_id="a", timestamp=datetime(2026, 1, 1)),
        ]
        results = asyncio.run(gen.generate_batch(reviews, [None, None], include_rag=False))
        ids = {r.review_id for r in results}
        assert "r1" in ids and "r2" in ids

    def test_batch_empty_input(self, mock_llm):
        gen = ResponseGenerator(mock_llm)
        results = asyncio.run(gen.generate_batch([], [], include_rag=False))
        assert results == []
