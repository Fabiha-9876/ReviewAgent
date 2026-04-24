"""Tests for Stage4bPipeline — orchestration with RAG and refinement."""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.common.schemas import ReviewObject, IssueSpec, GeneratedResponse
from src.stage4b.pipeline import Stage4bPipeline


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate = AsyncMock(return_value='{"specificity": "pass", "compliance": "pass", "empathy": "pass"}')
    return llm


@pytest.fixture
def reviews():
    return [
        ReviewObject(review_id="r1", text="App crashes on login", rating=1, app_id="app1", timestamp=datetime(2026, 3, 1)),
        ReviewObject(review_id="r2", text="Please add dark mode", rating=3, app_id="app1", timestamp=datetime(2026, 3, 2)),
    ]


@pytest.fixture
def specs():
    return [
        IssueSpec(issue_id="ISS-001", cluster_id="CLU-001", title="Login crash", issue_type="bug_report", description="Crash on login"),
        IssueSpec(issue_id="ISS-002", cluster_id="CLU-002", title="Dark mode request", issue_type="feature_request", description="Users want dark mode"),
    ]


@pytest.fixture
def pipeline(mock_llm):
    # Override LLM to return response text first, then critique
    call_count = [0]
    async def side_effect(**kwargs):
        call_count[0] += 1
        if "customer support" in kwargs.get("system_prompt", "").lower():
            return "Thank you for your feedback. We are working on this."
        return '{"specificity": "pass", "compliance": "pass", "empathy": "pass"}'
    mock_llm.generate = AsyncMock(side_effect=side_effect)
    return Stage4bPipeline(llm_client=mock_llm, retriever=None, max_refinement_iterations=1)


# ============================================================
# Process Tests
# ============================================================

class TestProcess:

    def test_returns_responses(self, pipeline, specs, reviews):
        results = asyncio.run(pipeline.process(specs, reviews, include_rag=False))
        assert len(results) == 2
        assert all(isinstance(r, GeneratedResponse) for r in results)

    def test_matches_reviews_to_specs(self, pipeline, specs, reviews):
        results = asyncio.run(pipeline.process(specs, reviews, include_rag=False, refine=False))
        ids = {r.review_id for r in results}
        assert "r1" in ids and "r2" in ids

    def test_extra_reviews_get_no_spec(self, pipeline, reviews):
        single_spec = [IssueSpec(issue_id="ISS-001", cluster_id="CLU-001", title="Test", issue_type="bug_report", description="Test")]
        results = asyncio.run(pipeline.process(single_spec, reviews, include_rag=False, refine=False))
        assert len(results) == 2

    def test_empty_reviews(self, pipeline, specs):
        results = asyncio.run(pipeline.process(specs, [], include_rag=False))
        assert results == []

    def test_no_refinement_when_disabled(self, mock_llm, reviews, specs):
        mock_llm.generate = AsyncMock(return_value="Simple response.")
        pipe = Stage4bPipeline(llm_client=mock_llm, retriever=None)
        results = asyncio.run(pipe.process(specs, reviews, include_rag=False, refine=False))
        assert all(r.refinement_iterations == 0 for r in results)

    def test_refinement_when_enabled(self, pipeline, reviews, specs):
        results = asyncio.run(pipeline.process(specs, reviews, include_rag=False, refine=True))
        assert all(isinstance(r, GeneratedResponse) for r in results)

    def test_no_issue_spec_when_disabled(self, mock_llm, reviews, specs):
        mock_llm.generate = AsyncMock(return_value="Response without spec.")
        pipe = Stage4bPipeline(llm_client=mock_llm, retriever=None)
        results = asyncio.run(pipe.process(specs, reviews, include_rag=False, include_issue_spec=False, refine=False))
        assert all(r.issue_id == "" for r in results)

    def test_single_review(self, pipeline, specs, reviews):
        results = asyncio.run(pipeline.process(specs[:1], reviews[:1], include_rag=False, refine=False))
        assert len(results) == 1
