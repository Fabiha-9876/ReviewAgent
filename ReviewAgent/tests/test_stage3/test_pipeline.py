"""Tests for Stage3Pipeline — orchestration with and without HITL."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.common.schemas import IssueCluster, IssueSpec, ExtractedEntities, RubricScores
from src.stage3.pipeline import Stage3Pipeline


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate = AsyncMock(
        return_value="**Title:** Test issue\n\n**Description:** A test issue.\n\n**Severity:** P2\n\n**Affected Component:** test_module"
    )
    return llm


@pytest.fixture
def clusters():
    return [
        IssueCluster(
            cluster_id="CLU-001",
            issue_type="bug_report",
            aspect="login",
            sub_category="crash",
            review_ids=["r1", "r2"],
            review_count=2,
            representative_reviews=["App crashes on login", "Login fails"],
            entities=ExtractedEntities(devices=["Pixel 8"]),
            priority_score=0.9,
        ),
        IssueCluster(
            cluster_id="CLU-002",
            issue_type="feature_request",
            aspect="dark_mode",
            sub_category="theme",
            review_ids=["r3"],
            review_count=1,
            representative_reviews=["Add dark mode please"],
            entities=ExtractedEntities(),
            priority_score=0.4,
        ),
    ]


@pytest.fixture
def pipeline(mock_llm, tmp_path):
    return Stage3Pipeline(
        llm_client=mock_llm,
        hitl=MagicMock(
            min_avg_score=3.0,
            scores_path=str(tmp_path / "scores.json"),
            all_scores=[],
        ),
    )


# ============================================================
# Process (without HITL) Tests
# ============================================================

class TestProcess:

    def test_returns_issue_specs(self, pipeline, clusters):
        specs = asyncio.run(pipeline.process(clusters))
        assert len(specs) == 2
        assert all(isinstance(s, IssueSpec) for s in specs)

    def test_specs_have_correct_cluster_ids(self, pipeline, clusters):
        specs = asyncio.run(pipeline.process(clusters))
        ids = {s.cluster_id for s in specs}
        assert "CLU-001" in ids
        assert "CLU-002" in ids

    def test_empty_input_returns_empty(self, pipeline):
        specs = asyncio.run(pipeline.process([]))
        assert specs == []

    def test_single_cluster(self, pipeline, clusters):
        specs = asyncio.run(pipeline.process([clusters[0]]))
        assert len(specs) == 1
        assert specs[0].cluster_id == "CLU-001"

    def test_passes_kg_context(self, pipeline, clusters, mock_llm):
        kg_map = {"CLU-001": {"related": ["CLU-003"]}}
        asyncio.run(pipeline.process(clusters, kg_context_map=kg_map))
        # LLM should have been called with kg context for CLU-001
        calls = mock_llm.generate.call_args_list
        assert any("CLU-003" in str(c) for c in calls)


# ============================================================
# Process with HITL Tests
# ============================================================

class TestProcessWithHITL:

    def test_auto_approve_when_no_callback(self, mock_llm, clusters, tmp_path):
        from src.stage3.hitl_checkpoint import Stage3HITLCheckpoint
        hitl = Stage3HITLCheckpoint(scores_path=str(tmp_path / "scores.json"))
        pipe = Stage3Pipeline(llm_client=mock_llm, hitl=hitl)
        specs = asyncio.run(pipe.process_with_hitl(clusters, score_callback=None))
        assert len(specs) == 2

    def test_validates_with_passing_scores(self, mock_llm, clusters, tmp_path):
        from src.stage3.hitl_checkpoint import Stage3HITLCheckpoint
        hitl = Stage3HITLCheckpoint(scores_path=str(tmp_path / "scores.json"))
        pipe = Stage3Pipeline(llm_client=mock_llm, hitl=hitl)

        def good_scores(spec):
            return {d: 4 for d in ["completeness", "accuracy", "actionability", "specificity", "clarity"]}

        specs = asyncio.run(pipe.process_with_hitl(clusters, score_callback=good_scores))
        assert all(s.validated for s in specs)

    def test_retries_on_failing_scores(self, mock_llm, clusters, tmp_path):
        from src.stage3.hitl_checkpoint import Stage3HITLCheckpoint
        hitl = Stage3HITLCheckpoint(scores_path=str(tmp_path / "scores.json"))
        pipe = Stage3Pipeline(llm_client=mock_llm, hitl=hitl)

        call_count = [0]

        def improving_scores(spec):
            call_count[0] += 1
            if call_count[0] <= 2:
                return {d: 1 for d in ["completeness", "accuracy", "actionability", "specificity", "clarity"]}
            return {d: 4 for d in ["completeness", "accuracy", "actionability", "specificity", "clarity"]}

        specs = asyncio.run(pipe.process_with_hitl(
            [clusters[0]], score_callback=improving_scores, max_retries=2
        ))
        assert call_count[0] == 3  # initial + 2 retries

    def test_max_retries_respected(self, mock_llm, clusters, tmp_path):
        from src.stage3.hitl_checkpoint import Stage3HITLCheckpoint
        hitl = Stage3HITLCheckpoint(scores_path=str(tmp_path / "scores.json"))
        pipe = Stage3Pipeline(llm_client=mock_llm, hitl=hitl)

        call_count = [0]

        def always_fail(spec):
            call_count[0] += 1
            return {d: 1 for d in ["completeness", "accuracy", "actionability", "specificity", "clarity"]}

        specs = asyncio.run(pipe.process_with_hitl(
            [clusters[0]], score_callback=always_fail, max_retries=2
        ))
        assert call_count[0] == 3  # initial + 2 retries, then stops
        assert specs[0].validated is False

    def test_still_returns_spec_on_failure(self, mock_llm, clusters, tmp_path):
        from src.stage3.hitl_checkpoint import Stage3HITLCheckpoint
        hitl = Stage3HITLCheckpoint(scores_path=str(tmp_path / "scores.json"))
        pipe = Stage3Pipeline(llm_client=mock_llm, hitl=hitl)

        def always_fail(spec):
            return {d: 1 for d in ["completeness", "accuracy", "actionability", "specificity", "clarity"]}

        specs = asyncio.run(pipe.process_with_hitl(
            [clusters[0]], score_callback=always_fail, max_retries=0
        ))
        assert len(specs) == 1
        assert isinstance(specs[0], IssueSpec)
