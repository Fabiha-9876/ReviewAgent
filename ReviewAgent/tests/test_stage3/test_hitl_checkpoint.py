"""Tests for Stage3HITLCheckpoint — rubric scoring and feedback."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.common.schemas import IssueSpec, IssueCluster, RubricScores, ExtractedEntities
from src.stage3.hitl_checkpoint import Stage3HITLCheckpoint, RUBRIC_DIMENSIONS


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def hitl(tmp_path):
    return Stage3HITLCheckpoint(
        min_avg_score=3.0,
        scores_path=str(tmp_path / "rubric_scores.json"),
    )


@pytest.fixture
def sample_spec():
    return IssueSpec(
        issue_id="ISS-ABC123",
        cluster_id="CLU-001",
        title="App crashes on login",
        issue_type="bug_report",
        description="Users report crashes on login screen.",
    )


@pytest.fixture
def passing_scores():
    return {"completeness": 4, "accuracy": 4, "actionability": 3, "specificity": 4, "clarity": 5}


@pytest.fixture
def failing_scores():
    return {"completeness": 2, "accuracy": 2, "actionability": 1, "specificity": 2, "clarity": 2}


@pytest.fixture
def sample_cluster():
    return IssueCluster(
        cluster_id="CLU-001",
        issue_type="bug_report",
        aspect="login",
        sub_category="crash",
        review_ids=["r1"],
        review_count=1,
        representative_reviews=["App crashes on login"],
        entities=ExtractedEntities(),
        priority_score=0.8,
    )


# ============================================================
# Record Scores Tests
# ============================================================

class TestRecordScores:

    def test_record_scores_appends(self, hitl, passing_scores):
        hitl.record_scores("ISS-001", passing_scores, "rater_1")
        assert len(hitl.all_scores) == 1

    def test_record_scores_multiple_raters(self, hitl, passing_scores):
        hitl.record_scores("ISS-001", passing_scores, "rater_1")
        hitl.record_scores("ISS-001", passing_scores, "rater_2")
        assert len(hitl.all_scores) == 2

    def test_record_scores_persists_to_file(self, hitl, passing_scores):
        hitl.record_scores("ISS-001", passing_scores, "rater_1")
        data = json.loads(hitl.scores_path.read_text())
        assert len(data) == 1
        assert data[0]["spec_id"] == "ISS-001"
        assert data[0]["rater_id"] == "rater_1"

    def test_record_scores_includes_timestamp(self, hitl, passing_scores):
        hitl.record_scores("ISS-001", passing_scores, "rater_1")
        assert "timestamp" in hitl.all_scores[0]

    def test_record_scores_stores_all_dimensions(self, hitl, passing_scores):
        hitl.record_scores("ISS-001", passing_scores, "rater_1")
        stored = hitl.all_scores[0]["scores"]
        for dim in RUBRIC_DIMENSIONS:
            assert dim in stored


# ============================================================
# Check Threshold Tests
# ============================================================

class TestCheckThreshold:

    def test_passes_with_high_scores(self, hitl, sample_spec):
        sample_spec.rubric_scores = RubricScores(
            completeness=4, accuracy=4, actionability=4, specificity=4, clarity=4
        )
        assert hitl.check_threshold(sample_spec) is True

    def test_fails_with_low_scores(self, hitl, sample_spec):
        sample_spec.rubric_scores = RubricScores(
            completeness=1, accuracy=2, actionability=1, specificity=2, clarity=1
        )
        assert hitl.check_threshold(sample_spec) is False

    def test_fails_with_no_scores(self, hitl, sample_spec):
        sample_spec.rubric_scores = None
        assert hitl.check_threshold(sample_spec) is False

    def test_boundary_score_passes(self, hitl, sample_spec):
        """Exactly 3.0 mean should pass."""
        sample_spec.rubric_scores = RubricScores(
            completeness=3, accuracy=3, actionability=3, specificity=3, clarity=3
        )
        assert hitl.check_threshold(sample_spec) is True

    def test_custom_threshold(self, tmp_path):
        strict_hitl = Stage3HITLCheckpoint(
            min_avg_score=4.0,
            scores_path=str(tmp_path / "strict.json"),
        )
        spec = IssueSpec(
            issue_id="ISS-001", cluster_id="CLU-001",
            title="Test", issue_type="bug_report", description="Test",
        )
        spec.rubric_scores = RubricScores(
            completeness=3, accuracy=4, actionability=3, specificity=4, clarity=3
        )
        assert strict_hitl.check_threshold(spec) is False


# ============================================================
# Aggregate Scores Tests
# ============================================================

class TestAggregateScores:

    def test_aggregate_single_rater(self, hitl, passing_scores):
        hitl.record_scores("ISS-001", passing_scores, "rater_1")
        agg = hitl.aggregate_scores("ISS-001")
        assert agg is not None
        assert agg.completeness == 4
        assert agg.clarity == 5

    def test_aggregate_multiple_raters(self, hitl):
        hitl.record_scores("ISS-001", {"completeness": 4, "accuracy": 4, "actionability": 4, "specificity": 4, "clarity": 4}, "rater_1")
        hitl.record_scores("ISS-001", {"completeness": 2, "accuracy": 2, "actionability": 2, "specificity": 2, "clarity": 2}, "rater_2")
        agg = hitl.aggregate_scores("ISS-001")
        assert agg.completeness == 3.0
        assert agg.clarity == 3.0

    def test_aggregate_returns_none_for_unknown_spec(self, hitl):
        assert hitl.aggregate_scores("ISS-UNKNOWN") is None

    def test_aggregate_ignores_other_specs(self, hitl, passing_scores, failing_scores):
        hitl.record_scores("ISS-001", passing_scores, "rater_1")
        hitl.record_scores("ISS-002", failing_scores, "rater_1")
        agg = hitl.aggregate_scores("ISS-001")
        assert agg.completeness == 4


# ============================================================
# Feedback Summary & Weak Dimensions Tests
# ============================================================

class TestFeedbackSummary:

    def test_feedback_summary_returns_dict(self, hitl, passing_scores):
        hitl.record_scores("ISS-001", passing_scores, "rater_1")
        summary = hitl.get_feedback_summary("ISS-001")
        assert isinstance(summary, dict)
        assert "completeness" in summary

    def test_feedback_summary_empty_for_unknown(self, hitl):
        assert hitl.get_feedback_summary("ISS-UNKNOWN") == {}

    def test_weak_dimensions_identifies_low_scores(self, hitl, failing_scores):
        hitl.record_scores("ISS-001", failing_scores, "rater_1")
        weak = hitl.get_weak_dimensions("ISS-001")
        assert "completeness" in weak
        assert "actionability" in weak
        assert len(weak) == 5  # all below 3.0

    def test_weak_dimensions_empty_for_high_scores(self, hitl, passing_scores):
        hitl.record_scores("ISS-001", passing_scores, "rater_1")
        weak = hitl.get_weak_dimensions("ISS-001")
        assert weak == []

    def test_weak_dimensions_returns_all_when_no_scores(self, hitl):
        weak = hitl.get_weak_dimensions("ISS-UNKNOWN")
        assert weak == RUBRIC_DIMENSIONS


# ============================================================
# Regenerate With Feedback Tests
# ============================================================

class TestRegenerateWithFeedback:

    def test_regenerate_calls_translator(self, hitl, sample_spec, sample_cluster, failing_scores):
        hitl.record_scores(sample_spec.issue_id, failing_scores, "rater_1")

        mock_translator = MagicMock()
        new_spec = IssueSpec(
            issue_id="ISS-NEW", cluster_id="CLU-001",
            title="Improved title", issue_type="bug_report",
            description="Better description",
        )
        mock_translator.translate = AsyncMock(return_value=new_spec)

        result = asyncio.run(hitl.regenerate_with_feedback(
            sample_spec, mock_translator, sample_cluster
        ))
        mock_translator.translate.assert_called_once()

    def test_regenerate_preserves_issue_id(self, hitl, sample_spec, sample_cluster, failing_scores):
        hitl.record_scores(sample_spec.issue_id, failing_scores, "rater_1")

        mock_translator = MagicMock()
        new_spec = IssueSpec(
            issue_id="ISS-NEW", cluster_id="CLU-001",
            title="Improved", issue_type="bug_report",
            description="Better",
        )
        mock_translator.translate = AsyncMock(return_value=new_spec)

        result = asyncio.run(hitl.regenerate_with_feedback(
            sample_spec, mock_translator, sample_cluster
        ))
        assert result.issue_id == sample_spec.issue_id

    def test_regenerate_passes_feedback_in_kg_context(self, hitl, sample_spec, sample_cluster, failing_scores):
        hitl.record_scores(sample_spec.issue_id, failing_scores, "rater_1")

        mock_translator = MagicMock()
        mock_translator.translate = AsyncMock(return_value=sample_spec)

        asyncio.run(hitl.regenerate_with_feedback(
            sample_spec, mock_translator, sample_cluster
        ))
        call_kwargs = mock_translator.translate.call_args.kwargs
        assert "feedback" in call_kwargs["kg_context"]

    def test_no_regenerate_when_no_weak_dims(self, hitl, sample_spec, sample_cluster, passing_scores):
        hitl.record_scores(sample_spec.issue_id, passing_scores, "rater_1")

        mock_translator = MagicMock()
        result = asyncio.run(hitl.regenerate_with_feedback(
            sample_spec, mock_translator, sample_cluster
        ))
        mock_translator.translate.assert_not_called()
        assert result is sample_spec


# ============================================================
# Persistence Tests
# ============================================================

class TestPersistence:

    def test_loads_existing_scores(self, tmp_path, passing_scores):
        path = tmp_path / "scores.json"
        path.write_text(json.dumps([{
            "spec_id": "ISS-OLD",
            "scores": passing_scores,
            "rater_id": "rater_1",
            "timestamp": "2026-01-01T00:00:00",
        }]))
        hitl = Stage3HITLCheckpoint(scores_path=str(path))
        assert len(hitl.all_scores) == 1
        assert hitl.all_scores[0]["spec_id"] == "ISS-OLD"

    def test_creates_parent_dirs(self, tmp_path, passing_scores):
        path = tmp_path / "nested" / "dir" / "scores.json"
        hitl = Stage3HITLCheckpoint(scores_path=str(path))
        hitl.record_scores("ISS-001", passing_scores, "rater_1")
        assert path.exists()
