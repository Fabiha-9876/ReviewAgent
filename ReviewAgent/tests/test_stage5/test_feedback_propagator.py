"""Tests for FeedbackPropagator — backward propagation to upstream stages."""

import json
import pytest

from src.stage5.feedback_propagator import FeedbackPropagator


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def propagator(tmp_path):
    return FeedbackPropagator(feedback_dir=str(tmp_path))


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


# ============================================================
# Propagate to Stage 1 Tests
# ============================================================

class TestPropagateToStage1:

    def test_saves_corrections(self, propagator, tmp_dir):
        corrections = [{"review_id": "r1", "corrected_labels": ["bug_report"]}]
        propagator.propagate_to_stage1(corrections)
        path = tmp_dir / "stage1_retraining_queue.json"
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert data[0]["review_id"] == "r1"

    def test_appends_to_existing(self, propagator, tmp_dir):
        propagator.propagate_to_stage1([{"review_id": "r1", "corrected_labels": ["bug_report"]}])
        propagator.propagate_to_stage1([{"review_id": "r2", "corrected_labels": ["praise"]}])
        path = tmp_dir / "stage1_retraining_queue.json"
        data = json.loads(path.read_text())
        assert len(data) == 2

    def test_empty_corrections(self, propagator, tmp_dir):
        propagator.propagate_to_stage1([])
        path = tmp_dir / "stage1_retraining_queue.json"
        data = json.loads(path.read_text())
        assert data == []


# ============================================================
# Propagate to Stage 3 Tests
# ============================================================

class TestPropagateToStage3:

    def test_returns_dimension_averages(self, propagator):
        feedback = [
            {"spec_id": "s1", "scores": {"completeness": 4, "accuracy": 2, "actionability": 3}},
            {"spec_id": "s2", "scores": {"completeness": 2, "accuracy": 4, "actionability": 3}},
        ]
        averages = propagator.propagate_to_stage3(feedback)
        assert averages["completeness"] == 3.0
        assert averages["accuracy"] == 3.0
        assert averages["actionability"] == 3.0

    def test_identifies_weak_dimensions(self, propagator, tmp_dir):
        feedback = [
            {"spec_id": "s1", "scores": {"completeness": 2, "accuracy": 1, "clarity": 4}},
        ]
        propagator.propagate_to_stage3(feedback)
        path = tmp_dir / "stage3_prompt_adjustments.json"
        data = json.loads(path.read_text())
        assert "completeness" in data["weak_dimensions"]
        assert "accuracy" in data["weak_dimensions"]
        assert "clarity" not in data["weak_dimensions"]

    def test_no_file_when_no_weak_dims(self, propagator, tmp_dir):
        feedback = [
            {"spec_id": "s1", "scores": {"completeness": 4, "accuracy": 4}},
        ]
        propagator.propagate_to_stage3(feedback)
        path = tmp_dir / "stage3_prompt_adjustments.json"
        assert not path.exists()

    def test_empty_feedback(self, propagator):
        averages = propagator.propagate_to_stage3([])
        assert averages == {}


# ============================================================
# Propagate to Stage 4b Tests
# ============================================================

class TestPropagateToStage4b:

    def test_saves_quality_scores(self, propagator, tmp_dir):
        scores = [{"response_id": "r1", "scores": {"helpfulness": 4}}]
        propagator.propagate_to_stage4b(scores)
        path = tmp_dir / "stage4b_rlhf_queue.json"
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert data[0]["response_id"] == "r1"

    def test_appends_to_existing(self, propagator, tmp_dir):
        propagator.propagate_to_stage4b([{"response_id": "r1", "scores": {"helpfulness": 4}}])
        propagator.propagate_to_stage4b([{"response_id": "r2", "scores": {"helpfulness": 3}}])
        path = tmp_dir / "stage4b_rlhf_queue.json"
        data = json.loads(path.read_text())
        assert len(data) == 2

    def test_empty_scores(self, propagator, tmp_dir):
        propagator.propagate_to_stage4b([])
        path = tmp_dir / "stage4b_rlhf_queue.json"
        data = json.loads(path.read_text())
        assert data == []


# ============================================================
# Directory Creation Tests
# ============================================================

class TestDirectoryCreation:

    def test_creates_feedback_dir(self, tmp_path):
        new_dir = tmp_path / "new_feedback_dir"
        prop = FeedbackPropagator(feedback_dir=str(new_dir))
        assert new_dir.exists()
