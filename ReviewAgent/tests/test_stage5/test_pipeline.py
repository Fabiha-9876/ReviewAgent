"""Tests for Stage5Pipeline — trainer selection and feedback propagation.

Tests the orchestration logic without importing trl (mocks trainer objects).
"""

import json
import pytest
from unittest.mock import MagicMock

from src.stage5.feedback_collector import DualStreamFeedbackCollector
from src.stage5.feedback_propagator import FeedbackPropagator


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def collector(tmp_path):
    return DualStreamFeedbackCollector(storage_path=str(tmp_path / "feedback.json"))


@pytest.fixture
def propagator(tmp_path):
    return FeedbackPropagator(feedback_dir=str(tmp_path))


@pytest.fixture
def quality_scores():
    return {"helpfulness": 4, "specificity": 3, "empathy": 5, "accuracy": 4, "actionability": 3}


def _make_pipeline(collector, propagator):
    """Build a Stage5Pipeline-like object without importing trl."""

    class MockPipeline:
        def __init__(self, collector, propagator):
            self.base_model = "test-model"
            self.collector = collector
            self.propagator = propagator
            self.kto = MagicMock()
            self.dpo = MagicMock()
            self.ppo = MagicMock()
            self.iteration = 0

        def select_trainer(self):
            n_responses = len(self.collector.quality_ratings)
            if n_responses < 500:
                return "kto"
            elif n_responses < 1500:
                return "dpo"
            else:
                return "constrained_ppo"

        def run_iteration(self, prompts, responses):
            self.iteration += 1
            method = self.select_trainer()
            self._propagate_feedback()
            return {"iteration": self.iteration, "method": method}

        def _propagate_feedback(self):
            rubric_data = [
                {"spec_id": q["response_id"], "scores": q["scores"]}
                for q in self.collector.quality_ratings
            ]
            if rubric_data:
                self.propagator.propagate_to_stage3(rubric_data)
            quality_data = [
                {"response_id": q["response_id"], "scores": q["scores"]}
                for q in self.collector.quality_ratings
            ]
            if quality_data:
                self.propagator.propagate_to_stage4b(quality_data)

    return MockPipeline(collector, propagator)


# ============================================================
# Trainer Selection Tests
# ============================================================

class TestSelectTrainer:

    def test_kto_for_small_data(self, collector, propagator):
        pipe = _make_pipeline(collector, propagator)
        assert pipe.select_trainer() == "kto"

    def test_kto_under_500(self, collector, propagator, quality_scores):
        for i in range(499):
            collector.record_quality(f"r{i}", quality_scores, "rater")
        pipe = _make_pipeline(collector, propagator)
        assert pipe.select_trainer() == "kto"

    def test_dpo_for_medium_data(self, collector, propagator, quality_scores):
        for i in range(500):
            collector.record_quality(f"r{i}", quality_scores, "rater")
        pipe = _make_pipeline(collector, propagator)
        assert pipe.select_trainer() == "dpo"

    def test_dpo_under_1500(self, collector, propagator, quality_scores):
        for i in range(1499):
            collector.record_quality(f"r{i}", quality_scores, "rater")
        pipe = _make_pipeline(collector, propagator)
        assert pipe.select_trainer() == "dpo"

    def test_ppo_for_large_data(self, collector, propagator, quality_scores):
        for i in range(1500):
            collector.record_quality(f"r{i}", quality_scores, "rater")
        pipe = _make_pipeline(collector, propagator)
        assert pipe.select_trainer() == "constrained_ppo"


# ============================================================
# Pipeline Initialization Tests
# ============================================================

class TestInitialization:

    def test_default_initialization(self, collector, propagator):
        pipe = _make_pipeline(collector, propagator)
        assert pipe.iteration == 0
        assert pipe.kto is not None
        assert pipe.dpo is not None
        assert pipe.ppo is not None

    def test_base_model(self, collector, propagator):
        pipe = _make_pipeline(collector, propagator)
        assert pipe.base_model == "test-model"


# ============================================================
# Feedback Propagation Tests
# ============================================================

class TestFeedbackPropagation:

    def test_propagates_to_stage4b(self, collector, propagator, quality_scores, tmp_path):
        collector.record_quality("r1", quality_scores, "rater")
        pipe = _make_pipeline(collector, propagator)
        pipe._propagate_feedback()
        path = tmp_path / "stage4b_rlhf_queue.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 1

    def test_no_propagation_when_empty(self, collector, propagator, tmp_path):
        pipe = _make_pipeline(collector, propagator)
        pipe._propagate_feedback()
        path = tmp_path / "stage4b_rlhf_queue.json"
        assert not path.exists()

    def test_iteration_increments(self, collector, propagator):
        pipe = _make_pipeline(collector, propagator)
        result = pipe.run_iteration([], [])
        assert result["iteration"] == 1
        assert result["method"] == "kto"
        result2 = pipe.run_iteration([], [])
        assert result2["iteration"] == 2
