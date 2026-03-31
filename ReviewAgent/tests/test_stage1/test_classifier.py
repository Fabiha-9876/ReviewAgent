"""Tests for the RoBERTa multi-label review classifier."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.stage1.classifier import ReviewClassifier, LABELS


class TestLabels:
    """Test that label definitions are correct."""

    def test_label_count(self):
        assert len(LABELS) == 7

    def test_expected_labels(self):
        expected = ["bug_report", "feature_request", "performance",
                    "usability", "compatibility", "praise", "other"]
        assert LABELS == expected


class TestNeedsHITL:
    """Test the confidence-based HITL flagging logic."""

    def test_high_confidence_no_hitl(self):
        """High confidence on one label → no HITL needed."""
        classifier = ReviewClassifier.__new__(ReviewClassifier)
        classifier.confidence_threshold = 0.7
        classifier.conflict_margin = 0.15

        confidences = {
            "bug_report": 0.95, "feature_request": 0.1, "performance": 0.05,
            "usability": 0.02, "compatibility": 0.01, "praise": 0.01, "other": 0.01,
        }
        assert classifier.needs_hitl(confidences) is False

    def test_low_confidence_needs_hitl(self):
        """Max confidence below threshold → needs HITL."""
        classifier = ReviewClassifier.__new__(ReviewClassifier)
        classifier.confidence_threshold = 0.7
        classifier.conflict_margin = 0.15

        confidences = {
            "bug_report": 0.55, "feature_request": 0.3, "performance": 0.1,
            "usability": 0.05, "compatibility": 0.02, "praise": 0.01, "other": 0.01,
        }
        assert classifier.needs_hitl(confidences) is True

    def test_conflicting_labels_needs_hitl(self):
        """Top two labels within conflict margin → needs HITL."""
        classifier = ReviewClassifier.__new__(ReviewClassifier)
        classifier.confidence_threshold = 0.7
        classifier.conflict_margin = 0.15

        confidences = {
            "bug_report": 0.75, "feature_request": 0.70, "performance": 0.1,
            "usability": 0.02, "compatibility": 0.01, "praise": 0.01, "other": 0.01,
        }
        # Difference is 0.05 < margin 0.15 → conflict
        assert classifier.needs_hitl(confidences) is True

    def test_no_conflict_above_margin(self):
        """Top two labels far apart → no conflict."""
        classifier = ReviewClassifier.__new__(ReviewClassifier)
        classifier.confidence_threshold = 0.7
        classifier.conflict_margin = 0.15

        confidences = {
            "bug_report": 0.90, "feature_request": 0.40, "performance": 0.1,
            "usability": 0.02, "compatibility": 0.01, "praise": 0.01, "other": 0.01,
        }
        # Difference is 0.50 > margin 0.15 → no conflict
        assert classifier.needs_hitl(confidences) is False

    def test_exactly_at_threshold(self):
        """Confidence exactly at threshold → no HITL (not below)."""
        classifier = ReviewClassifier.__new__(ReviewClassifier)
        classifier.confidence_threshold = 0.7
        classifier.conflict_margin = 0.15

        confidences = {
            "bug_report": 0.70, "feature_request": 0.3, "performance": 0.1,
            "usability": 0.02, "compatibility": 0.01, "praise": 0.01, "other": 0.01,
        }
        assert classifier.needs_hitl(confidences) is False

    def test_all_low_confidence(self):
        """All labels have low confidence → needs HITL."""
        classifier = ReviewClassifier.__new__(ReviewClassifier)
        classifier.confidence_threshold = 0.7
        classifier.conflict_margin = 0.15

        confidences = {l: 0.15 for l in LABELS}
        assert classifier.needs_hitl(confidences) is True


class TestPredictWithTrainedModel:
    """Test predict() with the actual trained model (if available)."""

    @pytest.fixture
    def trained_classifier(self):
        """Load the trained model if it exists, skip otherwise."""
        from pathlib import Path
        model_path = Path("models/stage1_classifier")
        if not (model_path / "model.safetensors").exists():
            pytest.skip("Trained model not found — run train_classifier.py first")
        return ReviewClassifier(model_name_or_path=str(model_path))

    def test_predict_returns_correct_format(self, trained_classifier):
        """predict() returns list of (labels, confidences) tuples."""
        results = trained_classifier.predict(["App crashes on login"])
        assert len(results) == 1
        labels, confidences = results[0]
        assert isinstance(labels, list)
        assert isinstance(confidences, dict)
        assert len(confidences) == 7
        assert all(label in LABELS for label in labels)
        assert all(0.0 <= v <= 1.0 for v in confidences.values())

    def test_predict_batch(self, trained_classifier):
        """predict() handles multiple reviews."""
        texts = ["App crashes", "Love this app!", "Please add dark mode"]
        results = trained_classifier.predict(texts)
        assert len(results) == 3

    def test_bug_report_classified(self, trained_classifier):
        """A clear bug report should be classified as bug_report."""
        results = trained_classifier.predict(["App crashes every time I try to login"])
        labels, confs = results[0]
        assert "bug_report" in labels

    def test_praise_classified(self, trained_classifier):
        """A clear praise review should be classified as praise."""
        results = trained_classifier.predict(["Love this app! Best ever! 5 stars!"])
        labels, confs = results[0]
        assert "praise" in labels

    def test_feature_request_classified(self, trained_classifier):
        """A clear feature request should be classified correctly."""
        results = trained_classifier.predict(["I wish you had dark mode"])
        labels, confs = results[0]
        assert "feature_request" in labels

    def test_always_returns_at_least_one_label(self, trained_classifier):
        """Even ambiguous text should get at least one label."""
        results = trained_classifier.predict(["ok"])
        labels, confs = results[0]
        assert len(labels) >= 1

    def test_empty_text_handled(self, trained_classifier):
        """Empty or very short text should not crash."""
        results = trained_classifier.predict(["", "a", "   "])
        assert len(results) == 3
        for labels, confs in results:
            assert len(labels) >= 1
