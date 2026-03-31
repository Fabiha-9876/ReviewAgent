"""Tests for evaluation metrics."""

import pytest
import numpy as np
from src.common.schemas import IssueSpec, ExtractedEntities
from src.evaluation.metrics import (
    compute_completeness_ratio,
    aggregate_rubric_scores,
)
from src.evaluation.statistical_tests import (
    paired_wilcoxon,
    friedman_test,
    mcnemar_test,
    bonferroni_correction,
    bradley_terry,
)


class TestCompletenessRatio:
    def test_fully_complete_bug(self):
        spec = IssueSpec(
            issue_id="test",
            cluster_id="test",
            title="Login crash",
            issue_type="bug_report",
            description="Crashes on login",
            steps_to_reproduce=["Open app", "Tap login"],
            expected_behavior="Login works",
            actual_behavior="App crashes",
            severity="P0",
            affected_component="auth",
            environment=ExtractedEntities(devices=["Pixel"]),
        )
        ratio = compute_completeness_ratio(spec)
        assert ratio == 1.0

    def test_partial_bug(self):
        spec = IssueSpec(
            issue_id="test",
            cluster_id="test",
            title="Login crash",
            issue_type="bug_report",
            description="Crashes on login",
            severity="P0",
        )
        ratio = compute_completeness_ratio(spec)
        assert 0 < ratio < 1.0

    def test_feature_request(self):
        spec = IssueSpec(
            issue_id="test",
            cluster_id="test",
            title="Dark mode",
            issue_type="feature_request",
            description="Add dark mode",
            user_story="As a user I want dark mode",
            acceptance_criteria=["Toggle exists"],
            severity="P2",
            affected_component="ui",
        )
        ratio = compute_completeness_ratio(spec)
        assert ratio == 1.0


class TestAggregateRubricScores:
    def test_aggregate(self):
        scores = [
            {"completeness": 4, "accuracy": 3},
            {"completeness": 5, "accuracy": 4},
            {"completeness": 3, "accuracy": 5},
        ]
        agg = aggregate_rubric_scores(scores)
        assert agg["completeness"] == 4.0
        assert agg["accuracy"] == 4.0


class TestStatisticalTests:
    def test_wilcoxon(self):
        a = [4, 5, 3, 4, 5, 3, 4, 5, 4, 3]
        b = [2, 3, 2, 3, 2, 2, 3, 3, 2, 2]
        result = paired_wilcoxon(a, b)
        assert result["p_value"] < 0.05
        assert result["cliffs_delta"] > 0

    def test_friedman(self):
        c1 = [4, 5, 3, 4, 5]
        c2 = [2, 3, 2, 3, 2]
        c3 = [3, 4, 3, 3, 4]
        result = friedman_test([c1, c2, c3])
        assert "chi_square" in result
        assert "p_value" in result

    def test_mcnemar(self):
        a = [True, False, True, False, True, False, True, False, True, True]
        b = [False, False, False, False, False, False, True, False, False, False]
        result = mcnemar_test(a, b)
        assert "p_value" in result
        assert result["rate_a"] > result["rate_b"]

    def test_bonferroni(self):
        p_values = [0.01, 0.03, 0.04, 0.001, 0.06, 0.02]
        corrected = bonferroni_correction(p_values)
        assert len(corrected) == 6
        # With 6 tests, alpha = 0.05/6 = 0.0083
        assert corrected[0]["significant"] is False  # 0.01 > 0.0083
        assert corrected[3]["significant"] is True  # 0.001 < 0.0083

    def test_bradley_terry(self):
        # Model 2 wins most
        prefs = [(2, 0), (2, 1), (2, 0), (1, 0), (2, 1), (2, 0)]
        result = bradley_terry(prefs, n_models=3)
        assert len(result["strengths"]) == 3
        assert result["strengths"][2] > result["strengths"][0]
