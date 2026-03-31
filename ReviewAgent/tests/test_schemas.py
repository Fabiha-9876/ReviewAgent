"""Tests for common data schemas."""

import pytest
from datetime import datetime
from src.common.schemas import (
    ReviewObject,
    AspectSentiment,
    ExtractedEntities,
    IssueCluster,
    IssueSpec,
    GeneratedResponse,
    RubricScores,
    ComplianceFlags,
)


class TestExtractedEntities:
    def test_merge(self):
        e1 = ExtractedEntities(devices=["iPhone 15"], os_versions=["iOS 18"])
        e2 = ExtractedEntities(devices=["Pixel 8", "iPhone 15"], os_versions=["Android 15"])
        merged = e1.merge(e2)
        assert "iPhone 15" in merged.devices
        assert "Pixel 8" in merged.devices
        assert len(merged.devices) == 2  # deduplicated
        assert len(merged.os_versions) == 2

    def test_merge_empty(self):
        e1 = ExtractedEntities()
        e2 = ExtractedEntities(devices=["Pixel 8"])
        merged = e1.merge(e2)
        assert merged.devices == ["Pixel 8"]


class TestReviewObject:
    def test_create(self):
        review = ReviewObject(
            review_id="test-1",
            text="App crashes on login",
            rating=1,
            app_id="com.test",
            timestamp=datetime.now(),
            labels=["bug_report"],
            label_confidences={"bug_report": 0.95},
        )
        assert review.review_id == "test-1"
        assert review.flagged_for_hitl is False

    def test_rating_validation(self):
        with pytest.raises(Exception):
            ReviewObject(
                review_id="bad",
                text="test",
                rating=6,  # Invalid
                app_id="test",
                timestamp=datetime.now(),
            )


class TestRubricScores:
    def test_mean(self):
        scores = RubricScores(
            completeness=4, accuracy=3, actionability=5, specificity=4, clarity=4
        )
        assert scores.mean == 4.0

    def test_low_mean(self):
        scores = RubricScores(
            completeness=1, accuracy=1, actionability=1, specificity=1, clarity=1
        )
        assert scores.mean == 1.0


class TestComplianceFlags:
    def test_compliant(self):
        flags = ComplianceFlags()
        assert flags.is_compliant is True

    def test_non_compliant(self):
        flags = ComplianceFlags(no_false_promises=False)
        assert flags.is_compliant is False


class TestIssueSpec:
    def test_create_bug_report(self):
        spec = IssueSpec(
            issue_id="ISS-001",
            cluster_id="CLU-001",
            title="Login crash on Android",
            issue_type="bug_report",
            description="App crashes during login",
            steps_to_reproduce=["Open app", "Tap login", "App crashes"],
            expected_behavior="User logs in",
            actual_behavior="App crashes",
            severity="P0",
            affected_component="auth_service",
        )
        assert spec.severity == "P0"
        assert len(spec.steps_to_reproduce) == 3

    def test_create_feature_request(self):
        spec = IssueSpec(
            issue_id="ISS-002",
            cluster_id="CLU-002",
            title="Add dark mode",
            issue_type="feature_request",
            description="Users want dark mode",
            user_story="As a user, I want dark mode, so that I can use the app at night",
            acceptance_criteria=["Toggle in settings", "All screens support dark mode"],
            severity="P2",
        )
        assert spec.user_story is not None
        assert len(spec.acceptance_criteria) == 2


class TestIssueCluster:
    def test_create(self):
        cluster = IssueCluster(
            cluster_id="CLU-001",
            issue_type="bug_report",
            aspect="login",
            sub_category="crash_on_action",
            review_ids=["r1", "r2", "r3"],
            review_count=3,
            representative_reviews=["App crashes on login"],
            entities=ExtractedEntities(devices=["Pixel 8"]),
            sentiment_distribution={"negative": 0.9, "neutral": 0.1},
            priority_score=0.85,
        )
        assert cluster.review_count == 3
        assert cluster.priority_score == 0.85
