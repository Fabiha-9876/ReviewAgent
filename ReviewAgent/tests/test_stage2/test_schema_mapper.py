"""Tests for schema mapper."""

import pytest
from datetime import datetime
from src.common.schemas import ReviewObject, AspectSentiment, ExtractedEntities, IssueCluster
from src.stage2.kg_builder import ReviewKnowledgeGraph
from src.stage2.schema_mapper import SchemaMapper


@pytest.fixture
def bug_reviews():
    """Create reviews that should map to a bug_report cluster."""
    return [
        ReviewObject(
            review_id=f"r{i}", text=f"App crashes on login since v3.2 update {i}",
            rating=1, app_id="com.test", timestamp=datetime(2026, 3, 15 + i),
            labels=["bug_report"],
            aspects=[AspectSentiment(aspect="login", sentiment="negative", intensity=0.9)],
            entities=ExtractedEntities(
                devices=["Samsung Galaxy S24"] if i % 2 == 0 else ["Pixel 8"],
                app_versions=["v3.2"],
            ),
        )
        for i in range(5)
    ]


@pytest.fixture
def kg_with_bugs(bug_reviews):
    kg = ReviewKnowledgeGraph()
    kg.add_reviews(bug_reviews)
    return kg


class TestSchemaMapper:

    def test_map_cluster_returns_issue_cluster(self, bug_reviews, kg_with_bugs):
        mapper = SchemaMapper()
        result = mapper.map_cluster(
            aspect_id="aspect:login", sub_label=0,
            review_ids=[r.review_id for r in bug_reviews],
            reviews=bug_reviews, kg=kg_with_bugs,
        )
        assert isinstance(result, IssueCluster)

    def test_cluster_id_generated(self, bug_reviews, kg_with_bugs):
        mapper = SchemaMapper()
        result = mapper.map_cluster("aspect:login", 0,
            [r.review_id for r in bug_reviews], bug_reviews, kg_with_bugs)
        assert result.cluster_id.startswith("CLU-")
        assert len(result.cluster_id) > 4

    def test_issue_type_majority_vote(self, bug_reviews, kg_with_bugs):
        """All reviews are bug_report → issue_type should be bug_report."""
        mapper = SchemaMapper()
        result = mapper.map_cluster("aspect:login", 0,
            [r.review_id for r in bug_reviews], bug_reviews, kg_with_bugs)
        assert result.issue_type == "bug_report"

    def test_aspect_extracted(self, bug_reviews, kg_with_bugs):
        mapper = SchemaMapper()
        result = mapper.map_cluster("aspect:login", 0,
            [r.review_id for r in bug_reviews], bug_reviews, kg_with_bugs)
        assert result.aspect == "login"

    def test_review_count(self, bug_reviews, kg_with_bugs):
        mapper = SchemaMapper()
        result = mapper.map_cluster("aspect:login", 0,
            [r.review_id for r in bug_reviews], bug_reviews, kg_with_bugs)
        assert result.review_count == 5

    def test_representative_reviews_selected(self, bug_reviews, kg_with_bugs):
        mapper = SchemaMapper()
        result = mapper.map_cluster("aspect:login", 0,
            [r.review_id for r in bug_reviews], bug_reviews, kg_with_bugs)
        assert len(result.representative_reviews) <= 3
        assert all(isinstance(r, str) for r in result.representative_reviews)

    def test_entities_merged(self, bug_reviews, kg_with_bugs):
        """Entities from all reviews should be merged and deduplicated."""
        mapper = SchemaMapper()
        result = mapper.map_cluster("aspect:login", 0,
            [r.review_id for r in bug_reviews], bug_reviews, kg_with_bugs)
        assert "Samsung Galaxy S24" in result.entities.devices
        assert "Pixel 8" in result.entities.devices
        assert "v3.2" in result.entities.app_versions

    def test_sentiment_distribution(self, bug_reviews, kg_with_bugs):
        """All reviews are negative about login → high negative ratio."""
        mapper = SchemaMapper()
        result = mapper.map_cluster("aspect:login", 0,
            [r.review_id for r in bug_reviews], bug_reviews, kg_with_bugs)
        assert result.sentiment_distribution.get("negative", 0) > 0.5

    def test_temporal_pattern_detected(self, bug_reviews, kg_with_bugs):
        """All reviews mention v3.2 → temporal pattern should detect spike."""
        mapper = SchemaMapper()
        result = mapper.map_cluster("aspect:login", 0,
            [r.review_id for r in bug_reviews], bug_reviews, kg_with_bugs)
        assert result.temporal_pattern is not None
        assert "v3.2" in result.temporal_pattern

    def test_mixed_labels_majority_vote(self, kg_with_bugs):
        """Mixed labels → majority should win."""
        mixed_reviews = [
            ReviewObject(review_id="m1", text="crash", rating=1, app_id="t",
                timestamp=datetime(2026, 3, 15), labels=["bug_report"],
                aspects=[AspectSentiment(aspect="app", sentiment="negative", intensity=0.8)],
                entities=ExtractedEntities()),
            ReviewObject(review_id="m2", text="crash too", rating=1, app_id="t",
                timestamp=datetime(2026, 3, 16), labels=["bug_report"],
                aspects=[AspectSentiment(aspect="app", sentiment="negative", intensity=0.8)],
                entities=ExtractedEntities()),
            ReviewObject(review_id="m3", text="slow", rating=2, app_id="t",
                timestamp=datetime(2026, 3, 17), labels=["performance"],
                aspects=[AspectSentiment(aspect="app", sentiment="negative", intensity=0.6)],
                entities=ExtractedEntities()),
        ]
        kg = ReviewKnowledgeGraph()
        kg.add_reviews(mixed_reviews)
        mapper = SchemaMapper()
        result = mapper.map_cluster("aspect:app", 0,
            ["m1", "m2", "m3"], mixed_reviews, kg)
        assert result.issue_type == "bug_report"  # 2 bug vs 1 performance
