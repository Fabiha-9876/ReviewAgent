"""Tests for hierarchical clustering."""

import pytest
from datetime import datetime
from src.common.schemas import ReviewObject, AspectSentiment, ExtractedEntities
from src.stage2.kg_builder import ReviewKnowledgeGraph
from src.stage2.clustering import HierarchicalClusterer


@pytest.fixture
def reviews_with_two_aspects():
    """Create reviews that should cluster into 2 aspect groups."""
    reviews = []
    # 10 login-related reviews
    for i in range(10):
        reviews.append(ReviewObject(
            review_id=f"login_{i}",
            text=f"App crashes on login attempt {i}. Very frustrating!",
            rating=1, app_id="test", timestamp=datetime(2026, 3, 15),
            labels=["bug_report"],
            aspects=[AspectSentiment(aspect="login", sentiment="negative", intensity=0.9)],
            entities=ExtractedEntities(devices=["Pixel 8"]),
        ))
    # 10 battery-related reviews
    for i in range(10):
        reviews.append(ReviewObject(
            review_id=f"battery_{i}",
            text=f"Battery drain is terrible, phone dies in {i+1} hours",
            rating=2, app_id="test", timestamp=datetime(2026, 3, 16),
            labels=["performance"],
            aspects=[AspectSentiment(aspect="battery", sentiment="negative", intensity=0.8)],
            entities=ExtractedEntities(),
        ))
    return reviews


@pytest.fixture
def kg_with_reviews(reviews_with_two_aspects):
    kg = ReviewKnowledgeGraph()
    kg.add_reviews(reviews_with_two_aspects)
    return kg


class TestHierarchicalClusterer:

    def test_cluster_returns_dict(self, kg_with_reviews, reviews_with_two_aspects):
        """cluster() should return a dict mapping aspect_id to sub-clusters."""
        clusterer = HierarchicalClusterer(min_cluster_size=3, min_samples=2)
        result = clusterer.cluster(kg_with_reviews, reviews_with_two_aspects)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_cluster_groups_by_aspect(self, kg_with_reviews, reviews_with_two_aspects):
        """Should have separate groups for login and battery."""
        clusterer = HierarchicalClusterer(min_cluster_size=3, min_samples=2)
        result = clusterer.cluster(kg_with_reviews, reviews_with_two_aspects)
        assert "aspect:login" in result
        assert "aspect:battery" in result

    def test_cluster_contains_review_ids(self, kg_with_reviews, reviews_with_two_aspects):
        """Each sub-cluster should contain valid review IDs."""
        clusterer = HierarchicalClusterer(min_cluster_size=3, min_samples=2)
        result = clusterer.cluster(kg_with_reviews, reviews_with_two_aspects)

        all_review_ids = {r.review_id for r in reviews_with_two_aspects}
        for aspect_id, sub_clusters in result.items():
            for label, review_ids in sub_clusters:
                for rid in review_ids:
                    assert rid in all_review_ids

    def test_login_reviews_in_login_cluster(self, kg_with_reviews, reviews_with_two_aspects):
        """Login reviews should be in the login aspect group."""
        clusterer = HierarchicalClusterer(min_cluster_size=3, min_samples=2)
        result = clusterer.cluster(kg_with_reviews, reviews_with_two_aspects)

        login_rids = set()
        for label, rids in result.get("aspect:login", []):
            login_rids.update(rids)

        for i in range(10):
            assert f"login_{i}" in login_rids

    def test_small_group_not_subclustered(self):
        """Groups smaller than min_cluster_size should stay as one cluster."""
        reviews = [
            ReviewObject(
                review_id=f"tiny_{i}", text=f"Small issue {i}",
                rating=3, app_id="test", timestamp=datetime(2026, 3, 15),
                labels=["other"],
                aspects=[AspectSentiment(aspect="misc", sentiment="neutral", intensity=0.5)],
                entities=ExtractedEntities(),
            )
            for i in range(3)  # Only 3 reviews — below min_cluster_size of 5
        ]
        kg = ReviewKnowledgeGraph()
        kg.add_reviews(reviews)

        clusterer = HierarchicalClusterer(min_cluster_size=5, min_samples=3)
        result = clusterer.cluster(kg, reviews)

        # Should have one group with one sub-cluster containing all 3
        assert "aspect:misc" in result
        assert len(result["aspect:misc"]) == 1
        assert len(result["aspect:misc"][0][1]) == 3
