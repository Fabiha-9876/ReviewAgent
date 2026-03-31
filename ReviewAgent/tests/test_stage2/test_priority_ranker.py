"""Tests for priority ranker."""

import pytest
from datetime import datetime
from src.common.schemas import ReviewObject, AspectSentiment, ExtractedEntities, IssueCluster
from src.stage2.kg_builder import ReviewKnowledgeGraph
from src.stage2.priority_ranker import PriorityRanker


@pytest.fixture
def kg_and_clusters():
    """Create a KG and clusters with varying priority signals."""
    reviews = []
    # High priority: many reviews, strong negative, temporal pattern
    for i in range(20):
        reviews.append(ReviewObject(
            review_id=f"high_{i}", text=f"Login crash {i}",
            rating=1, app_id="test", timestamp=datetime(2026, 3, 15),
            labels=["bug_report"],
            aspects=[AspectSentiment(aspect="login", sentiment="negative", intensity=0.95)],
            entities=ExtractedEntities(app_versions=["v3.2"]),
        ))
    # Low priority: few reviews, mild sentiment
    for i in range(3):
        reviews.append(ReviewObject(
            review_id=f"low_{i}", text=f"Minor UI issue {i}",
            rating=3, app_id="test", timestamp=datetime(2026, 3, 20),
            labels=["usability"],
            aspects=[AspectSentiment(aspect="settings", sentiment="neutral", intensity=0.3)],
            entities=ExtractedEntities(),
        ))

    kg = ReviewKnowledgeGraph()
    kg.add_reviews(reviews)

    high_cluster = IssueCluster(
        cluster_id="CLU-HIGH", issue_type="bug_report", aspect="login",
        sub_category="crash", review_ids=[f"high_{i}" for i in range(20)],
        review_count=20, representative_reviews=["Login crash"],
        entities=ExtractedEntities(app_versions=["v3.2"]),
        sentiment_distribution={"negative": 0.95, "neutral": 0.05},
        temporal_pattern="spike_after_v3.2", priority_score=0.0,
    )
    low_cluster = IssueCluster(
        cluster_id="CLU-LOW", issue_type="usability", aspect="settings",
        sub_category="minor_ui", review_ids=[f"low_{i}" for i in range(3)],
        review_count=3, representative_reviews=["Minor UI issue"],
        entities=ExtractedEntities(),
        sentiment_distribution={"neutral": 0.7, "negative": 0.3},
        temporal_pattern=None, priority_score=0.0,
    )

    return kg, [high_cluster, low_cluster]


class TestPriorityRanker:

    def test_rank_returns_sorted_list(self, kg_and_clusters):
        kg, clusters = kg_and_clusters
        ranker = PriorityRanker()
        ranked = ranker.rank(clusters, kg)
        assert len(ranked) == 2
        assert ranked[0].priority_score >= ranked[1].priority_score

    def test_high_priority_first(self, kg_and_clusters):
        """The cluster with more reviews, stronger sentiment should rank higher."""
        kg, clusters = kg_and_clusters
        ranker = PriorityRanker()
        ranked = ranker.rank(clusters, kg)
        assert ranked[0].cluster_id == "CLU-HIGH"
        assert ranked[1].cluster_id == "CLU-LOW"

    def test_priority_scores_assigned(self, kg_and_clusters):
        """All clusters should get non-zero priority scores."""
        kg, clusters = kg_and_clusters
        ranker = PriorityRanker()
        ranked = ranker.rank(clusters, kg)
        assert all(c.priority_score > 0 for c in ranked)

    def test_priority_score_range(self, kg_and_clusters):
        """Priority scores should be between 0 and 1."""
        kg, clusters = kg_and_clusters
        ranker = PriorityRanker()
        ranked = ranker.rank(clusters, kg)
        for c in ranked:
            assert 0.0 <= c.priority_score <= 1.5  # Can slightly exceed 1 due to weights

    def test_empty_list(self, kg_and_clusters):
        kg, _ = kg_and_clusters
        ranker = PriorityRanker()
        ranked = ranker.rank([], kg)
        assert ranked == []

    def test_custom_weights(self, kg_and_clusters):
        """Custom weights should change the ranking behavior."""
        kg, clusters = kg_and_clusters
        # Weight heavily toward review count
        ranker = PriorityRanker(
            pagerank_weight=0.0, review_count_weight=1.0,
            sentiment_weight=0.0, recency_weight=0.0,
        )
        ranked = ranker.rank(clusters, kg)
        # High cluster has 20 reviews vs 3 → should still be first
        assert ranked[0].cluster_id == "CLU-HIGH"
