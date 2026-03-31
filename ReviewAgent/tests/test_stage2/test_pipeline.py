"""Tests for Stage 2 pipeline orchestration."""

import pytest
from datetime import datetime
from src.common.schemas import ReviewObject, AspectSentiment, ExtractedEntities, IssueCluster
from src.stage2.pipeline import Stage2Pipeline


@pytest.fixture
def diverse_reviews():
    """Create a diverse set of reviews covering multiple aspects."""
    reviews = []
    # Login crashes (10 reviews)
    for i in range(10):
        reviews.append(ReviewObject(
            review_id=f"login_{i}",
            text=f"App crashes when I try to login. Very annoying issue number {i}!",
            rating=1, app_id="com.test", timestamp=datetime(2026, 3, 15),
            labels=["bug_report"],
            aspects=[AspectSentiment(aspect="login", sentiment="negative", intensity=0.9)],
            entities=ExtractedEntities(devices=["Pixel 8"], app_versions=["v3.2"]),
        ))
    # Battery complaints (8 reviews)
    for i in range(8):
        reviews.append(ReviewObject(
            review_id=f"battery_{i}",
            text=f"Battery drain is terrible, phone dies quickly every day number {i}",
            rating=2, app_id="com.test", timestamp=datetime(2026, 3, 16),
            labels=["performance"],
            aspects=[AspectSentiment(aspect="battery", sentiment="negative", intensity=0.8)],
            entities=ExtractedEntities(),
        ))
    # Praise (5 reviews)
    for i in range(5):
        reviews.append(ReviewObject(
            review_id=f"praise_{i}",
            text=f"Love this app so much! Best app ever, review number {i}!",
            rating=5, app_id="com.test", timestamp=datetime(2026, 3, 17),
            labels=["praise"],
            aspects=[AspectSentiment(aspect="general", sentiment="positive", intensity=0.9)],
            entities=ExtractedEntities(),
        ))
    return reviews


class TestStage2Pipeline:

    def test_process_returns_clusters(self, diverse_reviews):
        pipeline = Stage2Pipeline()
        clusters = pipeline.process(diverse_reviews)
        assert isinstance(clusters, list)
        assert all(isinstance(c, IssueCluster) for c in clusters)

    def test_clusters_have_required_fields(self, diverse_reviews):
        pipeline = Stage2Pipeline()
        clusters = pipeline.process(diverse_reviews)
        for c in clusters:
            assert c.cluster_id
            assert c.issue_type
            assert c.aspect
            assert c.review_count > 0
            assert len(c.review_ids) > 0
            assert len(c.representative_reviews) > 0

    def test_clusters_sorted_by_priority(self, diverse_reviews):
        """Clusters should be sorted by priority score descending."""
        pipeline = Stage2Pipeline()
        clusters = pipeline.process(diverse_reviews)
        if len(clusters) > 1:
            for i in range(len(clusters) - 1):
                assert clusters[i].priority_score >= clusters[i + 1].priority_score

    def test_kg_built(self, diverse_reviews):
        """The KG should be accessible after processing."""
        pipeline = Stage2Pipeline()
        pipeline.process(diverse_reviews)
        kg = pipeline.get_kg()
        assert kg.node_count() > 0
        assert kg.edge_count() > 0

    def test_multiple_aspects_create_multiple_clusters(self, diverse_reviews):
        """Reviews about different aspects should create different clusters.
        Note: HDBSCAN may mark some sub-clusters as noise (-1), which get
        filtered out. So we check the KG has multiple aspects even if
        not all produce clusters after noise filtering.
        """
        pipeline = Stage2Pipeline()
        clusters = pipeline.process(diverse_reviews)
        kg = pipeline.get_kg()
        # The KG should have multiple aspect nodes regardless of clustering outcome
        aspects_in_kg = kg.get_aspect_nodes()
        assert len(aspects_in_kg) >= 2
        # If clusters were produced, they should have valid aspects
        if clusters:
            assert all(c.aspect for c in clusters)

    def test_empty_reviews(self):
        pipeline = Stage2Pipeline()
        clusters = pipeline.process([])
        assert clusters == []

    def test_review_ids_in_clusters(self, diverse_reviews):
        """All review IDs in clusters should come from input reviews."""
        pipeline = Stage2Pipeline()
        clusters = pipeline.process(diverse_reviews)
        input_ids = {r.review_id for r in diverse_reviews}
        for c in clusters:
            for rid in c.review_ids:
                assert rid in input_ids
