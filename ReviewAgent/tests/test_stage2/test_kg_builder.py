"""Tests for Knowledge Graph builder."""

import pytest
from datetime import datetime
from src.common.schemas import ReviewObject, AspectSentiment, ExtractedEntities
from src.stage2.kg_builder import ReviewKnowledgeGraph


@pytest.fixture
def sample_reviews():
    return [
        ReviewObject(
            review_id="r1",
            text="App crashes on login",
            rating=1,
            app_id="test",
            timestamp=datetime(2026, 3, 15),
            labels=["bug_report"],
            aspects=[AspectSentiment(aspect="login", sentiment="negative", intensity=0.9)],
            entities=ExtractedEntities(devices=["Pixel 8"], os_versions=["Android 15"]),
        ),
        ReviewObject(
            review_id="r2",
            text="Login broken on Samsung",
            rating=1,
            app_id="test",
            timestamp=datetime(2026, 3, 16),
            labels=["bug_report"],
            aspects=[AspectSentiment(aspect="login", sentiment="negative", intensity=0.8)],
            entities=ExtractedEntities(devices=["Samsung Galaxy S24"]),
        ),
        ReviewObject(
            review_id="r3",
            text="Battery drain is terrible",
            rating=2,
            app_id="test",
            timestamp=datetime(2026, 3, 17),
            labels=["performance"],
            aspects=[AspectSentiment(aspect="battery", sentiment="negative", intensity=0.7)],
            entities=ExtractedEntities(),
        ),
    ]


class TestReviewKnowledgeGraph:
    def test_add_reviews(self, sample_reviews):
        kg = ReviewKnowledgeGraph()
        kg.add_reviews(sample_reviews)
        assert kg.node_count() > 0
        assert kg.edge_count() > 0

    def test_aspect_nodes(self, sample_reviews):
        kg = ReviewKnowledgeGraph()
        kg.add_reviews(sample_reviews)
        aspects = kg.get_aspect_nodes()
        assert "aspect:login" in aspects
        assert "aspect:battery" in aspects

    def test_reviews_for_aspect(self, sample_reviews):
        kg = ReviewKnowledgeGraph()
        kg.add_reviews(sample_reviews)
        login_reviews = kg.get_reviews_for_aspect("aspect:login")
        assert "r1" in login_reviews
        assert "r2" in login_reviews
        assert "r3" not in login_reviews

    def test_pagerank(self, sample_reviews):
        kg = ReviewKnowledgeGraph()
        kg.add_reviews(sample_reviews)
        pr = kg.compute_pagerank()
        assert len(pr) > 0
        assert all(v >= 0 for v in pr.values())

    def test_export_load(self, sample_reviews, tmp_path):
        kg = ReviewKnowledgeGraph()
        kg.add_reviews(sample_reviews)
        path = str(tmp_path / "kg.json")
        kg.export(path)

        kg2 = ReviewKnowledgeGraph()
        kg2.load(path)
        assert kg2.node_count() == kg.node_count()
