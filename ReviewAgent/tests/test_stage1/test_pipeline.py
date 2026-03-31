"""Tests for the Stage 1 pipeline orchestration."""

import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime
from src.common.schemas import ReviewObject, AspectSentiment, ExtractedEntities
from src.stage1.pipeline import Stage1Pipeline


class TestStage1Pipeline:
    """Test pipeline orchestration with mocked components."""

    @pytest.fixture
    def mock_classifier(self):
        clf = MagicMock()
        clf.predict.return_value = [
            (["bug_report"], {"bug_report": 0.95, "feature_request": 0.05, "performance": 0.02,
                              "usability": 0.01, "compatibility": 0.01, "praise": 0.01, "other": 0.01}),
            (["praise"], {"bug_report": 0.02, "feature_request": 0.01, "performance": 0.01,
                          "usability": 0.01, "compatibility": 0.01, "praise": 0.92, "other": 0.01}),
        ]
        clf.needs_hitl.side_effect = lambda confs: max(confs.values()) < 0.7
        return clf

    @pytest.fixture
    def mock_aspect_analyzer(self):
        analyzer = MagicMock()
        analyzer.analyze_batch = AsyncMock(return_value=[
            [AspectSentiment(aspect="login", sentiment="negative", intensity=0.9)],
            [AspectSentiment(aspect="general", sentiment="positive", intensity=0.8)],
        ])
        return analyzer

    @pytest.fixture
    def mock_entity_extractor(self):
        extractor = MagicMock()
        extractor.extract_batch = AsyncMock(return_value=[
            ExtractedEntities(devices=["Pixel 8"], os_versions=["Android 15"]),
            ExtractedEntities(),
        ])
        return extractor

    @pytest.fixture
    def pipeline(self, mock_classifier, mock_aspect_analyzer, mock_entity_extractor):
        return Stage1Pipeline(mock_classifier, mock_aspect_analyzer, mock_entity_extractor)

    @pytest.fixture
    def raw_reviews(self):
        return [
            {"text": "App crashes on login on my Pixel 8", "rating": 1, "app_id": "com.test"},
            {"text": "Love this app! Amazing!", "rating": 5, "app_id": "com.test"},
        ]

    def test_process_returns_review_objects(self, pipeline, raw_reviews):
        """process() should return list of ReviewObject."""
        results = asyncio.run(pipeline.process(raw_reviews))
        assert len(results) == 2
        assert all(isinstance(r, ReviewObject) for r in results)

    def test_review_object_has_required_fields(self, pipeline, raw_reviews):
        """Each ReviewObject should have all key fields populated."""
        results = asyncio.run(pipeline.process(raw_reviews))
        for r in results:
            assert r.review_id  # non-empty
            assert r.text  # non-empty
            assert r.labels  # at least one label
            assert r.label_confidences  # confidence dict
            assert isinstance(r.aspects, list)
            assert isinstance(r.entities, ExtractedEntities)
            assert isinstance(r.flagged_for_hitl, bool)

    def test_labels_from_classifier(self, pipeline, raw_reviews):
        """Labels should come from the classifier."""
        results = asyncio.run(pipeline.process(raw_reviews))
        assert results[0].labels == ["bug_report"]
        assert results[1].labels == ["praise"]

    def test_aspects_from_analyzer(self, pipeline, raw_reviews):
        """Aspects should come from the aspect analyzer."""
        results = asyncio.run(pipeline.process(raw_reviews))
        assert len(results[0].aspects) == 1
        assert results[0].aspects[0].aspect == "login"
        assert results[0].aspects[0].sentiment == "negative"

    def test_entities_from_extractor(self, pipeline, raw_reviews):
        """Entities should come from the entity extractor."""
        results = asyncio.run(pipeline.process(raw_reviews))
        assert "Pixel 8" in results[0].entities.devices
        assert "Android 15" in results[0].entities.os_versions
        assert results[1].entities.devices == []

    def test_hitl_flagging(self, pipeline, raw_reviews):
        """Reviews with low confidence should be flagged for HITL."""
        results = asyncio.run(pipeline.process(raw_reviews))
        # Both have high confidence (0.95 and 0.92) so neither should be flagged
        assert results[0].flagged_for_hitl is False
        assert results[1].flagged_for_hitl is False

    def test_process_with_hitl_applies_corrections(self, pipeline, raw_reviews, mock_classifier):
        """process_with_hitl should apply corrections from callback."""
        # Make one review flagged
        mock_classifier.needs_hitl.side_effect = lambda confs: confs.get("bug_report", 0) > 0.5

        def hitl_callback(review):
            return ["usability"]  # Expert corrects to usability

        results = asyncio.run(pipeline.process_with_hitl(raw_reviews, hitl_callback))
        # The first review (bug_report 0.95) gets flagged and corrected
        assert results[0].labels == ["usability"]
        assert results[0].flagged_for_hitl is False

    def test_process_preserves_rating(self, pipeline, raw_reviews):
        """Original rating should be preserved."""
        results = asyncio.run(pipeline.process(raw_reviews))
        assert results[0].rating == 1
        assert results[1].rating == 5

    def test_process_preserves_app_id(self, pipeline, raw_reviews):
        """Original app_id should be preserved."""
        results = asyncio.run(pipeline.process(raw_reviews))
        assert results[0].app_id == "com.test"

    def test_process_empty_list(self, pipeline):
        """Processing empty list should return empty list."""
        results = asyncio.run(pipeline.process([]))
        assert results == []
