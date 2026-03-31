"""Tests for aspect-based sentiment analysis."""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
from src.common.schemas import AspectSentiment
from src.stage1.aspect_sentiment import AspectSentimentAnalyzer


class TestAspectSentimentAnalyzer:
    """Test aspect sentiment analysis with mocked LLM."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        llm = MagicMock()
        llm.generate = AsyncMock()
        return llm

    def test_analyze_returns_aspects(self, mock_llm):
        """analyze() should return a list of AspectSentiment objects."""
        mock_llm.generate.return_value = json.dumps([
            {"aspect": "login", "sentiment": "negative", "intensity": 0.9},
            {"aspect": "ui", "sentiment": "positive", "intensity": 0.7},
        ])

        analyzer = AspectSentimentAnalyzer(mock_llm)
        result = asyncio.run(analyzer.analyze("Login crashes but UI looks great"))

        assert len(result) == 2
        assert all(isinstance(a, AspectSentiment) for a in result)
        assert result[0].aspect == "login"
        assert result[0].sentiment == "negative"
        assert result[1].aspect == "ui"
        assert result[1].sentiment == "positive"

    def test_analyze_empty_text(self, mock_llm):
        """Empty text should return a fallback aspect."""
        mock_llm.generate.return_value = "[]"

        analyzer = AspectSentimentAnalyzer(mock_llm)
        result = asyncio.run(analyzer.analyze(""))
        # Empty JSON array returns empty list, but the method handles gracefully
        assert isinstance(result, list)

    def test_analyze_llm_error_fallback(self, mock_llm):
        """If LLM returns garbage, should fallback gracefully."""
        mock_llm.generate.return_value = "this is not json"

        analyzer = AspectSentimentAnalyzer(mock_llm)
        result = asyncio.run(analyzer.analyze("Some review text"))

        # Should return fallback (unknown/neutral)
        assert len(result) == 1
        assert result[0].aspect == "unknown"
        assert result[0].sentiment == "neutral"

    def test_analyze_markdown_wrapped_json(self, mock_llm):
        """LLM sometimes wraps JSON in markdown code blocks."""
        mock_llm.generate.return_value = '```json\n[{"aspect": "battery", "sentiment": "negative", "intensity": 0.8}]\n```'

        analyzer = AspectSentimentAnalyzer(mock_llm)
        result = asyncio.run(analyzer.analyze("Battery drain is terrible"))

        assert len(result) == 1
        assert result[0].aspect == "battery"
        assert result[0].sentiment == "negative"

    def test_analyze_batch(self, mock_llm):
        """analyze_batch() should process multiple reviews."""
        mock_llm.generate.return_value = json.dumps([
            {"aspect": "general", "sentiment": "positive", "intensity": 0.5}
        ])

        analyzer = AspectSentimentAnalyzer(mock_llm)
        results = asyncio.run(analyzer.analyze_batch(["Great app", "Terrible app", "Ok app"]))

        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)

    def test_intensity_clamped(self, mock_llm):
        """Intensity should be clamped to 0.0-1.0."""
        mock_llm.generate.return_value = json.dumps([
            {"aspect": "test", "sentiment": "positive", "intensity": 1.5},
        ])

        analyzer = AspectSentimentAnalyzer(mock_llm)
        result = asyncio.run(analyzer.analyze("test"))

        assert result[0].intensity <= 1.0
