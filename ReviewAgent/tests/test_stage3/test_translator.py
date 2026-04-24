"""Tests for ReviewToIssueTranslator — prompt building and response parsing."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.common.schemas import IssueCluster, IssueSpec, ExtractedEntities
from src.stage3.translator import ReviewToIssueTranslator
from src.stage3.taxonomy import IssueTaxonomy


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate = AsyncMock(return_value="**Title:** App crashes on login\n\n**Description:** Multiple users report crashes.\n\n**Steps to Reproduce:**\n1. Open app\n2. Tap login\n3. App crashes\n\n**Expected Behavior:**\nUser logs in successfully.\n\n**Actual Behavior:**\nApp crashes with no error message.\n\n**Severity:** P1\n\n**Affected Component:** authentication_service")
    return llm


@pytest.fixture
def bug_cluster():
    return IssueCluster(
        cluster_id="CLU-001",
        issue_type="bug_report",
        aspect="login",
        sub_category="crash",
        review_ids=["r1", "r2", "r3"],
        review_count=3,
        representative_reviews=[
            "App crashes every time I try to login",
            "Login screen freezes then crashes",
            "Cannot log in, app closes immediately",
        ],
        entities=ExtractedEntities(
            devices=["Samsung Galaxy S24", "Pixel 8"],
            os_versions=["Android 14"],
            app_versions=["v3.2"],
        ),
        sentiment_distribution={"negative": 0.95, "neutral": 0.05},
        temporal_pattern="increasing after v3.2 release",
        priority_score=0.85,
    )


@pytest.fixture
def feature_cluster():
    return IssueCluster(
        cluster_id="CLU-002",
        issue_type="feature_request",
        aspect="dark_mode",
        sub_category="theme",
        review_ids=["r4", "r5"],
        review_count=2,
        representative_reviews=[
            "Please add dark mode",
            "Would love a dark theme option",
        ],
        entities=ExtractedEntities(),
        sentiment_distribution={"negative": 0.6, "neutral": 0.4},
        priority_score=0.5,
    )


@pytest.fixture
def translator(mock_llm):
    return ReviewToIssueTranslator(mock_llm, IssueTaxonomy())


# ============================================================
# Prompt Building Tests
# ============================================================

class TestBuildPrompt:
    """Tests for _build_prompt construction."""

    def test_prompt_contains_cluster_id(self, translator, bug_cluster):
        prompt = translator._build_prompt(bug_cluster, None)
        assert "CLU-001" in prompt

    def test_prompt_contains_issue_type(self, translator, bug_cluster):
        prompt = translator._build_prompt(bug_cluster, None)
        assert "bug_report" in prompt

    def test_prompt_contains_representative_reviews(self, translator, bug_cluster):
        prompt = translator._build_prompt(bug_cluster, None)
        assert "App crashes every time I try to login" in prompt
        assert "Login screen freezes then crashes" in prompt

    def test_prompt_contains_entities(self, translator, bug_cluster):
        prompt = translator._build_prompt(bug_cluster, None)
        assert "Samsung Galaxy S24" in prompt
        assert "Android 14" in prompt
        assert "v3.2" in prompt

    def test_prompt_contains_temporal_pattern(self, translator, bug_cluster):
        prompt = translator._build_prompt(bug_cluster, None)
        assert "increasing after v3.2 release" in prompt

    def test_prompt_contains_sentiment(self, translator, bug_cluster):
        prompt = translator._build_prompt(bug_cluster, None)
        assert "negative" in prompt

    def test_prompt_contains_priority(self, translator, bug_cluster):
        prompt = translator._build_prompt(bug_cluster, None)
        assert "0.85" in prompt

    def test_prompt_includes_kg_context(self, translator, bug_cluster):
        kg = {"related": ["CLU-003"], "edge": "similar_aspect"}
        prompt = translator._build_prompt(bug_cluster, kg)
        assert "Knowledge Graph Context" in prompt
        assert "CLU-003" in prompt

    def test_prompt_omits_kg_when_none(self, translator, bug_cluster):
        prompt = translator._build_prompt(bug_cluster, None)
        assert "Knowledge Graph Context" not in prompt

    def test_prompt_omits_empty_entity_fields(self, translator, feature_cluster):
        prompt = translator._build_prompt(feature_cluster, None)
        assert "Devices:" not in prompt
        assert "OS Versions:" not in prompt


# ============================================================
# Translation Tests
# ============================================================

class TestTranslate:
    """Tests for translate() — LLM call and parsing."""

    def test_translate_returns_issue_spec(self, translator, bug_cluster):
        spec = asyncio.run(translator.translate(bug_cluster))
        assert isinstance(spec, IssueSpec)

    def test_translate_parses_title(self, translator, bug_cluster):
        spec = asyncio.run(translator.translate(bug_cluster))
        assert "App crashes on login" in spec.title

    def test_translate_parses_description(self, translator, bug_cluster):
        spec = asyncio.run(translator.translate(bug_cluster))
        assert len(spec.description) > 0

    def test_translate_parses_steps(self, translator, bug_cluster):
        spec = asyncio.run(translator.translate(bug_cluster))
        assert spec.steps_to_reproduce is not None
        assert len(spec.steps_to_reproduce) == 3

    def test_translate_parses_expected_behavior(self, translator, bug_cluster):
        spec = asyncio.run(translator.translate(bug_cluster))
        assert spec.expected_behavior is not None
        assert "logs in" in spec.expected_behavior.lower()

    def test_translate_parses_actual_behavior(self, translator, bug_cluster):
        spec = asyncio.run(translator.translate(bug_cluster))
        assert spec.actual_behavior is not None
        assert "crash" in spec.actual_behavior.lower()

    def test_translate_parses_severity(self, translator, bug_cluster):
        spec = asyncio.run(translator.translate(bug_cluster))
        assert spec.severity == "P1"

    def test_translate_parses_component(self, translator, bug_cluster):
        spec = asyncio.run(translator.translate(bug_cluster))
        assert spec.affected_component == "authentication_service"

    def test_translate_sets_cluster_id(self, translator, bug_cluster):
        spec = asyncio.run(translator.translate(bug_cluster))
        assert spec.cluster_id == "CLU-001"

    def test_translate_sets_issue_type(self, translator, bug_cluster):
        spec = asyncio.run(translator.translate(bug_cluster))
        assert spec.issue_type == "bug_report"

    def test_translate_generates_issue_id(self, translator, bug_cluster):
        spec = asyncio.run(translator.translate(bug_cluster))
        assert spec.issue_id.startswith("ISS-")

    def test_translate_preserves_priority_score(self, translator, bug_cluster):
        spec = asyncio.run(translator.translate(bug_cluster))
        assert spec.priority_score == 0.85

    def test_translate_preserves_environment(self, translator, bug_cluster):
        spec = asyncio.run(translator.translate(bug_cluster))
        assert "Samsung Galaxy S24" in spec.environment.devices

    def test_translate_uses_taxonomy_template(self, translator, bug_cluster, mock_llm):
        asyncio.run(translator.translate(bug_cluster, use_taxonomy=True))
        call_args = mock_llm.generate.call_args
        system_prompt = call_args.kwargs.get("system_prompt", call_args[1].get("system_prompt", ""))
        assert "Zimmermann" in system_prompt

    def test_translate_without_taxonomy(self, translator, bug_cluster, mock_llm):
        asyncio.run(translator.translate(bug_cluster, use_taxonomy=False))
        call_args = mock_llm.generate.call_args
        system_prompt = call_args.kwargs.get("system_prompt", call_args[1].get("system_prompt", ""))
        assert "Zimmermann" not in system_prompt

    def test_translate_calls_llm_with_correct_params(self, translator, bug_cluster, mock_llm):
        asyncio.run(translator.translate(bug_cluster))
        mock_llm.generate.assert_called_once()
        call_kwargs = mock_llm.generate.call_args.kwargs
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 2048


# ============================================================
# Parse Response Tests
# ============================================================

class TestParseResponse:
    """Tests for _parse_response with various LLM output formats."""

    def setup_method(self):
        self.llm = MagicMock()
        self.translator = ReviewToIssueTranslator(self.llm)
        self.cluster = IssueCluster(
            cluster_id="CLU-999",
            issue_type="bug_report",
            aspect="payments",
            sub_category="timeout",
            review_ids=["r1"],
            review_count=1,
            representative_reviews=["Payment times out"],
            entities=ExtractedEntities(),
            priority_score=0.7,
        )

    def test_fallback_title_when_missing(self):
        spec = self.translator._parse_response("Just some text without a title.", self.cluster)
        assert spec.title == "payments — timeout"

    def test_fallback_description_when_missing(self):
        spec = self.translator._parse_response("", self.cluster)
        assert spec.description == "See representative reviews."

    def test_default_severity_when_missing(self):
        spec = self.translator._parse_response("No severity info here.", self.cluster)
        assert spec.severity == "P2"

    def test_parses_feature_request_fields(self):
        self.cluster.issue_type = "feature_request"
        raw = (
            "**Title:** Add dark mode\n"
            "**User Story:**\n"
            "As a user, I want dark mode so that I can reduce eye strain.\n"
            "**Acceptance Criteria:**\n"
            "1. Toggle in settings\n"
            "2. Persists across sessions\n"
            "**Severity:** P2\n"
        )
        spec = self.translator._parse_response(raw, self.cluster)
        assert spec.user_story is not None
        assert "dark mode" in spec.user_story
        assert spec.acceptance_criteria is not None
        assert len(spec.acceptance_criteria) == 2

    def test_parses_performance_nfr_category(self):
        self.cluster.issue_type = "performance"
        raw = (
            "**Title:** Slow startup\n"
            "**NFR Category:** startup_time\n"
            "**Description:** App takes 10+ seconds to load.\n"
            "**Severity:** P1\n"
        )
        spec = self.translator._parse_response(raw, self.cluster)
        assert spec.nfr_category == "startup_time"

    def test_parses_usability_nielsen_heuristic(self):
        self.cluster.issue_type = "usability"
        raw = (
            "**Title:** Confusing navigation\n"
            "**Nielsen Heuristic:** 3 - User control and freedom\n"
            "**Description:** Users cannot go back from settings.\n"
            "**Severity:** P2\n"
        )
        spec = self.translator._parse_response(raw, self.cluster)
        assert spec.nielsen_heuristic is not None
        assert "User control" in spec.nielsen_heuristic


# ============================================================
# Batch Translation Tests
# ============================================================

class TestTranslateBatch:
    """Tests for translate_batch — parallel translation."""

    def test_batch_returns_correct_count(self, translator, bug_cluster, feature_cluster):
        specs = asyncio.run(translator.translate_batch([bug_cluster, feature_cluster]))
        assert len(specs) == 2

    def test_batch_preserves_cluster_ids(self, translator, bug_cluster, feature_cluster):
        specs = asyncio.run(translator.translate_batch([bug_cluster, feature_cluster]))
        cluster_ids = {s.cluster_id for s in specs}
        assert "CLU-001" in cluster_ids
        assert "CLU-002" in cluster_ids

    def test_batch_empty_input(self, translator):
        specs = asyncio.run(translator.translate_batch([]))
        assert specs == []

    def test_batch_uses_kg_context_map(self, translator, bug_cluster, mock_llm):
        kg_map = {"CLU-001": {"related": ["CLU-005"]}}
        asyncio.run(translator.translate_batch([bug_cluster], kg_context_map=kg_map))
        call_kwargs = mock_llm.generate.call_args.kwargs
        assert "CLU-005" in call_kwargs["user_prompt"]
