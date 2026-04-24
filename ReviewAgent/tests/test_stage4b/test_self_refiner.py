"""Tests for SelfRefiner — critique and refinement loop."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.common.schemas import GeneratedResponse, IssueSpec
from src.stage4b.self_refiner import SelfRefiner, CRITIQUE_PROMPT, REVISE_PROMPT


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_response():
    return GeneratedResponse(
        response_id="resp-001",
        issue_id="ISS-001",
        review_id="rev-001",
        text="We're sorry you're having trouble. Please try again later.",
        refinement_iterations=0,
    )


@pytest.fixture
def issue_spec():
    return IssueSpec(
        issue_id="ISS-001",
        cluster_id="CLU-001",
        title="Login crash on Android 14",
        issue_type="bug_report",
        description="App crashes on login",
    )


# ============================================================
# Critique Tests
# ============================================================

class TestCritique:

    def test_all_pass_returns_pass_dict(self):
        llm = MagicMock()
        llm.generate = AsyncMock(return_value='{"specificity": "pass", "compliance": "pass", "empathy": "pass"}')
        refiner = SelfRefiner(llm)

        result = asyncio.run(refiner._critique("Good response", None))
        assert result == {"specificity": "pass", "compliance": "pass", "empathy": "pass"}

    def test_critique_with_suggestions(self):
        llm = MagicMock()
        llm.generate = AsyncMock(return_value='{"specificity": "Be more specific about the crash", "compliance": "pass", "empathy": "pass"}')
        refiner = SelfRefiner(llm)

        result = asyncio.run(refiner._critique("Generic response", None))
        assert result["specificity"] != "pass"
        assert result["compliance"] == "pass"

    def test_critique_includes_issue_spec_in_context(self, issue_spec):
        llm = MagicMock()
        llm.generate = AsyncMock(return_value='{"specificity": "pass", "compliance": "pass", "empathy": "pass"}')
        refiner = SelfRefiner(llm)

        asyncio.run(refiner._critique("Some response", issue_spec))
        call_kwargs = llm.generate.call_args.kwargs
        assert "Login crash on Android 14" in call_kwargs["user_prompt"]

    def test_critique_handles_malformed_json(self):
        llm = MagicMock()
        llm.generate = AsyncMock(return_value="not valid json at all")
        refiner = SelfRefiner(llm)

        result = asyncio.run(refiner._critique("response", None))
        assert result == {"specificity": "pass", "compliance": "pass", "empathy": "pass"}

    def test_critique_handles_markdown_code_block(self):
        llm = MagicMock()
        llm.generate = AsyncMock(return_value='```json\n{"specificity": "pass", "compliance": "pass", "empathy": "pass"}\n```')
        refiner = SelfRefiner(llm)

        result = asyncio.run(refiner._critique("response", None))
        assert result == {"specificity": "pass", "compliance": "pass", "empathy": "pass"}

    def test_critique_uses_correct_prompt(self):
        llm = MagicMock()
        llm.generate = AsyncMock(return_value='{"specificity": "pass", "compliance": "pass", "empathy": "pass"}')
        refiner = SelfRefiner(llm)

        asyncio.run(refiner._critique("test", None))
        call_kwargs = llm.generate.call_args.kwargs
        assert call_kwargs["system_prompt"] == CRITIQUE_PROMPT
        assert call_kwargs["temperature"] == 0.1


# ============================================================
# Revise Tests
# ============================================================

class TestRevise:

    def test_revise_returns_string(self):
        llm = MagicMock()
        llm.generate = AsyncMock(return_value="Improved response text.")
        refiner = SelfRefiner(llm)

        result = asyncio.run(refiner._revise(
            "Old response",
            {"specificity": "Be more specific", "compliance": "pass", "empathy": "pass"}
        ))
        assert result == "Improved response text."

    def test_revise_only_includes_non_pass_feedback(self):
        llm = MagicMock()
        llm.generate = AsyncMock(return_value="Better response.")
        refiner = SelfRefiner(llm)

        asyncio.run(refiner._revise(
            "Old response",
            {"specificity": "Be specific", "compliance": "pass", "empathy": "Add empathy"}
        ))
        call_kwargs = llm.generate.call_args.kwargs
        assert "specificity" in call_kwargs["user_prompt"]
        assert "empathy" in call_kwargs["user_prompt"]
        # compliance is "pass", should not appear as critique
        assert "compliance: pass" not in call_kwargs["user_prompt"]


# ============================================================
# Refine Loop Tests
# ============================================================

class TestRefine:

    def test_stops_when_all_pass(self, sample_response):
        llm = MagicMock()
        llm.generate = AsyncMock(return_value='{"specificity": "pass", "compliance": "pass", "empathy": "pass"}')
        refiner = SelfRefiner(llm, max_iterations=3)

        result = asyncio.run(refiner.refine(sample_response))
        assert result.refinement_iterations == 1
        assert llm.generate.call_count == 1  # only critique, no revise

    def test_iterates_until_pass(self, sample_response):
        llm = MagicMock()
        call_count = [0]

        async def mock_generate(**kwargs):
            call_count[0] += 1
            if "quality reviewer" in kwargs.get("system_prompt", "").lower() or kwargs.get("system_prompt") == CRITIQUE_PROMPT:
                if call_count[0] <= 2:
                    return '{"specificity": "needs work", "compliance": "pass", "empathy": "pass"}'
                return '{"specificity": "pass", "compliance": "pass", "empathy": "pass"}'
            return "Revised response"

        llm.generate = AsyncMock(side_effect=mock_generate)
        refiner = SelfRefiner(llm, max_iterations=3)

        result = asyncio.run(refiner.refine(sample_response))
        assert result.refinement_iterations >= 2

    def test_respects_max_iterations(self, sample_response):
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=[
            '{"specificity": "fix it", "compliance": "pass", "empathy": "pass"}',
            "Revised v1",
            '{"specificity": "still bad", "compliance": "pass", "empathy": "pass"}',
            "Revised v2",
        ])
        refiner = SelfRefiner(llm, max_iterations=2)

        result = asyncio.run(refiner.refine(sample_response))
        assert result.refinement_iterations == 2

    def test_updates_response_text(self, sample_response):
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=[
            '{"specificity": "be specific", "compliance": "pass", "empathy": "pass"}',
            "Much better specific response",
            '{"specificity": "pass", "compliance": "pass", "empathy": "pass"}',
        ])
        refiner = SelfRefiner(llm, max_iterations=3)

        result = asyncio.run(refiner.refine(sample_response))
        assert result.text == "Much better specific response"

    def test_preserves_response_id(self, sample_response):
        llm = MagicMock()
        llm.generate = AsyncMock(return_value='{"specificity": "pass", "compliance": "pass", "empathy": "pass"}')
        refiner = SelfRefiner(llm, max_iterations=1)

        result = asyncio.run(refiner.refine(sample_response))
        assert result.response_id == "resp-001"
        assert result.review_id == "rev-001"
        assert result.issue_id == "ISS-001"

    def test_refine_with_issue_spec(self, sample_response, issue_spec):
        llm = MagicMock()
        llm.generate = AsyncMock(return_value='{"specificity": "pass", "compliance": "pass", "empathy": "pass"}')
        refiner = SelfRefiner(llm, max_iterations=1)

        result = asyncio.run(refiner.refine(sample_response, issue_spec))
        call_kwargs = llm.generate.call_args.kwargs
        assert "Login crash on Android 14" in call_kwargs["user_prompt"]

    def test_single_iteration_max(self, sample_response):
        llm = MagicMock()
        llm.generate = AsyncMock(return_value='{"specificity": "fix", "compliance": "pass", "empathy": "pass"}')
        refiner = SelfRefiner(llm, max_iterations=1)

        result = asyncio.run(refiner.refine(sample_response))
        assert result.refinement_iterations == 1
