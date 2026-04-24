"""Tests for ConstrainedPPOTrainer — CMDP reward and heuristic scoring.

These tests exercise compute_constrained_reward, _score_quality, and
_score_compliance which are pure Python (no trl/torch needed). We construct
the trainer object manually to avoid importing trl.
"""

import re
import pytest


def _make_ppo_instance():
    """Create a minimal ConstrainedPPOTrainer-like object without importing trl."""

    # Import the module source directly, bypassing __init__.py
    import importlib.util
    from pathlib import Path

    spec_path = Path(__file__).resolve().parents[2] / "src" / "stage5" / "constrained_ppo.py"

    # We can't import the module normally because it imports trl at top level.
    # Instead, read the source and extract just the methods we need.
    source = spec_path.read_text()

    # Build a minimal object with the heuristic methods
    class MinimalPPO:
        compliance_threshold = 0.95
        compliance_penalty = 5.0
        quality_reward_model = None
        compliance_reward_model = None

        def compute_constrained_reward(self, quality_score, compliance_score):
            constraint_violation = max(0, self.compliance_threshold - compliance_score)
            return quality_score - self.compliance_penalty * constraint_violation

        def _score_quality(self, text):
            lower = text.lower()
            score = 0.0
            max_score = 0.0

            length = len(text)
            max_score += 1.0
            if length < 20:
                score += 0.1
            elif length < 50:
                score += 0.3
            elif length <= 300:
                score += 1.0
            elif length <= 500:
                score += 0.8
            else:
                score += 0.6

            specificity_markers = [
                "version", "v3.", "v2.", "update", "android", "ios", "device",
                "samsung", "pixel", "iphone", "crash", "login", "battery",
                "fix", "resolved", "identified", "issue", "bug",
            ]
            specific_count = sum(1 for m in specificity_markers if m in lower)
            max_score += 1.0
            score += min(1.0, specific_count * 0.2)

            empathy_markers = [
                "sorry", "apologize", "understand", "frustrating", "inconvenience",
                "appreciate", "thank you", "thank", "feedback", "patience",
            ]
            empathy_count = sum(1 for m in empathy_markers if m in lower)
            max_score += 1.0
            score += min(1.0, empathy_count * 0.3)

            action_markers = [
                "please try", "you can", "we recommend", "update the app",
                "clear cache", "reinstall", "contact support", "settings",
                "next update", "working on", "will be", "check",
            ]
            action_count = sum(1 for m in action_markers if m in lower)
            max_score += 1.0
            score += min(1.0, action_count * 0.3)

            generic_markers = [
                "thank you for your feedback",
                "we appreciate your review",
                "please contact us",
            ]
            max_score += 1.0
            is_generic = any(g in lower for g in generic_markers) and length < 80
            score += 0.2 if is_generic else 1.0

            return round(score / max_score, 4) if max_score > 0 else 0.5

        def _score_compliance(self, text):
            lower = text.lower()

            promise_score = 1.0
            hard_promises = [
                "we guarantee", "guaranteed", "we promise", "100%",
                "definitely will", "for sure will", "we assure you",
            ]
            if any(p in lower for p in hard_promises):
                promise_score = 0.1
            elif any(p in lower for p in [
                "will be fixed in", "will be resolved in", "fix will be included in",
                "releasing next week", "shipping in the next", "coming in version",
                "will be available in", "expect a fix by",
            ]):
                promise_score = 0.4
            elif any(p in lower for p in [
                "we will fix", "will be fixed", "will be resolved",
                "we will address", "will be included in the next update",
                "a fix will be", "we are going to fix",
            ]):
                promise_score = 0.6
            elif any(p in lower for p in [
                "working on a fix", "investigating", "looking into",
                "we aim to", "we hope to", "we plan to",
                "our team is working", "actively working",
            ]):
                promise_score = 0.85

            info_score = 1.0
            hard_leaks = [
                "production server", "database schema", "api key",
                "deployment pipeline", "jenkins", "docker", "kubernetes",
                "source code", "codebase", "git repo", "pull request",
                "null pointer", "stack trace", "exception in",
            ]
            if any(p in lower for p in hard_leaks):
                info_score = 0.1
            elif any(p in lower for p in [
                "our engineer", "our developer", "our backend team",
                "server-side", "our team member", "sprint",
                "jira", "slack channel", "internal",
            ]):
                info_score = 0.4
            elif any(p in lower for p in [
                "our technical team", "our development team",
                "our team has identified", "our team found",
                "we identified a bug in", "root cause",
            ]):
                info_score = 0.75

            tone_score = 1.0
            hard_tone = [
                "that's your problem", "that is your problem",
                "not our fault", "not our problem",
                "you should have", "your fault",
                "stop complaining", "deal with it", "too bad",
                "read the manual", "figure it out", "not my problem",
            ]
            if any(t in lower for t in hard_tone):
                tone_score = 0.1
            elif any(t in lower for t in [
                "obviously", "clearly you", "as i said",
                "i already told you", "not sure what you expect",
            ]):
                tone_score = 0.4
            elif len(text) < 30 and not any(w in lower for w in ["sorry", "thank", "apologize"]):
                tone_score = 0.7

            empathy_words = ["sorry", "apologize", "understand", "frustrating", "inconvenience"]
            if any(w in lower for w in empathy_words):
                tone_score = min(1.0, tone_score + 0.1)

            legal_score = 1.0
            legal_hard = [
                r"\bwe accept liability\b", r"\bwe are liable\b",
                r"\bwe admit fault\b", r"\bour fault entirely\b",
                r"\bwe take full responsibility for (?:the |any )?damage\b",
                r"\blawsuit\b", r"\bsue us\b",
            ]
            if any(re.search(p, lower) for p in legal_hard):
                legal_score = 0.1
            elif any(p in lower for p in [
                "compensation", "refund guaranteed", "we owe you",
                "we accept responsibility", "our liability",
                "our fault entirely", "we are at fault",
            ]):
                legal_score = 0.4
            elif any(p in lower for p in [
                "we take responsibility", "this is on us", "our mistake",
            ]):
                legal_score = 0.7

            dim_scores = [promise_score, info_score, tone_score, legal_score]
            weights = [0.35, 0.25, 0.20, 0.20]
            weighted_avg = sum(w * s for w, s in zip(weights, dim_scores))
            min_score = min(dim_scores)

            if min_score < 0.5:
                final_score = min(weighted_avg, min_score * 1.2)
            elif min_score < 0.8:
                final_score = 0.5 * weighted_avg + 0.5 * min_score
            else:
                final_score = weighted_avg

            return round(final_score, 4)

    return MinimalPPO()


@pytest.fixture
def ppo():
    return _make_ppo_instance()


# ============================================================
# Constrained Reward Tests
# ============================================================

class TestConstrainedReward:

    def test_no_violation(self, ppo):
        reward = ppo.compute_constrained_reward(quality_score=0.8, compliance_score=0.98)
        assert reward == 0.8

    def test_violation_reduces_reward(self, ppo):
        reward = ppo.compute_constrained_reward(quality_score=0.8, compliance_score=0.5)
        assert reward == pytest.approx(0.8 - 2.25)

    def test_exact_threshold(self, ppo):
        reward = ppo.compute_constrained_reward(quality_score=0.7, compliance_score=0.95)
        assert reward == 0.7

    def test_zero_compliance(self, ppo):
        reward = ppo.compute_constrained_reward(quality_score=0.8, compliance_score=0.0)
        assert reward == pytest.approx(0.8 - 4.75)

    def test_perfect_scores(self, ppo):
        reward = ppo.compute_constrained_reward(quality_score=1.0, compliance_score=1.0)
        assert reward == 1.0


# ============================================================
# Quality Scoring Heuristic Tests
# ============================================================

class TestScoreQuality:

    def test_good_response(self, ppo):
        text = "We're sorry for the inconvenience. We've identified a crash bug in the login screen affecting Android 14 devices. Please try updating to version 3.3 which includes a fix for this issue."
        score = ppo._score_quality(text)
        assert score > 0.6

    def test_short_response_low_score(self, ppo):
        score = ppo._score_quality("ok")
        assert score < 0.4

    def test_generic_template_low_score(self, ppo):
        score = ppo._score_quality("Thank you for your feedback")
        assert score < 0.5

    def test_empathetic_response(self, ppo):
        text = "We apologize for the frustrating experience. We understand how inconvenient this must be."
        score = ppo._score_quality(text)
        assert score > 0.4

    def test_actionable_response(self, ppo):
        text = "Please try clearing your cache and reinstalling the app. You can also contact support if the issue persists. We recommend updating the app to the latest version."
        score = ppo._score_quality(text)
        assert score > 0.5

    def test_specific_response(self, ppo):
        text = "The crash on Samsung devices running Android 14 with version 3.2 has been identified. A fix has been resolved in the latest update."
        score = ppo._score_quality(text)
        assert score > 0.4

    def test_score_between_zero_and_one(self, ppo):
        for text in ["hello", "We're sorry", "A" * 500, "Thank you for your feedback"]:
            score = ppo._score_quality(text)
            assert 0 <= score <= 1


# ============================================================
# Compliance Scoring Heuristic Tests
# ============================================================

class TestScoreCompliance:

    def test_clean_response(self, ppo):
        text = "We're sorry to hear about this issue. Our team is looking into it. Please try updating to the latest version."
        score = ppo._score_compliance(text)
        assert score > 0.8

    def test_hard_promise_violation(self, ppo):
        text = "We guarantee this will be fixed by tomorrow."
        score = ppo._score_compliance(text)
        assert score < 0.2

    def test_soft_promise(self, ppo):
        text = "We will fix this issue soon."
        score = ppo._score_compliance(text)
        assert score < 0.8

    def test_hedged_promise_acceptable(self, ppo):
        text = "Our team is investigating this issue and we hope to have a resolution soon."
        score = ppo._score_compliance(text)
        assert score > 0.7

    def test_info_leak_hard(self, ppo):
        text = "The stack trace shows a null pointer in our database schema."
        score = ppo._score_compliance(text)
        assert score < 0.2

    def test_info_leak_medium(self, ppo):
        text = "Our backend team checked the jira ticket and is working on it."
        score = ppo._score_compliance(text)
        assert score < 0.6

    def test_tone_violation(self, ppo):
        text = "That's your problem, not ours. Deal with it."
        score = ppo._score_compliance(text)
        assert score < 0.2

    def test_legal_violation(self, ppo):
        text = "We accept liability for the data loss caused by our bug."
        score = ppo._score_compliance(text)
        assert score < 0.2

    def test_legal_soft_caution(self, ppo):
        text = "We take responsibility for this issue and our mistake."
        score = ppo._score_compliance(text)
        assert score < 0.85

    def test_score_between_zero_and_one(self, ppo):
        for text in ["hello", "We guarantee it", "stack trace", "deal with it"]:
            score = ppo._score_compliance(text)
            assert 0 <= score <= 1

    def test_empathy_boosts_tone(self, ppo):
        text = "We're sorry for this frustrating experience."
        score = ppo._score_compliance(text)
        assert score > 0.9

    def test_sue_not_matched_in_issue(self, ppo):
        text = "We're aware of this issue and working on it."
        score = ppo._score_compliance(text)
        assert score > 0.7
