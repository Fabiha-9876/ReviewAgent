"""Tests for ConstrainedPPOTrainer — CMDP reward and heuristic scoring.

These tests exercise compute_constrained_reward, _score_quality, and
_score_compliance which are pure Python (no trl/torch needed). We construct
the trainer object manually to avoid importing trl.
"""

import re
import pytest


def _make_ppo_instance():
    """Bind to the SHIPPED heuristics in src/stage5/constrained_ppo.py.

    The module imports trl at top level and TRL 1.0 removed the PPO API it targets,
    so we cannot import it normally in every environment. Instead of hand-copying the
    logic (which would let these tests pass even if the real code changed or vanished),
    we parse the module with `ast`, lift the three pure-Python methods out of the
    ConstrainedPPOTrainer class body, and exec only those. If a method is renamed,
    removed, or its body stops parsing, these tests fail rather than silently testing
    a stale copy.
    """
    import ast
    import textwrap
    from pathlib import Path

    spec_path = Path(__file__).resolve().parents[2] / "src" / "stage5" / "constrained_ppo.py"
    tree = ast.parse(spec_path.read_text())

    cls = next((n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "ConstrainedPPOTrainer"), None)
    assert cls is not None, "ConstrainedPPOTrainer not found in src/stage5/constrained_ppo.py"

    wanted = {"compute_constrained_reward", "_score_quality", "_score_compliance"}
    methods = {n.name: n for n in cls.body
               if isinstance(n, ast.FunctionDef) and n.name in wanted}
    missing = wanted - set(methods)
    assert not missing, f"shipped ConstrainedPPOTrainer is missing {sorted(missing)}"

    ns = {"re": re}
    body = "\n".join(textwrap.dedent(ast.unparse(methods[name])) for name in sorted(wanted))
    exec(compile(ast.parse(body), str(spec_path), "exec"), ns)

    class ExtractedPPO:
        compliance_threshold = 0.95
        compliance_penalty = 5.0
        quality_reward_model = None
        compliance_reward_model = None

    for name in wanted:
        setattr(ExtractedPPO, name, ns[name])
    return ExtractedPPO()


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
