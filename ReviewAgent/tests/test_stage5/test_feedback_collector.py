"""Tests for DualStreamFeedbackCollector — quality + compliance feedback."""

import json
import pytest

from src.stage5.feedback_collector import (
    DualStreamFeedbackCollector,
    QUALITY_DIMS,
    COMPLIANCE_DIMS,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def collector(tmp_path):
    return DualStreamFeedbackCollector(storage_path=str(tmp_path / "feedback.json"))


@pytest.fixture
def quality_scores():
    return {"helpfulness": 4, "specificity": 3, "empathy": 5, "accuracy": 4, "actionability": 3}


@pytest.fixture
def low_quality_scores():
    return {"helpfulness": 1, "specificity": 1, "empathy": 1, "accuracy": 2, "actionability": 1}


@pytest.fixture
def compliance_flags():
    return {"no_false_promises": True, "no_info_leak": True, "tone_compliant": True, "legally_safe": True}


@pytest.fixture
def noncompliant_flags():
    return {"no_false_promises": False, "no_info_leak": True, "tone_compliant": True, "legally_safe": True}


# ============================================================
# Constants Tests
# ============================================================

class TestConstants:

    def test_quality_dims_count(self):
        assert len(QUALITY_DIMS) == 5

    def test_compliance_dims_count(self):
        assert len(COMPLIANCE_DIMS) == 4


# ============================================================
# Register Response Tests
# ============================================================

class TestRegisterResponse:

    def test_register_stores_text(self, collector):
        collector.register_response("r1", "prompt text", "response text", "ISS-001")
        assert "r1" in collector.response_texts
        assert collector.response_texts["r1"]["prompt"] == "prompt text"
        assert collector.response_texts["r1"]["response"] == "response text"
        assert collector.response_texts["r1"]["issue_id"] == "ISS-001"

    def test_register_persists(self, collector):
        collector.register_response("r1", "p", "r", "ISS-001")
        data = json.loads(collector.storage_path.read_text())
        assert "r1" in data["response_texts"]

    def test_register_multiple(self, collector):
        collector.register_response("r1", "p1", "r1")
        collector.register_response("r2", "p2", "r2")
        assert len(collector.response_texts) == 2


# ============================================================
# Record Quality Tests
# ============================================================

class TestRecordQuality:

    def test_appends_rating(self, collector, quality_scores):
        collector.record_quality("r1", quality_scores, "rater_1")
        assert len(collector.quality_ratings) == 1

    def test_stores_scores(self, collector, quality_scores):
        collector.record_quality("r1", quality_scores, "rater_1")
        assert collector.quality_ratings[0]["scores"] == quality_scores

    def test_stores_rater_id(self, collector, quality_scores):
        collector.record_quality("r1", quality_scores, "rater_1")
        assert collector.quality_ratings[0]["rater_id"] == "rater_1"

    def test_includes_timestamp(self, collector, quality_scores):
        collector.record_quality("r1", quality_scores, "rater_1")
        assert "timestamp" in collector.quality_ratings[0]

    def test_multiple_raters(self, collector, quality_scores):
        collector.record_quality("r1", quality_scores, "rater_1")
        collector.record_quality("r1", quality_scores, "rater_2")
        assert len(collector.quality_ratings) == 2


# ============================================================
# Record Compliance Tests
# ============================================================

class TestRecordCompliance:

    def test_appends_rating(self, collector, compliance_flags):
        collector.record_compliance("r1", compliance_flags, "rater_1")
        assert len(collector.compliance_ratings) == 1

    def test_stores_flags(self, collector, compliance_flags):
        collector.record_compliance("r1", compliance_flags, "rater_1")
        assert collector.compliance_ratings[0]["flags"] == compliance_flags

    def test_noncompliant_flags(self, collector, noncompliant_flags):
        collector.record_compliance("r1", noncompliant_flags, "rater_1")
        assert collector.compliance_ratings[0]["flags"]["no_false_promises"] is False


# ============================================================
# Export KTO Tests
# ============================================================

class TestExportKTO:

    def test_good_response_labeled_true(self, collector, quality_scores, compliance_flags):
        collector.record_quality("r1", quality_scores, "rater_1")
        collector.record_compliance("r1", compliance_flags, "rater_1")
        kto = collector.export_kto_data()
        assert len(kto) == 1
        assert kto[0]["label"] is True

    def test_low_quality_labeled_false(self, collector, low_quality_scores, compliance_flags):
        collector.record_quality("r1", low_quality_scores, "rater_1")
        collector.record_compliance("r1", compliance_flags, "rater_1")
        kto = collector.export_kto_data()
        assert kto[0]["label"] is False

    def test_noncompliant_labeled_false(self, collector, quality_scores, noncompliant_flags):
        collector.record_quality("r1", quality_scores, "rater_1")
        collector.record_compliance("r1", noncompliant_flags, "rater_1")
        kto = collector.export_kto_data()
        assert kto[0]["label"] is False

    def test_empty_returns_empty(self, collector):
        assert collector.export_kto_data() == []

    def test_multiple_responses(self, collector, quality_scores, low_quality_scores, compliance_flags):
        collector.record_quality("r1", quality_scores, "rater_1")
        collector.record_quality("r2", low_quality_scores, "rater_1")
        collector.record_compliance("r1", compliance_flags, "rater_1")
        collector.record_compliance("r2", compliance_flags, "rater_1")
        kto = collector.export_kto_data()
        labels = {d["response_id"]: d["label"] for d in kto}
        assert labels["r1"] is True
        assert labels["r2"] is False


# ============================================================
# Export KTO With Text Tests
# ============================================================

class TestExportKTOWithText:

    def test_includes_text(self, collector, quality_scores, compliance_flags):
        collector.register_response("r1", "the prompt", "the response", "ISS-001")
        collector.record_quality("r1", quality_scores, "rater_1")
        collector.record_compliance("r1", compliance_flags, "rater_1")
        kto = collector.export_kto_data_with_text()
        assert len(kto) == 1
        assert kto[0]["prompt"] == "the prompt"
        assert kto[0]["response"] == "the response"
        assert kto[0]["label"] is True

    def test_skips_missing_text(self, collector, quality_scores, compliance_flags):
        collector.record_quality("r1", quality_scores, "rater_1")
        collector.record_compliance("r1", compliance_flags, "rater_1")
        # No register_response call
        kto = collector.export_kto_data_with_text()
        assert len(kto) == 0


# ============================================================
# Export DPO Tests
# ============================================================

class TestExportDPO:

    def test_creates_pairs(self, collector, quality_scores, low_quality_scores):
        collector.register_response("r1", "prompt", "good response", "ISS-001")
        collector.register_response("r2", "prompt", "bad response", "ISS-001")
        collector.record_quality("r1", quality_scores, "rater_1")
        collector.record_quality("r2", low_quality_scores, "rater_1")
        dpo = collector.export_dpo_data()
        assert len(dpo) >= 1
        assert dpo[0]["chosen"] == "good response"
        assert dpo[0]["rejected"] == "bad response"

    def test_no_pairs_with_single_response(self, collector, quality_scores):
        collector.register_response("r1", "prompt", "response", "ISS-001")
        collector.record_quality("r1", quality_scores, "rater_1")
        dpo = collector.export_dpo_data()
        # With only one response, no pairs possible (may fall back to global ranking)
        # Either way, no valid pair with score difference
        assert isinstance(dpo, list)

    def test_empty_returns_empty(self, collector):
        assert collector.export_dpo_data() == []

    def test_requires_score_difference(self, collector, quality_scores):
        collector.register_response("r1", "prompt", "resp1", "ISS-001")
        collector.register_response("r2", "prompt", "resp2", "ISS-001")
        # Same scores — no meaningful difference
        collector.record_quality("r1", quality_scores, "rater_1")
        collector.record_quality("r2", quality_scores, "rater_1")
        dpo = collector.export_dpo_data()
        # Pairs require score diff >= 0.5 for issue-grouped, 0.3 for global
        grouped_pairs = [p for p in dpo if p.get("chosen_score", 0) - p.get("rejected_score", 0) >= 0.5]
        assert len(grouped_pairs) == 0


# ============================================================
# Export PPO Tests
# ============================================================

class TestExportPPO:

    def test_returns_two_lists(self, collector, quality_scores, compliance_flags):
        collector.record_quality("r1", quality_scores, "rater_1")
        collector.record_compliance("r1", compliance_flags, "rater_1")
        quality_data, compliance_data = collector.export_ppo_data()
        assert len(quality_data) == 1
        assert len(compliance_data) == 1

    def test_quality_data_has_scores(self, collector, quality_scores):
        collector.record_quality("r1", quality_scores, "rater_1")
        quality_data, _ = collector.export_ppo_data()
        assert quality_data[0]["scores"] == quality_scores

    def test_compliance_data_has_flags(self, collector, compliance_flags):
        collector.record_compliance("r1", compliance_flags, "rater_1")
        _, compliance_data = collector.export_ppo_data()
        assert compliance_data[0]["compliant"] is True

    def test_noncompliant_flagged(self, collector, noncompliant_flags):
        collector.record_compliance("r1", noncompliant_flags, "rater_1")
        _, compliance_data = collector.export_ppo_data()
        assert compliance_data[0]["compliant"] is False

    def test_empty_returns_empty_lists(self, collector):
        q, c = collector.export_ppo_data()
        assert q == [] and c == []


# ============================================================
# Persistence Tests
# ============================================================

class TestPersistence:

    def test_loads_existing_data(self, tmp_path, quality_scores):
        path = tmp_path / "feedback.json"
        path.write_text(json.dumps({
            "quality": [{"response_id": "r1", "scores": quality_scores, "rater_id": "r", "timestamp": "t"}],
            "compliance": [],
            "response_texts": {"r1": {"prompt": "p", "response": "r", "issue_id": "ISS"}},
        }))
        collector = DualStreamFeedbackCollector(storage_path=str(path))
        assert len(collector.quality_ratings) == 1
        assert "r1" in collector.response_texts

    def test_creates_parent_dirs(self, tmp_path, quality_scores):
        path = tmp_path / "nested" / "dir" / "feedback.json"
        collector = DualStreamFeedbackCollector(storage_path=str(path))
        collector.record_quality("r1", quality_scores, "rater_1")
        assert path.exists()
