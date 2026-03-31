"""Tests for Stage 1 HITL checkpoint."""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from src.common.schemas import ReviewObject
from src.stage1.hitl_checkpoint import Stage1HITLCheckpoint


class TestStage1HITLCheckpoint:
    """Test HITL flagging, correction recording, and retraining data export."""

    @pytest.fixture
    def hitl(self, tmp_path):
        """Create HITL checkpoint with temp storage."""
        return Stage1HITLCheckpoint(corrections_path=str(tmp_path / "corrections.json"))

    @pytest.fixture
    def flagged_review(self):
        return ReviewObject(
            review_id="test-1",
            text="App sometimes crashes maybe",
            rating=2,
            app_id="com.test",
            timestamp=datetime.now(),
            labels=["bug_report"],
            label_confidences={"bug_report": 0.55, "usability": 0.50},
            flagged_for_hitl=True,
        )

    @pytest.fixture
    def clean_review(self):
        return ReviewObject(
            review_id="test-2",
            text="App crashes on login",
            rating=1,
            app_id="com.test",
            timestamp=datetime.now(),
            labels=["bug_report"],
            label_confidences={"bug_report": 0.95},
            flagged_for_hitl=False,
        )

    def test_flag_for_review(self, hitl, flagged_review, clean_review):
        """Only flagged reviews should be marked for review."""
        assert hitl.flag_for_review(flagged_review) is True
        assert hitl.flag_for_review(clean_review) is False

    def test_get_flagged_reviews(self, hitl, flagged_review, clean_review):
        """get_flagged_reviews should filter correctly."""
        reviews = [flagged_review, clean_review]
        flagged = hitl.get_flagged_reviews(reviews)
        assert len(flagged) == 1
        assert flagged[0].review_id == "test-1"

    def test_record_correction(self, hitl):
        """Corrections should be saved to disk."""
        hitl.record_correction(
            review_id="test-1",
            original_labels=["bug_report"],
            corrected_labels=["usability"],
            rater_id="expert_1",
        )
        assert len(hitl.corrections) == 1
        assert hitl.corrections[0]["corrected_labels"] == ["usability"]
        # Verify persisted to disk
        assert Path(hitl.corrections_path).exists()
        saved = json.loads(Path(hitl.corrections_path).read_text())
        assert len(saved) == 1

    def test_apply_correction(self, hitl, flagged_review):
        """apply_correction should update labels and clear flag."""
        corrected = hitl.apply_correction(flagged_review, ["usability"])
        assert corrected.labels == ["usability"]
        assert corrected.flagged_for_hitl is False
        assert corrected.hitl_corrections is not None

    def test_get_retraining_data(self, hitl):
        """Retraining data should include all corrections."""
        hitl.record_correction("r1", ["bug_report"], ["usability"], "expert_1")
        hitl.record_correction("r2", ["praise"], ["feature_request"], "expert_1")

        data = hitl.get_retraining_data()
        assert len(data) == 2
        assert data[0]["review_id"] == "r1"
        assert data[0]["labels"] == ["usability"]
        assert data[1]["labels"] == ["feature_request"]

    def test_multiple_corrections_accumulate(self, hitl):
        """Multiple corrections should stack up."""
        for i in range(5):
            hitl.record_correction(f"r{i}", ["old"], ["new"], f"expert_{i}")
        assert len(hitl.corrections) == 5

    def test_persistence_across_instances(self, tmp_path):
        """Corrections should survive creating a new instance."""
        path = str(tmp_path / "corrections.json")
        hitl1 = Stage1HITLCheckpoint(corrections_path=path)
        hitl1.record_correction("r1", ["bug"], ["feature"], "expert")

        hitl2 = Stage1HITLCheckpoint(corrections_path=path)
        assert len(hitl2.corrections) == 1
        assert hitl2.corrections[0]["review_id"] == "r1"
