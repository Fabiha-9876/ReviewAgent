"""HITL Checkpoint #1: confidence-based flagging for classification verification."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

from src.common.schemas import ReviewObject


class Stage1HITLCheckpoint:
    """Manages human-in-the-loop verification for ambiguous classifications."""

    def __init__(self, corrections_path: str = "data/feedback/stage1_corrections.json"):
        self.corrections_path = Path(corrections_path)
        self.corrections: list[dict] = []
        if self.corrections_path.exists():
            self.corrections = json.loads(self.corrections_path.read_text())

    def flag_for_review(self, review: ReviewObject) -> bool:
        """Determine if a review needs human verification."""
        return review.flagged_for_hitl

    def get_flagged_reviews(self, reviews: list[ReviewObject]) -> list[ReviewObject]:
        """Filter reviews that need human review."""
        return [r for r in reviews if r.flagged_for_hitl]

    def record_correction(
        self,
        review_id: str,
        original_labels: list[str],
        corrected_labels: list[str],
        rater_id: str,
    ) -> None:
        """Record a human correction for a flagged review."""
        self.corrections.append(
            {
                "review_id": review_id,
                "original_labels": original_labels,
                "corrected_labels": corrected_labels,
                "rater_id": rater_id,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self._save()

    def apply_correction(self, review: ReviewObject, corrected_labels: list[str]) -> ReviewObject:
        """Apply a human correction to a review."""
        review.labels = corrected_labels
        review.hitl_corrections = {"corrected_labels": corrected_labels}
        review.flagged_for_hitl = False
        return review

    def get_retraining_data(self) -> list[dict]:
        """Export corrections as training data for active learning."""
        return [
            {"review_id": c["review_id"], "labels": c["corrected_labels"]}
            for c in self.corrections
        ]

    def _save(self) -> None:
        self.corrections_path.parent.mkdir(parents=True, exist_ok=True)
        self.corrections_path.write_text(json.dumps(self.corrections, indent=2))
