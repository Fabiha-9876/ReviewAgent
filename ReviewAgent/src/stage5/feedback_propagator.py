"""Backward feedback propagation to upstream stages."""

from __future__ import annotations

import json
from pathlib import Path


class FeedbackPropagator:
    """Routes corrections and feedback backward to Stages 1, 3, and 4b."""

    def __init__(self, feedback_dir: str = "data/feedback"):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

    def propagate_to_stage1(self, corrections: list[dict]) -> None:
        """Export label corrections for Stage 1 classifier retraining.

        Args:
            corrections: list of {"review_id": str, "corrected_labels": list[str]}
        """
        path = self.feedback_dir / "stage1_retraining_queue.json"
        existing = json.loads(path.read_text()) if path.exists() else []
        existing.extend(corrections)
        path.write_text(json.dumps(existing, indent=2))

    def propagate_to_stage3(self, rubric_feedback: list[dict]) -> dict[str, float]:
        """Identify systematically weak dimensions in Stage 3 output.

        Args:
            rubric_feedback: list of {"spec_id": str, "scores": {"dim": float}}

        Returns:
            dict mapping dimension -> average score (weak dims have low scores)
        """
        dim_totals: dict[str, list[float]] = {}
        for fb in rubric_feedback:
            for dim, score in fb["scores"].items():
                dim_totals.setdefault(dim, []).append(score)

        dim_averages = {
            dim: sum(scores) / len(scores)
            for dim, scores in dim_totals.items()
        }

        # Save weak dimensions for Stage 3 prompt adjustment
        weak_dims = {d: s for d, s in dim_averages.items() if s < 3.0}
        if weak_dims:
            path = self.feedback_dir / "stage3_prompt_adjustments.json"
            path.write_text(json.dumps({
                "weak_dimensions": weak_dims,
                "suggestion": f"Improve these dimensions: {', '.join(weak_dims.keys())}",
            }, indent=2))

        return dim_averages

    def propagate_to_stage4b(self, quality_scores: list[dict]) -> None:
        """Feed quality scores into the RLHF training data queue.

        Args:
            quality_scores: list of {"response_id": str, "scores": {"dim": float}}
        """
        path = self.feedback_dir / "stage4b_rlhf_queue.json"
        existing = json.loads(path.read_text()) if path.exists() else []
        existing.extend(quality_scores)
        path.write_text(json.dumps(existing, indent=2))
