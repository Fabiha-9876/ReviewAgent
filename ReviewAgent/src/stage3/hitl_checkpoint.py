"""HITL Checkpoint #2: expert rubric-based validation of issue specs."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

from src.common.schemas import IssueSpec, RubricScores


RUBRIC_DIMENSIONS = ["completeness", "accuracy", "actionability", "specificity", "clarity"]


class Stage3HITLCheckpoint:
    """Expert rubric validation for generated issue specifications."""

    def __init__(
        self,
        min_avg_score: float = 3.0,
        scores_path: str = "data/feedback/stage3_rubric_scores.json",
    ):
        self.min_avg_score = min_avg_score
        self.scores_path = Path(scores_path)
        self.all_scores: list[dict] = []
        if self.scores_path.exists():
            self.all_scores = json.loads(self.scores_path.read_text())

    def record_scores(
        self,
        spec_id: str,
        scores: dict[str, int],
        rater_id: str,
    ) -> None:
        """Record an expert's rubric scores for an issue spec."""
        self.all_scores.append(
            {
                "spec_id": spec_id,
                "scores": scores,
                "rater_id": rater_id,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self._save()

    def check_threshold(self, spec: IssueSpec) -> bool:
        """Check if a spec passes the quality threshold."""
        if spec.rubric_scores is None:
            return False
        return spec.rubric_scores.mean >= self.min_avg_score

    def get_scores_for_spec(self, spec_id: str) -> list[dict]:
        """Get all ratings for a specific spec."""
        return [s for s in self.all_scores if s["spec_id"] == spec_id]

    def aggregate_scores(self, spec_id: str) -> RubricScores | None:
        """Compute mean scores across raters for a spec."""
        ratings = self.get_scores_for_spec(spec_id)
        if not ratings:
            return None

        agg = {dim: 0.0 for dim in RUBRIC_DIMENSIONS}
        for rating in ratings:
            for dim in RUBRIC_DIMENSIONS:
                agg[dim] += rating["scores"].get(dim, 0)

        n = len(ratings)
        return RubricScores(**{dim: round(val / n, 2) for dim, val in agg.items()})

    def get_feedback_summary(self, spec_id: str) -> dict[str, float]:
        """Get per-dimension average scores for feedback."""
        scores = self.aggregate_scores(spec_id)
        if scores is None:
            return {}
        return scores.model_dump()

    def get_weak_dimensions(self, spec_id: str, threshold: float = 3.0) -> list[str]:
        """Identify dimensions scoring below threshold — for targeted regeneration."""
        scores = self.aggregate_scores(spec_id)
        if scores is None:
            return RUBRIC_DIMENSIONS
        return [
            dim for dim in RUBRIC_DIMENSIONS
            if getattr(scores, dim) < threshold
        ]

    async def regenerate_with_feedback(
        self,
        spec: IssueSpec,
        translator,  # ReviewToIssueTranslator — avoid circular import
        cluster,  # IssueCluster
        kg_context: dict | None = None,
    ) -> IssueSpec:
        """Regenerate a spec with dimension-level feedback."""
        weak = self.get_weak_dimensions(spec.issue_id)
        if not weak:
            return spec

        # Prepend feedback to the translation prompt
        feedback_note = (
            f"IMPORTANT: The previous version of this issue spec scored poorly on: "
            f"{', '.join(weak)}. Please specifically improve these dimensions. "
            f"Previous title was: '{spec.title}'"
        )
        # Re-translate with feedback context
        new_spec = await translator.translate(
            cluster,
            kg_context={"feedback": feedback_note, **(kg_context or {})},
        )
        new_spec.issue_id = spec.issue_id  # Keep same ID
        return new_spec

    def _save(self) -> None:
        self.scores_path.parent.mkdir(parents=True, exist_ok=True)
        self.scores_path.write_text(json.dumps(self.all_scores, indent=2))
