"""Dual-stream feedback collection: quality + compliance."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime


QUALITY_DIMS = ["helpfulness", "specificity", "empathy", "accuracy", "actionability"]
COMPLIANCE_DIMS = ["no_false_promises", "no_info_leak", "tone_compliant", "legally_safe"]


class DualStreamFeedbackCollector:
    """Collects and exports feedback in dual-stream format (quality + compliance)."""

    def __init__(self, storage_path: str = "data/feedback/stage5_feedback.json"):
        self.storage_path = Path(storage_path)
        self.quality_ratings: list[dict] = []
        self.compliance_ratings: list[dict] = []
        if self.storage_path.exists():
            data = json.loads(self.storage_path.read_text())
            self.quality_ratings = data.get("quality", [])
            self.compliance_ratings = data.get("compliance", [])

    def record_quality(
        self, response_id: str, scores: dict[str, int], rater_id: str
    ) -> None:
        """Record quality scores (Stream 1)."""
        self.quality_ratings.append({
            "response_id": response_id,
            "scores": scores,
            "rater_id": rater_id,
            "timestamp": datetime.now().isoformat(),
        })
        self._save()

    def record_compliance(
        self, response_id: str, flags: dict[str, bool], rater_id: str
    ) -> None:
        """Record compliance flags (Stream 2)."""
        self.compliance_ratings.append({
            "response_id": response_id,
            "flags": flags,
            "rater_id": rater_id,
            "timestamp": datetime.now().isoformat(),
        })
        self._save()

    def export_kto_data(self) -> list[dict]:
        """Export as binary good/bad for KTO training."""
        response_data = {}
        for q in self.quality_ratings:
            rid = q["response_id"]
            if rid not in response_data:
                response_data[rid] = {"quality_scores": [], "compliant": True}
            response_data[rid]["quality_scores"].append(
                sum(q["scores"].values()) / len(q["scores"])
            )
        for c in self.compliance_ratings:
            rid = c["response_id"]
            if rid in response_data and not all(c["flags"].values()):
                response_data[rid]["compliant"] = False

        results = []
        for rid, data in response_data.items():
            avg_quality = sum(data["quality_scores"]) / len(data["quality_scores"])
            is_good = avg_quality >= 3.0 and data["compliant"]
            results.append({"response_id": rid, "label": is_good})
        return results

    def export_dpo_data(self) -> list[tuple[str, str]]:
        """Export as paired preferences for DPO training."""
        # Group by response, compute average quality
        scores_by_response = {}
        for q in self.quality_ratings:
            rid = q["response_id"]
            avg = sum(q["scores"].values()) / len(q["scores"])
            scores_by_response.setdefault(rid, []).append(avg)

        avg_scores = {
            rid: sum(s) / len(s) for rid, s in scores_by_response.items()
        }

        # Create pairs: (chosen, rejected) where chosen has higher score
        sorted_responses = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        pairs = []
        for i in range(len(sorted_responses) - 1):
            chosen_id = sorted_responses[i][0]
            rejected_id = sorted_responses[i + 1][0]
            pairs.append((chosen_id, rejected_id))
        return pairs

    def export_ppo_data(self) -> tuple[list[dict], list[dict]]:
        """Export as separate quality scores + compliance labels for Constrained PPO."""
        quality_data = []
        for q in self.quality_ratings:
            quality_data.append({
                "response_id": q["response_id"],
                "scores": q["scores"],
            })

        compliance_data = []
        for c in self.compliance_ratings:
            compliance_data.append({
                "response_id": c["response_id"],
                "compliant": all(c["flags"].values()),
                "flags": c["flags"],
            })

        return quality_data, compliance_data

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps({
            "quality": self.quality_ratings,
            "compliance": self.compliance_ratings,
        }, indent=2))
