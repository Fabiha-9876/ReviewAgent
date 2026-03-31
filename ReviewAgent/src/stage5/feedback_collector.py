"""Dual-stream feedback collection: quality + compliance."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime


QUALITY_DIMS = ["helpfulness", "specificity", "empathy", "accuracy", "actionability"]
COMPLIANCE_DIMS = ["no_false_promises", "no_info_leak", "tone_compliant", "legally_safe"]


class DualStreamFeedbackCollector:
    """Collects and exports feedback in dual-stream format (quality + compliance).

    Stores the actual prompt and response texts alongside scores so that
    DPO/KTO/PPO trainers can access the full text without a separate lookup.
    """

    def __init__(self, storage_path: str = "data/feedback/stage5_feedback.json"):
        self.storage_path = Path(storage_path)
        self.quality_ratings: list[dict] = []
        self.compliance_ratings: list[dict] = []
        self.response_texts: dict[str, dict] = {}  # response_id -> {"prompt": ..., "response": ...}
        if self.storage_path.exists():
            data = json.loads(self.storage_path.read_text())
            self.quality_ratings = data.get("quality", [])
            self.compliance_ratings = data.get("compliance", [])
            self.response_texts = data.get("response_texts", {})

    def register_response(
        self, response_id: str, prompt_text: str, response_text: str,
        issue_id: str = "",
    ) -> None:
        """Register a response's actual text content for later DPO/KTO export.

        Call this when a response is generated (Stage 4b), BEFORE collecting feedback.
        """
        self.response_texts[response_id] = {
            "prompt": prompt_text,
            "response": response_text,
            "issue_id": issue_id,
        }
        self._save()

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

    def export_dpo_data(self) -> list[dict]:
        """Export as paired preferences with full text for DPO training.

        Returns list of {"prompt": str, "chosen": str, "rejected": str} dicts.
        Pairs are formed by grouping responses to the same issue/prompt and
        picking the higher-scored response as "chosen" and lower as "rejected".
        """
        # Step 1: Compute average quality score per response
        scores_by_response = {}
        for q in self.quality_ratings:
            rid = q["response_id"]
            avg = sum(q["scores"].values()) / len(q["scores"])
            scores_by_response.setdefault(rid, []).append(avg)

        avg_scores = {
            rid: sum(s) / len(s) for rid, s in scores_by_response.items()
        }

        # Step 2: Group responses by issue_id (same prompt)
        issue_groups: dict[str, list[str]] = {}
        for rid, text_data in self.response_texts.items():
            if rid not in avg_scores:
                continue
            issue_id = text_data.get("issue_id", "")
            # If no issue_id, use the prompt text as grouping key
            group_key = issue_id if issue_id else text_data.get("prompt", "")[:100]
            issue_groups.setdefault(group_key, []).append(rid)

        # Step 3: Create (prompt, chosen, rejected) triples from each group
        pairs = []
        for group_key, response_ids in issue_groups.items():
            if len(response_ids) < 2:
                continue

            # Sort by score descending
            sorted_rids = sorted(response_ids, key=lambda rid: avg_scores.get(rid, 0), reverse=True)

            # Pair the best with each worse response
            best_rid = sorted_rids[0]
            best_data = self.response_texts.get(best_rid, {})

            for worse_rid in sorted_rids[1:]:
                worse_data = self.response_texts.get(worse_rid, {})

                # Only pair if there is a meaningful score difference (> 0.5)
                score_diff = avg_scores.get(best_rid, 0) - avg_scores.get(worse_rid, 0)
                if score_diff < 0.5:
                    continue

                pairs.append({
                    "prompt": best_data.get("prompt", ""),
                    "chosen": best_data.get("response", ""),
                    "rejected": worse_data.get("response", ""),
                    "chosen_score": avg_scores.get(best_rid, 0),
                    "rejected_score": avg_scores.get(worse_rid, 0),
                })

        # Step 4: If not enough issue-grouped pairs, fall back to global ranking
        if len(pairs) < 5:
            sorted_all = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            for i in range(len(sorted_all) - 1):
                chosen_rid = sorted_all[i][0]
                rejected_rid = sorted_all[i + 1][0]
                chosen_data = self.response_texts.get(chosen_rid, {})
                rejected_data = self.response_texts.get(rejected_rid, {})

                if not chosen_data.get("response") or not rejected_data.get("response"):
                    continue

                score_diff = sorted_all[i][1] - sorted_all[i + 1][1]
                if score_diff < 0.3:
                    continue

                pairs.append({
                    "prompt": chosen_data.get("prompt", ""),
                    "chosen": chosen_data.get("response", ""),
                    "rejected": rejected_data.get("response", ""),
                    "chosen_score": sorted_all[i][1],
                    "rejected_score": sorted_all[i + 1][1],
                })

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

    def export_kto_data_with_text(self) -> list[dict]:
        """Export KTO data with actual text (not just IDs).

        Returns list of {"prompt": str, "response": str, "label": bool}.
        """
        # First get the binary labels
        kto_labels = {d["response_id"]: d["label"] for d in self.export_kto_data()}

        results = []
        for rid, label in kto_labels.items():
            text_data = self.response_texts.get(rid, {})
            if text_data.get("prompt") and text_data.get("response"):
                results.append({
                    "prompt": text_data["prompt"],
                    "response": text_data["response"],
                    "label": label,
                })
        return results

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps({
            "quality": self.quality_ratings,
            "compliance": self.compliance_ratings,
            "response_texts": self.response_texts,
        }, indent=2))
