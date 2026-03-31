"""Experiment 3: Dual-Objective vs Single-Objective RLHF (RQ3).

Compares KTO, DPO, and Constrained PPO across 3 iterations.
Statistical test: Bradley-Terry + McNemar.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

from .statistical_tests import bradley_terry, mcnemar_test


@dataclass
class Experiment3Results:
    """Results for Experiment 3."""

    iteration_metrics: list[dict] = field(default_factory=list)
    preference_data: list[tuple[int, int]] = field(default_factory=list)
    safety_violations: dict[str, list[bool]] = field(default_factory=dict)
    bradley_terry_results: dict = field(default_factory=dict)
    mcnemar_results: dict = field(default_factory=dict)


class Experiment3Runner:
    """Runs Experiment 3: Dual vs Single Objective RLHF."""

    CONDITIONS = ["kto_single", "dpo_single", "constrained_ppo_dual"]

    def __init__(
        self,
        output_dir: str = "data/processed/experiment3",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = Experiment3Results()

    def record_iteration(
        self,
        iteration: int,
        metrics_per_model: dict[str, dict],
    ) -> None:
        """Record metrics for one RLHF iteration.

        Args:
            metrics_per_model: {model_name: {"avg_quality": float, "violation_rate": float, ...}}
        """
        self.results.iteration_metrics.append({
            "iteration": iteration,
            "models": metrics_per_model,
        })

    def record_preferences(
        self, preferences: list[tuple[int, int]]
    ) -> None:
        """Record pairwise preferences (winner_idx, loser_idx).

        Model indices: 0=KTO, 1=DPO, 2=Constrained PPO.
        """
        self.results.preference_data.extend(preferences)

    def record_safety_violations(
        self, violations: dict[str, list[bool]]
    ) -> None:
        """Record safety violations per model.

        Args:
            violations: {"kto_single": [True, False, ...], "dpo_single": [...], ...}
        """
        self.results.safety_violations = violations

    def compare(self) -> dict:
        """Run Bradley-Terry + McNemar tests."""
        results = {}

        # Bradley-Terry for preference data
        if self.results.preference_data:
            bt = bradley_terry(self.results.preference_data, n_models=3)
            self.results.bradley_terry_results = bt
            results["bradley_terry"] = bt

            # Compute pairwise win rates
            wins = {i: 0 for i in range(3)}
            total = {i: 0 for i in range(3)}
            for winner, loser in self.results.preference_data:
                wins[winner] += 1
                total[winner] += 1
                total[loser] += 1
            results["win_rates"] = {
                self.CONDITIONS[i]: wins[i] / total[i] if total[i] > 0 else 0
                for i in range(3)
            }

        # McNemar for safety violations
        if self.results.safety_violations:
            pairs = [
                ("kto_single", "constrained_ppo_dual"),
                ("dpo_single", "constrained_ppo_dual"),
                ("kto_single", "dpo_single"),
            ]
            mcnemar_results = {}
            for model_a, model_b in pairs:
                if model_a in self.results.safety_violations and model_b in self.results.safety_violations:
                    test = mcnemar_test(
                        self.results.safety_violations[model_a],
                        self.results.safety_violations[model_b],
                    )
                    test["comparison"] = f"{model_a} vs {model_b}"
                    mcnemar_results[f"{model_a}_vs_{model_b}"] = test

            self.results.mcnemar_results = mcnemar_results
            results["mcnemar"] = mcnemar_results

        return results

    def report(self) -> str:
        """Generate summary report."""
        lines = ["# Experiment 3: Dual-Objective vs Single-Objective RLHF (RQ3)\n"]

        # Iteration metrics
        if self.results.iteration_metrics:
            lines.append("## Per-Iteration Metrics")
            for it in self.results.iteration_metrics:
                lines.append(f"\n### Iteration {it['iteration']}")
                lines.append("| Model | Avg Quality | Violation Rate |")
                lines.append("|---|---|---|")
                for model, metrics in it["models"].items():
                    lines.append(
                        f"| {model} | {metrics.get('avg_quality', 0):.3f} | "
                        f"{metrics.get('violation_rate', 0):.3f} |"
                    )

        # Bradley-Terry
        if self.results.bradley_terry_results:
            lines.append("\n## Bradley-Terry Model Strengths")
            bt = self.results.bradley_terry_results
            for i, cond in enumerate(self.CONDITIONS):
                lines.append(f"- {cond}: strength = {bt['strengths'][i]:.3f}")
            lines.append("\nPairwise win probabilities:")
            for pair, prob in bt.get("win_probabilities", {}).items():
                lines.append(f"- {pair}: {prob:.3f}")

        # McNemar
        if self.results.mcnemar_results:
            lines.append("\n## McNemar Tests (Safety Violations)")
            lines.append("| Comparison | Rate A | Rate B | p-value | Significant |")
            lines.append("|---|---|---|---|---|")
            for key, test in self.results.mcnemar_results.items():
                sig = "Yes" if test["significant"] else "No"
                lines.append(
                    f"| {test['comparison']} | {test.get('rate_a', 0):.3f} | "
                    f"{test.get('rate_b', 0):.3f} | {test['p_value']:.4f} | {sig} |"
                )

        report = "\n".join(lines)
        (self.output_dir / "experiment3_report.md").write_text(report)
        return report
