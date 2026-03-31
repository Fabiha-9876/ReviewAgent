"""Statistical tests for experiment evaluation."""

from __future__ import annotations

import numpy as np
from scipy import stats


def paired_wilcoxon(scores_a: list[float], scores_b: list[float]) -> dict:
    """Paired Wilcoxon signed-rank test with Cliff's delta effect size."""
    stat, p_value = stats.wilcoxon(scores_a, scores_b)

    # Cliff's delta
    n = len(scores_a)
    comparisons = 0
    dominant = 0
    for a, b in zip(scores_a, scores_b):
        if a > b:
            dominant += 1
        elif a < b:
            dominant -= 1
        comparisons += 1
    cliffs_delta = dominant / comparisons if comparisons > 0 else 0.0

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "cliffs_delta": cliffs_delta,
        "significant": p_value < 0.05,
    }


def friedman_test(conditions: list[list[float]]) -> dict:
    """Friedman test for comparing multiple related conditions."""
    stat, p_value = stats.friedmanchisquare(*conditions)
    return {
        "chi_square": float(stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }


def nemenyi_posthoc(conditions: list[list[float]], names: list[str] | None = None) -> dict:
    """Nemenyi post-hoc test after Friedman (using pingouin)."""
    import pandas as pd
    import pingouin as pg

    n_items = len(conditions[0])
    n_conditions = len(conditions)
    names = names or [f"C{i}" for i in range(n_conditions)]

    # Build long-format DataFrame
    data = []
    for i in range(n_items):
        for j, name in enumerate(names):
            data.append({"item": i, "condition": name, "score": conditions[j][i]})
    df = pd.DataFrame(data)

    # Pairwise Wilcoxon as approximation (pingouin doesn't have Nemenyi directly)
    posthoc = pg.pairwise_tests(
        data=df, dv="score", within="condition", subject="item", parametric=False
    )
    return posthoc.to_dict("records")


def bradley_terry(preferences: list[tuple[int, int]], n_models: int = 3) -> dict:
    """Bradley-Terry model for pairwise preference data.

    Args:
        preferences: list of (winner_idx, loser_idx) pairs
        n_models: number of models being compared
    """
    strengths = np.ones(n_models)

    # Iterative MLE
    for _ in range(100):
        new_strengths = np.zeros(n_models)
        for winner, loser in preferences:
            total = strengths[winner] + strengths[loser]
            new_strengths[winner] += 1.0 / total * strengths[winner]
            new_strengths[loser] += 1.0 / total * strengths[loser]

        new_strengths = new_strengths / new_strengths.sum() * n_models
        if np.allclose(strengths, new_strengths, atol=1e-6):
            break
        strengths = new_strengths

    # Compute pairwise win probabilities
    win_probs = {}
    for i in range(n_models):
        for j in range(i + 1, n_models):
            prob_i = strengths[i] / (strengths[i] + strengths[j])
            win_probs[f"{i}_vs_{j}"] = float(prob_i)

    return {
        "strengths": strengths.tolist(),
        "win_probabilities": win_probs,
    }


def mcnemar_test(violations_a: list[bool], violations_b: list[bool]) -> dict:
    """McNemar's test for paired binary outcomes."""
    # Build contingency table
    b_c = sum(1 for a, b in zip(violations_a, violations_b) if a and not b)
    c_b = sum(1 for a, b in zip(violations_a, violations_b) if not a and b)

    if b_c + c_b == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    stat = (abs(b_c - c_b) - 1) ** 2 / (b_c + c_b)
    p_value = 1 - stats.chi2.cdf(stat, df=1)

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "rate_a": sum(violations_a) / len(violations_a),
        "rate_b": sum(violations_b) / len(violations_b),
    }


def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Apply Bonferroni correction to multiple p-values."""
    n = len(p_values)
    corrected_alpha = alpha / n
    return [
        {
            "original_p": p,
            "corrected_alpha": corrected_alpha,
            "significant": p < corrected_alpha,
        }
        for p in p_values
    ]
