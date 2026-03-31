"""Evaluation metrics: BLEU, ROUGE-L, BERTScore, rubric aggregation."""

from __future__ import annotations

import numpy as np
from src.common.schemas import IssueSpec


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Compute corpus-level BLEU score."""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    refs = [[ref.split()] for ref in references]
    hyps = [pred.split() for pred in predictions]
    return corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)


def compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    """Compute average ROUGE-L F1 score."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
    return float(np.mean(scores))


def compute_bert_score(
    predictions: list[str], references: list[str], lang: str = "en"
) -> dict[str, float]:
    """Compute BERTScore (precision, recall, F1)."""
    from bert_score import score

    P, R, F1 = score(predictions, references, lang=lang, verbose=False)
    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean()),
    }


def compute_completeness_ratio(spec: IssueSpec) -> float:
    """Compute fraction of required schema fields that are non-empty."""
    required_fields = ["title", "issue_type", "description", "severity", "affected_component"]
    type_specific = {
        "bug_report": ["steps_to_reproduce", "expected_behavior", "actual_behavior"],
        "feature_request": ["user_story", "acceptance_criteria"],
        "performance": ["nfr_category"],
        "usability": ["nielsen_heuristic"],
        "compatibility": ["device_os_matrix"],
    }

    fields_to_check = required_fields + type_specific.get(spec.issue_type, [])
    filled = 0
    for field in fields_to_check:
        value = getattr(spec, field, None)
        if value is not None and value != "" and value != []:
            filled += 1

    return filled / len(fields_to_check) if fields_to_check else 0.0


def compute_krippendorff_alpha(ratings: np.ndarray) -> float:
    """Compute Krippendorff's alpha for inter-annotator agreement.

    Args:
        ratings: 2D array of shape (n_raters, n_items). Use np.nan for missing.
    """
    import krippendorff

    return krippendorff.alpha(reliability_data=ratings, level_of_measurement="ordinal")


def aggregate_rubric_scores(scores: list[dict[str, int]]) -> dict[str, float]:
    """Compute mean per-dimension scores across multiple raters."""
    if not scores:
        return {}
    dims = scores[0].keys()
    return {dim: float(np.mean([s[dim] for s in scores])) for dim in dims}
