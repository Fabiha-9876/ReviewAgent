"""
Score the 490-review expert-annotated gold standard against V2 LLM, V5,
and corrected_v2 (cleanlab-pipeline) labels.

Methodology framing:
    The 490 reviews were labeled by the lead author (Fabiha Jalal) as the
    domain expert. The three classifiers (V2 LLM, V5 RoBERTa, cleanlab-
    corrected pipeline) act as independent annotators against this gold
    standard. We report:
      - Per-classifier accuracy and Cohen's κ vs expert
      - Per-class precision/recall/F1
      - Confusion matrices
      - Pairwise Cohen's κ between classifiers (independent of expert)

This is a legitimate alternative to 3-human-annotator α/κ — common in
app-review classification papers where expert labels serve as gold.

Output:
    annotator_materials/gold_standard_results.json
    annotator_materials/gold_standard_report.txt   (human-readable)
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from numbers_parser import Document
from sklearn.metrics import (
    classification_report, confusion_matrix, cohen_kappa_score,
)

LABELS = ["bug_report", "feature_request", "performance", "usability",
          "compatibility", "praise", "other"]
LABEL_NORMALIZE = {
    "bug report": "bug_report", "bug_report": "bug_report",
    "feature request": "feature_request", "feature_request": "feature_request",
    "performance": "performance", "usability": "usability",
    "compatibility": "compatibility", "praise": "praise", "other": "other",
}


def normalize(label):
    if label is None:
        return None
    return LABEL_NORMALIZE.get(str(label).strip().lower())


def load_expert_labels():
    """Read the 490 expert-annotated labels from any of the identical files."""
    doc = Document("annotator_materials/annotator_A.numbers")
    for sheet in doc.sheets:
        for tbl in sheet.tables:
            rows = tbl.rows(values_only=True)
            if not rows:
                continue
            header = rows[0]
            if not any(h and "correct_yn" in str(h).lower() for h in header):
                continue
            col = {h: i for i, h in enumerate(header) if h}

            expert = {}
            for r in rows[1:]:
                row_id = r[col["row_id"]]
                if row_id is None:
                    continue
                yn = r[col["correct_yn"]]
                pred = r[col["predicted_label"]]
                final = r[col.get("correct_label_if_no")] if col.get("correct_label_if_no") is not None else None

                if yn is None or str(yn).strip() == "":
                    continue
                yn_str = str(yn).strip().upper()

                if yn_str == "Y":
                    label = normalize(pred)
                elif yn_str == "N":
                    label = normalize(final)
                else:
                    continue

                if label not in LABELS:
                    continue
                expert[int(row_id)] = label
            return expert
    return {}


def load_master_key():
    with open("annotator_materials/master_key.json") as f:
        return json.load(f)


def compute_classifier_metrics(expert, classifier_labels, name):
    """Compute accuracy, Cohen κ, and per-class metrics."""
    aligned_expert = []
    aligned_classifier = []
    for row_id, expert_label in expert.items():
        cl = classifier_labels.get(str(row_id)) or classifier_labels.get(row_id)
        if cl is None:
            continue
        cl = normalize(cl)
        if cl not in LABELS:
            continue
        aligned_expert.append(expert_label)
        aligned_classifier.append(cl)

    n = len(aligned_expert)
    if n == 0:
        return None

    correct = sum(1 for e, c in zip(aligned_expert, aligned_classifier) if e == c)
    accuracy = correct / n
    kappa = cohen_kappa_score(aligned_expert, aligned_classifier, labels=LABELS)
    rep = classification_report(
        aligned_expert, aligned_classifier, labels=LABELS,
        target_names=LABELS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(aligned_expert, aligned_classifier, labels=LABELS).tolist()

    return {
        "name": name,
        "n_aligned": n,
        "accuracy": round(accuracy, 4),
        "cohen_kappa": round(kappa, 4),
        "macro_f1": round(rep["macro avg"]["f1-score"], 4),
        "weighted_f1": round(rep["weighted avg"]["f1-score"], 4),
        "per_class_f1": {l: round(rep[l]["f1-score"], 3) for l in LABELS},
        "per_class_support_in_expert": {l: int(rep[l]["support"]) for l in LABELS},
        "confusion_matrix": cm,
    }


def main():
    out_dir = Path("annotator_materials")

    print("Loading expert-annotated 490 reviews")
    expert = load_expert_labels()
    print(f"  {len(expert)} expert labels loaded")

    expert_dist = Counter(expert.values())
    print(f"\nExpert label distribution:")
    for l in LABELS:
        print(f"  {l:18s} {expert_dist.get(l, 0):>4}")

    print("\nLoading classifier labels from master_key.json")
    key = load_master_key()
    v2_labels  = key["main_v2_labels"]
    v5_labels  = key["main_v5_labels"]
    corr_labels = key["main_corrected_v2_labels"]

    classifier_metrics = {
        "V2_LLM_original":         compute_classifier_metrics(expert, v2_labels, "V2_LLM_original"),
        "corrected_v2_cleanlab":   compute_classifier_metrics(expert, corr_labels, "corrected_v2_cleanlab"),
        "V5_classifier":           compute_classifier_metrics(expert, v5_labels, "V5_classifier"),
    }

    # Pairwise Cohen κ between the three classifiers (no expert)
    aligned_v2, aligned_corr, aligned_v5 = [], [], []
    for row_id in sorted(expert.keys()):
        v2  = normalize(v2_labels.get(str(row_id)) or v2_labels.get(row_id))
        co  = normalize(corr_labels.get(str(row_id)) or corr_labels.get(row_id))
        v5_ = normalize(v5_labels.get(str(row_id)) or v5_labels.get(row_id))
        if v2 in LABELS and co in LABELS and v5_ in LABELS:
            aligned_v2.append(v2)
            aligned_corr.append(co)
            aligned_v5.append(v5_)

    pairwise = {
        "V2_vs_corrected":   round(cohen_kappa_score(aligned_v2, aligned_corr, labels=LABELS), 4),
        "V2_vs_V5":          round(cohen_kappa_score(aligned_v2, aligned_v5, labels=LABELS), 4),
        "corrected_vs_V5":   round(cohen_kappa_score(aligned_corr, aligned_v5, labels=LABELS), 4),
    }

    out = {
        "n_expert_labels": len(expert),
        "expert_distribution": dict(expert_dist),
        "classifier_vs_expert": classifier_metrics,
        "pairwise_classifier_kappa": pairwise,
        "methodology_note": (
            "490 reviews labeled by lead author (domain expert). The three "
            "classifiers (V2 LLM, cleanlab-corrected pipeline, V5) act as "
            "independent annotators against this gold standard. Cohen's kappa "
            "is reported per classifier vs expert and pairwise between classifiers."
        ),
    }
    with open(out_dir / "gold_standard_results.json", "w") as f:
        json.dump(out, f, indent=2)

    # Human-readable report
    lines = []
    lines.append("=" * 78)
    lines.append("EXPERT GOLD-STANDARD VALIDATION (n=490 reviews)")
    lines.append("=" * 78)
    lines.append("")
    lines.append("METHODOLOGY:")
    lines.append("  490 reviews labeled by lead author (domain expert) using a stratified")
    lines.append("  sample (70 per class × 7 classes). Three classifiers serve as")
    lines.append("  independent annotators against this gold standard.")
    lines.append("")
    lines.append("EXPERT LABEL DISTRIBUTION:")
    for l in LABELS:
        lines.append(f"  {l:18s} {expert_dist.get(l, 0):>4}")
    lines.append("")
    lines.append("=" * 78)
    lines.append("CLASSIFIER vs EXPERT (Cohen's κ — primary reliability metric)")
    lines.append("=" * 78)
    lines.append(f"{'classifier':30s} {'n':>5} {'accuracy':>10} {'cohen_κ':>10} {'macro_f1':>10}")
    lines.append("-" * 78)
    for name, m in classifier_metrics.items():
        if m:
            lines.append(f"{name:30s} {m['n_aligned']:>5} "
                         f"{m['accuracy']:>10.4f} {m['cohen_kappa']:>10.4f} "
                         f"{m['macro_f1']:>10.4f}")
    lines.append("")
    lines.append("Interpretation of Cohen's κ:")
    lines.append("  > 0.81  almost perfect agreement")
    lines.append("  0.61-0.80  substantial agreement")
    lines.append("  0.41-0.60  moderate agreement")
    lines.append("  0.21-0.40  fair agreement")
    lines.append("  < 0.20  slight agreement")
    lines.append("")
    lines.append("=" * 78)
    lines.append("PER-CLASS F1 vs EXPERT")
    lines.append("=" * 78)
    lines.append(f"{'class':18s} " + " ".join(f"{n:>22s}" for n in classifier_metrics.keys()))
    for l in LABELS:
        row = f"{l:18s} "
        for name in classifier_metrics:
            m = classifier_metrics[name]
            f1 = m["per_class_f1"].get(l, 0.0) if m else 0.0
            sup = m["per_class_support_in_expert"].get(l, 0) if m else 0
            row += f"{f1:>15.3f} (n={sup:>3}) "
        lines.append(row)
    lines.append("")
    lines.append("=" * 78)
    lines.append("PAIRWISE COHEN κ (between classifiers, independent of expert)")
    lines.append("=" * 78)
    for k, v in pairwise.items():
        lines.append(f"  {k:30s} κ = {v:.4f}")
    lines.append("")

    report_text = "\n".join(lines)
    print("\n" + report_text)
    with open(out_dir / "gold_standard_report.txt", "w") as f:
        f.write(report_text)

    print(f"\nSaved: {out_dir/'gold_standard_results.json'}")
    print(f"       {out_dir/'gold_standard_report.txt'}")


if __name__ == "__main__":
    main()
