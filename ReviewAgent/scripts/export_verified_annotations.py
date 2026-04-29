"""
Extract verified annotations from Apple Numbers files into a canonical JSON.

Input:  three .numbers files where rows have columns
        id, text, rating, predicted_label, confidence, needs_hitl, app_id,
        correct_yn (Y/N), correct_label_if_no, comments

Final label = predicted_label if correct_yn == 'Y'
              correct_label_if_no (normalized) if correct_yn == 'N'

Output: data/processed/verified_annotations.json
        List of {text, labels: [final_label], original_llm_label, source, rating, app_id}

Usage:
    python3 scripts/export_verified_annotations.py
"""

import json
import sys
from pathlib import Path
from collections import Counter

from numbers_parser import Document

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.stage1.classifier import LABELS

# Map human-readable labels in the Numbers files to canonical LABELS
LABEL_NORMALIZE = {
    "bug report": "bug_report",
    "bug_report": "bug_report",
    "feature request": "feature_request",
    "feature_request": "feature_request",
    "performance": "performance",
    "usability": "usability",
    "compatibility": "compatibility",
    "praise": "praise",
    "other": "other",
}


def normalize(label: str | None) -> str | None:
    if label is None:
        return None
    return LABEL_NORMALIZE.get(str(label).strip().lower())


SOURCES = [
    "/Users/fabihajalal/Desktop/Review Agent/RRGen_Annotation/compatibility.numbers",
    "/Users/fabihajalal/Desktop/Review Agent/RRGen_Annotation/performance.numbers",
    "/Users/fabihajalal/Desktop/Review Agent/RRGen_Annotation/praise.numbers",
]


def main():
    out_path = Path("data/processed/verified_annotations.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    skipped_bad_label = []
    per_file_stats = []

    for path in SOURCES:
        doc = Document(path)
        for sheet in doc.sheets:
            for tbl in sheet.tables:
                rows = tbl.rows(values_only=True)
                header = rows[0]
                col = {name: i for i, name in enumerate(header)}
                file_annotated = 0
                file_y = 0
                file_n = 0
                for r in rows[1:]:
                    yn = r[col["correct_yn"]]
                    if yn is None or str(yn).strip() == "":
                        continue
                    yn_str = str(yn).strip().upper()
                    text = r[col["text"]]
                    if text is None or not str(text).strip():
                        continue
                    llm_label = normalize(r[col["predicted_label"]])

                    if yn_str == "Y":
                        final = llm_label
                        file_y += 1
                    elif yn_str == "N":
                        corrected = normalize(r[col["correct_label_if_no"]])
                        if corrected is None or corrected not in LABELS:
                            skipped_bad_label.append({
                                "file": Path(path).name,
                                "id": r[col["id"]],
                                "raw_correction": r[col["correct_label_if_no"]],
                                "text": str(text)[:80],
                            })
                            continue
                        final = corrected
                        file_n += 1
                    else:
                        continue

                    if final not in LABELS:
                        continue

                    records.append({
                        "text": str(text),
                        "labels": [final],
                        "original_llm_label": llm_label,
                        "rating": r[col["rating"]],
                        "app_id": r[col["app_id"]],
                        "confidence": r[col["confidence"]],
                        "source": Path(path).stem,
                    })
                    file_annotated += 1

                per_file_stats.append({
                    "file": Path(path).name,
                    "annotated": file_annotated,
                    "confirmed_Y": file_y,
                    "corrected_N": file_n,
                })

    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    # Summary
    print(f"Exported {len(records):,} verified annotations to {out_path}\n")
    for s in per_file_stats:
        print(f"  {s['file']:30s} annotated={s['annotated']:5d}  Y={s['confirmed_Y']}  N={s['corrected_N']}")
    print()

    label_dist = Counter(r["labels"][0] for r in records)
    print("Final label distribution:")
    for lbl in LABELS:
        n = label_dist.get(lbl, 0)
        bar = "#" * (n * 40 // max(label_dist.values()) if label_dist else 0)
        print(f"  {lbl:20s} {n:5d}  {bar}")

    if skipped_bad_label:
        print(f"\nSkipped {len(skipped_bad_label)} rows with unrecognized correction labels:")
        for s in skipped_bad_label[:10]:
            print(f"  {s}")


if __name__ == "__main__":
    main()
