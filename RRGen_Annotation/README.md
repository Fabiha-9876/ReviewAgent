# RRGen_Annotation/

This folder originally held the per-category CSV exports of LLM-labeled RRGen reviews (created early in the project for verification by volunteers). Those CSVs (`all_reviews.csv`, `bug_report.csv`, `feature_request.csv`, `other.csv`, `usability.csv`, `annotation_summary.xlsx`) have been **removed**: they were unannotated and superseded by the actual verification work captured in the `.numbers` files below.

## Current contents

| file | description |
|---|---|
| `compatibility.numbers` | 8 LLM-predicted compatibility reviews + lead-author Y/N verification (8/8 verified, 4 corrected to Performance) |
| `performance.numbers` | 184 LLM-predicted performance reviews + verification (183 Y, 1 N) |
| `praise.numbers` | 5,041 LLM-predicted praise reviews + verification (3,736 Y, 1,305 N → corrected to other categories) |
| `Synthetic_Data_Validation_Form_500.numbers` | Self-validation form for the 500 synthetic reviews used in V1 training |

These three `.numbers` files together constitute the **5,230-review verified anchor** used in the cleanlab + verified-anchor correction pipeline (see `paper/02_methodology.md` Section 3.3 and `data/processed/verified_annotations.json`).

## To regenerate the CSV exports if needed

```
python3 scripts/prepare_full_annotation.py
```

This recreates the per-category CSVs from `data/processed/rrgen_full_labeled/rrgen_full_labeled.json`.
