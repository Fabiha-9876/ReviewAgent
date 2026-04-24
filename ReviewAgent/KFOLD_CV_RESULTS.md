# 5-Fold Stratified Cross-Validation Results

**Date:** April 1, 2026
**Model:** RoBERTa-base (multi-label classification)
**Dataset:** MAALEJ labeled (5,008) + Synthetic (500) = 5,508 total samples
**Stratification:** StratifiedKFold on primary label, shuffle=True, random_state=42
**Training:** 3 epochs per fold, batch_size=8, lr=2e-5, warmup_ratio=0.1, weight_decay=0.01

---

## Dataset Distribution

| Label | Total | % of Dataset |
|-------|-------|-------------|
| praise | 2,447 | 44.4% |
| bug_report | 1,010 | 18.3% |
| other | 704 | 12.8% |
| usability | 692 | 12.6% |
| feature_request | 535 | 9.7% |
| performance | 70 | 1.3% |
| compatibility | 50 | 0.9% |

---

## Per-Fold Train/Val Label Distributions

| Label | Fold 1 Train | Fold 1 Val | Fold 2 Train | Fold 2 Val | Fold 3 Train | Fold 3 Val | Fold 4 Train | Fold 4 Val | Fold 5 Train | Fold 5 Val |
|-------|-------------|-----------|-------------|-----------|-------------|-----------|-------------|-----------|-------------|-----------|
| bug_report | 808 | 202 | 808 | 202 | 808 | 202 | 808 | 202 | 808 | 202 |
| feature_request | 428 | 107 | 428 | 107 | 428 | 107 | 428 | 107 | 428 | 107 |
| performance | 56 | 14 | 56 | 14 | 56 | 14 | 56 | 14 | 56 | 14 |
| usability | 554 | 138 | 554 | 138 | 553 | 139 | 553 | 139 | 554 | 138 |
| compatibility | 40 | 10 | 40 | 10 | 40 | 10 | 40 | 10 | 40 | 10 |
| praise | 1,957 | 490 | 1,957 | 490 | 1,958 | 489 | 1,958 | 489 | 1,958 | 489 |
| other | 563 | 141 | 563 | 141 | 563 | 141 | 564 | 140 | 563 | 141 |
| **Total** | **4,406** | **1,102** | **4,406** | **1,102** | **4,406** | **1,102** | **4,407** | **1,101** | **4,407** | **1,101** |

---

## Per-Fold Overall Metrics

| Fold | F1 Micro | F1 Macro | Precision | Recall | Training Loss |
|------|----------|----------|-----------|--------|---------------|
| 1 | 0.7568 | 0.7985 | 0.7840 | 0.7314 | 0.1883 |
| 2 | 0.7723 | 0.8054 | 0.7914 | 0.7541 | 0.1921 |
| 3 | 0.7522 | 0.7928 | 0.7712 | 0.7341 | 0.1859 |
| 4 | 0.7779 | 0.7997 | 0.7895 | 0.7666 | 0.1896 |
| 5 | 0.7634 | 0.7904 | 0.7770 | 0.7502 | 0.1878 |
| **Mean** | **0.7645** | **0.7974** | **0.7827** | **0.7473** | — |
| **Std** | **0.0095** | **0.0053** | **0.0076** | **0.0131** | — |

---

## Per-Fold Per-Label F1 Scores

| Label | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|-------|--------|--------|--------|--------|--------|
| bug_report | 0.8276 | 0.8200 | 0.7848 | 0.8090 | 0.7888 |
| feature_request | 0.6473 | 0.6768 | 0.6697 | 0.6537 | 0.6301 |
| performance | 1.0000 | 0.9655 | 1.0000 | 1.0000 | 1.0000 |
| usability | 0.4835 | 0.5328 | 0.5000 | 0.4754 | 0.5000 |
| compatibility | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| praise | 0.7953 | 0.8136 | 0.8025 | 0.8399 | 0.8394 |
| other | 0.8358 | 0.8288 | 0.7927 | 0.8000 | 0.7742 |

---

## Per-Label Average Across 5 Folds

| Label | Mean F1 | Std F1 | Mean Precision | Mean Recall | Avg Support |
|-------|---------|--------|---------------|-------------|-------------|
| bug_report | 0.8101 | 0.0193 | 0.8167 | 0.8040 | 202.0 |
| feature_request | 0.6555 | 0.0165 | 0.6710 | 0.6430 | 107.0 |
| performance | 0.9931 | 0.0138 | 0.9867 | 1.0000 | 14.0 |
| usability | 0.4984 | 0.0197 | 0.5296 | 0.4726 | 138.4 |
| compatibility | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 10.0 |
| praise | 0.8181 | 0.0185 | 0.8393 | 0.7990 | 489.4 |
| other | 0.8063 | 0.0229 | 0.8220 | 0.7926 | 140.8 |

---

## Comparison to Previous Single-Split (Run 2)

| Metric | Single 80/20 Split | 5-Fold CV (Mean) |
|--------|-------------------|------------------|
| F1 Micro | 0.7671 | 0.7645 |
| F1 Macro | 0.7992 | 0.7974 |
| Precision | 0.7915 | 0.7827 |
| Recall | 0.7441 | 0.7473 |
