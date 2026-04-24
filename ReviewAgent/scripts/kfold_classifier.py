"""
K-Fold Cross Validation for RoBERTa Classifier
================================================

Runs 5-fold stratified cross validation on MAALEJ labeled + synthetic data.
Reports per-fold and average F1 scores for each label.

Usage:
    python3 scripts/kfold_classifier.py              # 5-fold (default)
    python3 scripts/kfold_classifier.py --folds 10   # 10-fold

Output:
    models/stage1_classifier/kfold_results.json
    Prints per-fold table + average scores
"""

import sys
import json
import random
import argparse
import warnings
import logging
import os
from pathlib import Path
from datetime import datetime
from collections import Counter
from dataclasses import dataclass

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.stage1.classifier import LABELS


# ============================================================
# Data Preparation
# ============================================================

def labels_to_vector(label_list):
    vector = [0] * len(LABELS)
    for label in label_list:
        if label in LABELS:
            vector[LABELS.index(label)] = 1
    if sum(vector) == 0:
        vector[LABELS.index("other")] = 1
    return vector


def load_data():
    """Load MAALEJ labeled + synthetic data."""
    all_texts = []
    all_labels = []
    all_primary = []  # For stratification

    # MAALEJ labeled
    maalej = json.loads(Path("data/raw/maalej/maalej_labeled.json").read_text())
    for r in maalej:
        all_texts.append(r["text"])
        vec = labels_to_vector(r["labels"])
        all_labels.append(vec)
        all_primary.append(vec.index(1) if 1 in vec else 6)

    # Synthetic
    synth = json.loads(Path("data/raw/sample_reviews.json").read_text())
    for r in synth:
        all_texts.append(r["text"])
        vec = labels_to_vector(r["labels"])
        all_labels.append(vec)
        all_primary.append(vec.index(1) if 1 in vec else 6)

    return all_texts, all_labels, all_primary


# ============================================================
# Custom Collator
# ============================================================

from dataclasses import dataclass as dc

@dc
class MultiLabelCollator:
    tokenizer: object
    def __call__(self, features):
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.tensor(
                [f["labels"].tolist() for f in features], dtype=torch.float32
            ),
        }
        return batch


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="micro", zero_division=0),
        "recall": recall_score(labels, preds, average="micro", zero_division=0),
    }


# ============================================================
# K-Fold Cross Validation
# ============================================================

def run_kfold(n_folds=5, epochs=3, batch_size=8, lr=2e-5, model_name="roberta-base"):
    """Run stratified k-fold cross validation."""

    print("=" * 70)
    print(f"K-FOLD CROSS VALIDATION (k={n_folds})")
    print("=" * 70)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Model: {model_name}")
    print(f"  Epochs per fold: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")

    # Load data
    print("\n--- Loading Data ---")
    texts, labels, primary_labels = load_data()
    print(f"  Total samples: {len(texts)}")
    dist = Counter(primary_labels)
    for idx in sorted(dist.keys()):
        print(f"    {LABELS[idx]}: {dist[idx]}")

    # Tokenizer (shared across folds)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=128
        )
        tokens["labels"] = [[float(x) for x in l] for l in examples["labels"]]
        return tokens

    # K-Fold splitter
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    texts_arr = np.array(texts)
    labels_arr = np.array(labels)
    primary_arr = np.array(primary_labels)

    # Storage for results
    fold_results = []
    all_val_preds = []
    all_val_labels = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts_arr, primary_arr)):
        print(f"\n{'=' * 70}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'=' * 70}")
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

        # Split
        train_texts = texts_arr[train_idx].tolist()
        val_texts = texts_arr[val_idx].tolist()
        train_labels = labels_arr[train_idx].tolist()
        val_labels = labels_arr[val_idx].tolist()

        # Print fold distribution
        train_dist = Counter(primary_arr[train_idx].tolist())
        val_dist = Counter(primary_arr[val_idx].tolist())
        print(f"  Train distribution: { {LABELS[k]: v for k, v in sorted(train_dist.items())} }")
        print(f"  Val distribution:   { {LABELS[k]: v for k, v in sorted(val_dist.items())} }")

        # Tokenize
        train_ds = Dataset.from_dict({"text": train_texts, "labels": train_labels})
        val_ds = Dataset.from_dict({"text": val_texts, "labels": val_labels})
        train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"]).with_format("torch")
        val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"]).with_format("torch")

        # Fresh model for each fold
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(LABELS), problem_type="multi_label_classification"
        )

        output_dir = f"models/stage1_classifier/kfold/fold_{fold_idx + 1}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=16,
            eval_strategy="epoch",
            save_strategy="no",
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_steps=500,
            fp16=False,
            dataloader_pin_memory=False,
            report_to="none",
            disable_tqdm=True,
        )

        collator = MultiLabelCollator(tokenizer=tokenizer)
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            data_collator=collator,
        )

        # Train
        print(f"  Training...")
        result = trainer.train()
        print(f"  Training loss: {result.training_loss:.4f}")

        # Evaluate
        eval_result = trainer.evaluate()
        print(f"  F1 micro: {eval_result['eval_f1_micro']:.4f}")
        print(f"  F1 macro: {eval_result['eval_f1_macro']:.4f}")

        # Per-label F1
        preds_out = trainer.predict(val_ds)
        probs = 1 / (1 + np.exp(-preds_out.predictions))
        preds = (probs > 0.5).astype(int)
        val_labels_np = np.array(val_labels)

        per_label = {}
        print(f"\n  {'Label':<18} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
        print(f"  {'-' * 54}")
        for i, label in enumerate(LABELS):
            sup = int(val_labels_np[:, i].sum())
            if sup == 0:
                per_label[label] = {"precision": 0, "recall": 0, "f1": 0, "support": 0}
                print(f"  {label:<18} {'N/A':>8} {'N/A':>8} {'N/A':>8} {0:>8}")
                continue
            p = precision_score(val_labels_np[:, i], preds[:, i], zero_division=0)
            r = recall_score(val_labels_np[:, i], preds[:, i], zero_division=0)
            f = f1_score(val_labels_np[:, i], preds[:, i], zero_division=0)
            per_label[label] = {"precision": float(p), "recall": float(r), "f1": float(f), "support": sup}
            print(f"  {label:<18} {p:>8.4f} {r:>8.4f} {f:>8.4f} {sup:>8}")

        fold_results.append({
            "fold": fold_idx + 1,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "training_loss": float(result.training_loss),
            "f1_micro": float(eval_result["eval_f1_micro"]),
            "f1_macro": float(eval_result["eval_f1_macro"]),
            "precision": float(eval_result["eval_precision"]),
            "recall": float(eval_result["eval_recall"]),
            "per_label": per_label,
        })

        # Collect for overall
        all_val_preds.extend(preds.tolist())
        all_val_labels.extend(val_labels)

        # Clean up model to free memory
        del model, trainer
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n\n{'=' * 70}")
    print(f"K-FOLD CROSS VALIDATION SUMMARY (k={n_folds})")
    print(f"{'=' * 70}")

    # Per-fold summary
    print(f"\n{'Fold':>6} {'F1 micro':>10} {'F1 macro':>10} {'Precision':>10} {'Recall':>10} {'Loss':>10}")
    print(f"{'-' * 58}")
    for fr in fold_results:
        print(f"{fr['fold']:>6} {fr['f1_micro']:>10.4f} {fr['f1_macro']:>10.4f} "
              f"{fr['precision']:>10.4f} {fr['recall']:>10.4f} {fr['training_loss']:>10.4f}")

    # Averages
    avg_micro = np.mean([fr["f1_micro"] for fr in fold_results])
    std_micro = np.std([fr["f1_micro"] for fr in fold_results])
    avg_macro = np.mean([fr["f1_macro"] for fr in fold_results])
    std_macro = np.std([fr["f1_macro"] for fr in fold_results])
    avg_prec = np.mean([fr["precision"] for fr in fold_results])
    std_prec = np.std([fr["precision"] for fr in fold_results])
    avg_rec = np.mean([fr["recall"] for fr in fold_results])
    std_rec = np.std([fr["recall"] for fr in fold_results])

    print(f"{'':>6} {'─' * 48}")
    print(f"{'Mean':>6} {avg_micro:>10.4f} {avg_macro:>10.4f} {avg_prec:>10.4f} {avg_rec:>10.4f}")
    print(f"{'Std':>6} {std_micro:>10.4f} {std_macro:>10.4f} {std_prec:>10.4f} {std_rec:>10.4f}")

    # Per-label average across folds
    print(f"\n\nPer-Label Average F1 Across {n_folds} Folds:")
    print(f"  {'Label':<18} {'Mean F1':>10} {'Std F1':>10} {'Mean Prec':>10} {'Mean Rec':>10} {'Avg Support':>12}")
    print(f"  {'-' * 72}")
    for label in LABELS:
        f1s = [fr["per_label"][label]["f1"] for fr in fold_results]
        precs = [fr["per_label"][label]["precision"] for fr in fold_results]
        recs = [fr["per_label"][label]["recall"] for fr in fold_results]
        sups = [fr["per_label"][label]["support"] for fr in fold_results]
        print(f"  {label:<18} {np.mean(f1s):>10.4f} {np.std(f1s):>10.4f} "
              f"{np.mean(precs):>10.4f} {np.mean(recs):>10.4f} {np.mean(sups):>12.1f}")

    # Overall (all predictions pooled)
    all_preds_np = np.array(all_val_preds)
    all_labels_np = np.array(all_val_labels)
    overall_micro = f1_score(all_labels_np, all_preds_np, average="micro", zero_division=0)
    overall_macro = f1_score(all_labels_np, all_preds_np, average="macro", zero_division=0)

    print(f"\n\nOverall (all folds pooled):")
    print(f"  F1 micro: {overall_micro:.4f}")
    print(f"  F1 macro: {overall_macro:.4f}")

    # Save results
    results = {
        "config": {
            "n_folds": n_folds,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "model": model_name,
            "total_samples": len(texts),
            "date": datetime.now().isoformat(),
        },
        "summary": {
            "f1_micro_mean": float(avg_micro),
            "f1_micro_std": float(std_micro),
            "f1_macro_mean": float(avg_macro),
            "f1_macro_std": float(std_macro),
            "precision_mean": float(avg_prec),
            "precision_std": float(std_prec),
            "recall_mean": float(avg_rec),
            "recall_std": float(std_rec),
            "overall_f1_micro": float(overall_micro),
            "overall_f1_macro": float(overall_macro),
        },
        "per_label_average": {
            label: {
                "f1_mean": float(np.mean([fr["per_label"][label]["f1"] for fr in fold_results])),
                "f1_std": float(np.std([fr["per_label"][label]["f1"] for fr in fold_results])),
                "precision_mean": float(np.mean([fr["per_label"][label]["precision"] for fr in fold_results])),
                "recall_mean": float(np.mean([fr["per_label"][label]["recall"] for fr in fold_results])),
                "avg_support": float(np.mean([fr["per_label"][label]["support"] for fr in fold_results])),
            }
            for label in LABELS
        },
        "per_fold": fold_results,
    }

    out_path = Path("models/stage1_classifier/kfold_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n\nResults saved to: {out_path}")
    print("DONE!")

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    run_kfold(n_folds=args.folds, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
