"""
Train a RoBERTa anchor classifier on the combined verified + MAALEJ dataset.

Purpose: replace the TF-IDF + LogReg anchor (macro-F1 0.53) with a stronger
RoBERTa anchor for the second cleanlab correction pass on the 215K RRGen set.

Input:
    data/processed/anchor_combined.json  (10,238 samples: 5,230 verified + 5,008 MAALEJ)

Usage:
    python3 scripts/train_anchor_roberta.py

Output:
    models/anchor_roberta/
        model.safetensors, tokenizer.json, config.json
        eval_metrics.json     held-out test metrics
        split_info.json
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.stage1.classifier import LABELS


def to_multilabel(label: str) -> list[int]:
    vec = [0] * len(LABELS)
    vec[LABELS.index(label)] = 1
    return vec


class WeightedBCETrainer(Trainer):
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = probs.argmax(axis=1)
    true = labels.argmax(axis=1)
    p, r, f1, _ = precision_recall_fscore_support(true, preds, average="macro", zero_division=0)
    f1_micro = f1_score(true, preds, average="micro", zero_division=0)
    return {
        "macro_f1": float(f1),
        "micro_f1": float(f1_micro),
        "macro_precision": float(p),
        "macro_recall": float(r),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", type=Path,
                    default=Path("data/processed/anchor_combined.json"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("models/anchor_roberta"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--pos-weight-cap", type=float, default=20.0,
                    help="Cap pos_weight per class (prevents the tiny-class blow-up we saw in V3).")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    print(f"Loading anchor data: {args.anchor}")
    with open(args.anchor) as f:
        data = json.load(f)
    texts = [r["text"] for r in data]
    labels_str = [r["labels"][0] for r in data]
    print(f"  {len(data):,} samples")
    dist = Counter(labels_str)
    print(f"  distribution: {dict(dist)}")

    # 80/10/10 stratified
    train_idx, tmp_idx = train_test_split(
        range(len(data)), test_size=0.20, stratify=labels_str, random_state=args.seed
    )
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=0.50,
        stratify=[labels_str[i] for i in tmp_idx], random_state=args.seed
    )
    print(f"Split: train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")

    # Class weights (capped)
    label_counts = np.array([dist.get(l, 1) for l in LABELS], dtype=np.float32)
    raw_weights = len(data) / (len(LABELS) * label_counts)
    pos_weight = torch.tensor(np.minimum(raw_weights, args.pos_weight_cap), dtype=torch.float32)
    print(f"  pos_weights (capped @ {args.pos_weight_cap}): {dict(zip(LABELS, pos_weight.tolist()))}")

    def build_ds(idxs):
        return Dataset.from_dict({
            "text": [texts[i] for i in idxs],
            "labels": [to_multilabel(labels_str[i]) for i in idxs],
        })

    train_ds = build_ds(train_idx)
    val_ds = build_ds(val_idx)
    test_ds = build_ds(test_idx)

    print("Loading tokenizer + model (roberta-base)")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
    )

    def tokenize(examples):
        tok = tokenizer(examples["text"], padding="max_length",
                        truncation=True, max_length=args.max_length)
        tok["labels"] = [[float(x) for x in lv] for lv in examples["labels"]]
        return tok

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])
    for d in (train_ds, val_ds, test_ds):
        d.set_format("torch")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )

    trainer = WeightedBCETrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        pos_weight=pos_weight,
    )

    print("\n" + "=" * 70)
    print("TRAINING RoBERTa ANCHOR")
    print("=" * 70)
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f"\nTraining time: {train_time/3600:.2f}h")

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    print("\n" + "=" * 70)
    print("TEST-SET EVALUATION")
    print("=" * 70)
    pred_output = trainer.predict(test_ds)
    probs = 1 / (1 + np.exp(-pred_output.predictions))
    y_pred = probs.argmax(axis=1)
    y_true = np.array([LABELS.index(labels_str[i]) for i in test_idx])

    report_dict = classification_report(y_true, y_pred, target_names=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    print(classification_report(y_true, y_pred, target_names=LABELS, zero_division=0))

    metrics_out = {
        "train_time_hours": train_time / 3600,
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "data_distribution": dict(dist),
        "test_classification_report": report_dict,
        "test_confusion_matrix": cm.tolist(),
        "labels": LABELS,
        "hyperparams": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "max_length": args.max_length,
            "pos_weight_cap": args.pos_weight_cap, "seed": args.seed,
        },
    }
    with open(args.output_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    with open(args.output_dir / "split_info.json", "w") as f:
        json.dump({"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}, f)

    print(f"\nMacro F1 (test): {report_dict['macro avg']['f1-score']:.4f}")
    print(f"Micro F1 (test): {f1_score(y_true, y_pred, average='micro'):.4f}")
    print(f"Model saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
