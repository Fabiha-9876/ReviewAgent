"""
Fine-tune RoBERTa Classifier for App Review Classification
============================================================

This script:
1. Loads labeled data (synthetic 500 + auto-labeled MAALEJ subset)
2. Auto-labels unlabeled MAALEJ reviews using keyword heuristics
3. Splits into train/val (80/20)
4. Fine-tunes RoBERTa for multi-label classification
5. Evaluates with F1, precision, recall per label
6. Saves the model for pipeline use

Usage:
    python3 scripts/train_classifier.py                    # Full training
    python3 scripts/train_classifier.py --epochs 3         # Quick training
    python3 scripts/train_classifier.py --auto-label-only  # Just label data, no training

Output:
    models/stage1_classifier/  — saved model + tokenizer
    data/processed/labeled_reviews.json — all labeled training data
"""

import json
import sys
import os
import random
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stage1.classifier import LABELS


# ============================================================
# Step 1: Keyword-based Auto-Labeling for Unlabeled Reviews
# ============================================================

# Keyword patterns for each category
LABEL_KEYWORDS = {
    "bug_report": [
        "crash", "crashes", "crashing", "bug", "broken", "not working", "doesn't work",
        "doesnt work", "won't open", "wont open", "force close", "error", "freeze",
        "freezes", "freezing", "stuck", "glitch", "glitchy", "black screen",
        "white screen", "blank screen", "can't open", "cant open", "not loading",
        "fails", "failed", "failure", "unresponsive", "hang", "hangs", "hanging",
        "stopped working", "not responding", "keeps closing", "shuts down",
    ],
    "feature_request": [
        "wish", "would be nice", "please add", "should add", "could you add",
        "need feature", "want feature", "missing feature", "add option",
        "suggestion", "suggest", "would love", "it would be great",
        "hope you can", "request", "requesting", "implement", "bring back",
        "add support", "add ability", "need option", "want option",
        "dark mode", "widget", "offline", "export", "import", "customize",
    ],
    "performance": [
        "slow", "laggy", "lag", "lags", "lagging", "takes forever", "battery drain",
        "battery", "drains battery", "heats up", "heating", "hot", "overheating",
        "memory", "ram", "storage", "too much space", "heavy", "resource",
        "loading time", "takes long", "takes too long", "speed", "sluggish",
        "seconds to load", "minutes to load", "consumes", "consumption",
    ],
    "usability": [
        "confusing", "confused", "hard to use", "hard to find", "not intuitive",
        "complicated", "complex", "difficult", "user friendly", "ux", "ui",
        "interface", "design", "layout", "navigation", "navigate", "menu",
        "button", "buttons too small", "text too small", "can't find",
        "where is", "how do i", "how to", "unintuitive", "clunky",
        "ugly", "looks bad", "looks terrible", "awful design",
    ],
    "compatibility": [
        "doesn't work on", "not compatible", "incompatible", "not supported",
        "my device", "my phone", "my tablet", "screen size", "resolution",
        "android version", "ios version", "older device", "newer device",
        "samsung", "pixel", "iphone", "ipad", "specific device",
        "only works on", "doesn't support", "tablet mode",
    ],
    "praise": [
        "love", "great", "awesome", "amazing", "excellent", "fantastic",
        "perfect", "best app", "wonderful", "brilliant", "superb",
        "love it", "love this", "thank you", "thanks", "good job",
        "well done", "keep it up", "5 stars", "five stars", "recommend",
        "highly recommend", "must have", "beautiful", "smooth", "works great",
        "works perfectly", "no issues", "no problems", "flawless",
    ],
}


def auto_label_review(text: str, rating: int) -> list[str]:
    """Assign labels to a review using keyword heuristics + rating signals."""
    text_lower = text.lower()
    matched_labels = []

    # Keyword matching
    for label, keywords in LABEL_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                matched_labels.append(label)
                break

    # Rating-based boost
    if not matched_labels:
        if rating <= 2:
            # Low rating without specific keywords → likely a bug or general complaint
            if any(w in text_lower for w in ["crash", "error", "bug", "broken", "fix"]):
                matched_labels.append("bug_report")
            elif any(w in text_lower for w in ["slow", "battery", "lag"]):
                matched_labels.append("performance")
            else:
                matched_labels.append("bug_report")  # Default for low-rating
        elif rating >= 4:
            matched_labels.append("praise")
        else:
            matched_labels.append("other")

    return list(set(matched_labels))


def labels_to_vector(label_list: list[str]) -> list[int]:
    """Convert label names to multi-hot vector."""
    vector = [0] * len(LABELS)
    for label in label_list:
        if label in LABELS:
            idx = LABELS.index(label)
            vector[idx] = 1
    # If no valid label matched, set "other"
    if sum(vector) == 0:
        vector[LABELS.index("other")] = 1
    return vector


# ============================================================
# Step 2: Data Preparation
# ============================================================

def prepare_training_data(max_maalej: int = 10000) -> tuple[list[str], list[list[int]]]:
    """Prepare labeled training data from all available sources."""

    all_texts = []
    all_labels = []
    source_counts = Counter()

    # Source 1: Synthetic data (500 reviews, already labeled)
    synth_path = Path("data/raw/sample_reviews.json")
    if synth_path.exists():
        synth = json.loads(synth_path.read_text())
        for r in synth:
            all_texts.append(r["text"])
            all_labels.append(labels_to_vector(r["labels"]))
            source_counts["synthetic"] += 1
        print(f"  Loaded {len(synth)} synthetic labeled reviews")

    # Source 2: MAALEJ data (auto-labeled using keywords)
    maalej_path = Path("data/raw/maalej/maalej_reviews.json")
    if maalej_path.exists():
        maalej = json.loads(maalej_path.read_text())
        random.shuffle(maalej)

        labeled_count = 0
        for r in maalej[:max_maalej]:
            text = r.get("text", "").strip()
            if not text or len(text) < 10:
                continue
            rating = r.get("rating", 3)
            labels = auto_label_review(text, rating)
            all_texts.append(text)
            all_labels.append(labels_to_vector(labels))
            source_counts["maalej_auto"] += 1
            labeled_count += 1

        print(f"  Auto-labeled {labeled_count} MAALEJ reviews (from {len(maalej)} total)")

    # Source 3: RRGen reviews (auto-labeled, for additional volume)
    rrgen_path = Path("data/raw/rrgen/rrgen_reviews.json")
    if rrgen_path.exists():
        rrgen = json.loads(rrgen_path.read_text())
        random.shuffle(rrgen)

        rrgen_count = 0
        for r in rrgen[:5000]:  # Take 5K from RRGen
            text = r.get("text", "").strip()
            if not text or len(text) < 10:
                continue
            rating = r.get("rating", 3)
            labels = auto_label_review(text, rating)
            all_texts.append(text)
            all_labels.append(labels_to_vector(labels))
            source_counts["rrgen_auto"] += 1
            rrgen_count += 1

        print(f"  Auto-labeled {rrgen_count} RRGen reviews (from {len(rrgen)} total)")

    print(f"\n  Total training data: {len(all_texts)} reviews")
    print(f"  Sources: {dict(source_counts)}")

    # Print label distribution
    label_counts = Counter()
    for vec in all_labels:
        for i, v in enumerate(vec):
            if v == 1:
                label_counts[LABELS[i]] += 1
    print(f"  Label distribution:")
    for label in LABELS:
        count = label_counts.get(label, 0)
        pct = count / len(all_texts) * 100 if all_texts else 0
        print(f"    {label}: {count} ({pct:.1f}%)")

    # Save labeled data
    labeled_data = [
        {"text": t, "labels": [LABELS[i] for i, v in enumerate(l) if v == 1], "label_vector": l}
        for t, l in zip(all_texts, all_labels)
    ]
    Path("data/processed/labeled_reviews.json").write_text(json.dumps(labeled_data[:100], indent=2))
    print(f"\n  Sample saved to: data/processed/labeled_reviews.json")

    return all_texts, all_labels


# ============================================================
# Step 3: Custom Metrics for Evaluation
# ============================================================

def compute_metrics(eval_pred):
    """Compute multi-label F1, precision, recall during training."""
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs > 0.5).astype(int)

    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    precision = precision_score(labels, preds, average="micro", zero_division=0)
    recall = recall_score(labels, preds, average="micro", zero_division=0)

    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "precision": precision,
        "recall": recall,
    }


# ============================================================
# Step 4: Training
# ============================================================

def train_model(
    texts: list[str],
    labels: list[list[int]],
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
    model_name: str = "roberta-base",
    output_dir: str = "models/stage1_classifier",
):
    """Fine-tune RoBERTa on the labeled data."""
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )
    from datasets import Dataset

    print(f"\n{'=' * 60}")
    print(f"TRAINING RoBERTa CLASSIFIER")
    print(f"{'=' * 60}")
    print(f"  Model: {model_name}")
    print(f"  Device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Labels: {LABELS}")
    print(f"  Training samples: {len(texts)}")

    # Train/val split (80/20)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(train_texts)}, Val: {len(val_texts)}")

    # Load model and tokenizer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
    )

    # Tokenize
    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )
        tokens["labels"] = [[float(x) for x in label] for label in examples["labels"]]
        return tokens

    train_ds = Dataset.from_dict({"text": train_texts, "labels": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "labels": val_labels})

    print("  Tokenizing...")
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    # Training arguments
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        logging_dir=f"{output_dir}/logs",
        save_total_limit=2,
        fp16=False,  # MPS doesn't support fp16 well
        dataloader_pin_memory=False,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    print("\n  Starting training...\n")
    result = trainer.train()

    # Save
    print(f"\n  Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Print training results
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Training loss: {result.training_loss:.4f}")
    for key, val in result.metrics.items():
        print(f"  {key}: {val}")

    return model, tokenizer, trainer


# ============================================================
# Step 5: Detailed Evaluation
# ============================================================

def evaluate_model(
    model,
    tokenizer,
    texts: list[str],
    labels: list[list[int]],
    device: str = "cpu",
):
    """Run detailed evaluation with per-label metrics."""

    print(f"\n{'=' * 60}")
    print("DETAILED EVALUATION")
    print(f"{'=' * 60}")

    model.eval()
    model.to(device)

    all_preds = []
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**tokens).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds.tolist())

    # Overall metrics
    labels_np = np.array(labels)
    preds_np = np.array(all_preds)

    print(f"\n  Overall Metrics:")
    print(f"    F1 (micro): {f1_score(labels_np, preds_np, average='micro', zero_division=0):.4f}")
    print(f"    F1 (macro): {f1_score(labels_np, preds_np, average='macro', zero_division=0):.4f}")
    print(f"    Precision (micro): {precision_score(labels_np, preds_np, average='micro', zero_division=0):.4f}")
    print(f"    Recall (micro): {recall_score(labels_np, preds_np, average='micro', zero_division=0):.4f}")

    # Per-label metrics
    print(f"\n  Per-Label Metrics:")
    print(f"    {'Label':<18} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support'}")
    print(f"    {'-'*66}")

    for i, label in enumerate(LABELS):
        if labels_np[:, i].sum() == 0:
            print(f"    {label:<18} {'N/A':<12} {'N/A':<12} {'N/A':<12} 0")
            continue
        p = precision_score(labels_np[:, i], preds_np[:, i], zero_division=0)
        r = recall_score(labels_np[:, i], preds_np[:, i], zero_division=0)
        f = f1_score(labels_np[:, i], preds_np[:, i], zero_division=0)
        support = int(labels_np[:, i].sum())
        print(f"    {label:<18} {p:<12.4f} {r:<12.4f} {f:<12.4f} {support}")

    # Save evaluation results
    eval_results = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(texts),
        "f1_micro": float(f1_score(labels_np, preds_np, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(labels_np, preds_np, average="macro", zero_division=0)),
        "per_label": {
            label: {
                "precision": float(precision_score(labels_np[:, i], preds_np[:, i], zero_division=0)),
                "recall": float(recall_score(labels_np[:, i], preds_np[:, i], zero_division=0)),
                "f1": float(f1_score(labels_np[:, i], preds_np[:, i], zero_division=0)),
                "support": int(labels_np[:, i].sum()),
            }
            for i, label in enumerate(LABELS)
        },
    }
    eval_path = Path("models/stage1_classifier/eval_results.json")
    eval_path.write_text(json.dumps(eval_results, indent=2))
    print(f"\n  Evaluation saved to: {eval_path}")

    return eval_results


# ============================================================
# Step 6: Quick Test with Sample Predictions
# ============================================================

def test_predictions(model_dir: str = "models/stage1_classifier"):
    """Test the trained model on a few example reviews."""
    from src.stage1.classifier import ReviewClassifier

    print(f"\n{'=' * 60}")
    print("SAMPLE PREDICTIONS")
    print(f"{'=' * 60}")

    classifier = ReviewClassifier(model_name_or_path=model_dir)

    test_reviews = [
        "App crashes every time I try to login. Please fix!",
        "I wish you had dark mode. Using the app at night hurts my eyes.",
        "The app is super slow and drains my battery in 2 hours.",
        "Love this app! Best photo editor ever. 5 stars!",
        "The checkout page is so confusing. Can't find the coupon field.",
        "Doesn't work on my Samsung Galaxy S24 with Android 15.",
        "Good app but needs some improvements to the search feature.",
    ]

    results = classifier.predict(test_reviews)

    for review, (labels, confidences) in zip(test_reviews, results):
        top_confs = sorted(confidences.items(), key=lambda x: x[1], reverse=True)[:3]
        needs_hitl = classifier.needs_hitl(confidences)
        print(f"\n  Review: \"{review[:70]}...\"")
        print(f"  Labels: {labels}")
        print(f"  Top confidences: {[(l, f'{c:.3f}') for l, c in top_confs]}")
        print(f"  Needs HITL: {needs_hitl}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa classifier")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-maalej", type=int, default=10000, help="Max MAALEJ reviews to use")
    parser.add_argument("--model", type=str, default="roberta-base", help="Base model name")
    parser.add_argument("--output-dir", type=str, default="models/stage1_classifier")
    parser.add_argument("--auto-label-only", action="store_true", help="Only prepare data, don't train")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("ReviewAgent — RoBERTa Classifier Fine-Tuning")
    print("=" * 60)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Device: {'MPS (Apple Silicon)' if torch.backends.mps.is_available() else 'CPU'}")
    print(f"  PyTorch: {torch.__version__}")

    # Step 1: Prepare data
    print(f"\n--- Step 1: Preparing Training Data ---")
    texts, labels = prepare_training_data(max_maalej=args.max_maalej)

    if args.auto_label_only:
        print("\n  --auto-label-only flag set. Skipping training.")
        return

    if len(texts) < 100:
        print("\n  ERROR: Not enough training data. Need at least 100 labeled reviews.")
        return

    # Step 2: Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Step 3: Train
    model, tokenizer, trainer = train_model(
        texts=train_texts + val_texts,  # trainer handles split internally
        labels=train_labels + val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_name=args.model,
        output_dir=args.output_dir,
    )

    # Step 4: Evaluate on validation set
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    evaluate_model(model, tokenizer, val_texts, val_labels, device=device)

    # Step 5: Test predictions
    test_predictions(args.output_dir)

    print(f"\n{'=' * 60}")
    print("ALL DONE!")
    print(f"{'=' * 60}")
    print(f"  Model saved to: {args.output_dir}/")
    print(f"  To use in pipeline:")
    print(f"    from src.stage1.classifier import ReviewClassifier")
    print(f"    classifier = ReviewClassifier(model_name_or_path='{args.output_dir}')")
    print(f"    results = classifier.predict(['App crashes on login'])")


if __name__ == "__main__":
    main()
