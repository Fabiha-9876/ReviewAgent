"""Multi-label RoBERTa classifier for app review classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset


LABELS = [
    "bug_report",
    "feature_request",
    "performance",
    "usability",
    "compatibility",
    "praise",
    "other",
]


class ReviewClassifier:
    """Multi-label RoBERTa classifier for app reviews."""

    def __init__(
        self,
        model_name_or_path: str = "roberta-base",
        num_labels: int = 7,
        confidence_threshold: float = 0.7,
        conflict_margin: float = 0.15,
        device: str | None = None,
    ):
        self.labels = LABELS
        self.num_labels = num_labels
        self.confidence_threshold = confidence_threshold
        self.conflict_margin = conflict_margin
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            problem_type="multi_label_classification",
        ).to(self.device)

    def train(
        self,
        train_texts: list[str],
        train_labels: list[list[int]],
        val_texts: list[str],
        val_labels: list[list[int]],
        output_dir: str = "models/stage1_classifier",
        epochs: int = 5,
        lr: float = 2e-5,
        batch_size: int = 16,
    ) -> dict:
        """Fine-tune the classifier on labeled review data."""

        def tokenize(examples):
            tokens = self.tokenizer(
                examples["text"], padding="max_length", truncation=True, max_length=256
            )
            tokens["labels"] = [
                [float(x) for x in label] for label in examples["labels"]
            ]
            return tokens

        train_ds = Dataset.from_dict({"text": train_texts, "labels": train_labels})
        val_ds = Dataset.from_dict({"text": val_texts, "labels": val_labels})
        train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
        val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])
        train_ds.set_format("torch")
        val_ds.set_format("torch")

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            warmup_ratio=0.1,
            logging_steps=50,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
        )
        result = trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        return result.metrics

    def predict(self, texts: list[str]) -> list[tuple[list[str], dict[str, float]]]:
        """Predict labels and confidence scores for a batch of texts."""
        self.model.eval()
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**tokens).logits
            probs = torch.sigmoid(logits).cpu().numpy()

        results = []
        for prob_row in probs:
            confidences = {label: float(prob_row[i]) for i, label in enumerate(self.labels)}
            predicted_labels = [
                label for label, conf in confidences.items() if conf >= 0.5
            ]
            if not predicted_labels:
                best_idx = int(np.argmax(prob_row))
                predicted_labels = [self.labels[best_idx]]
            results.append((predicted_labels, confidences))
        return results

    def needs_hitl(self, confidences: dict[str, float]) -> bool:
        """Check if this prediction needs human review."""
        sorted_confs = sorted(confidences.values(), reverse=True)
        if sorted_confs[0] < self.confidence_threshold:
            return True
        if len(sorted_confs) > 1 and (sorted_confs[0] - sorted_confs[1]) < self.conflict_margin:
            return True
        return False

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, **kwargs) -> ReviewClassifier:
        instance = cls(model_name_or_path=path, **kwargs)
        return instance
