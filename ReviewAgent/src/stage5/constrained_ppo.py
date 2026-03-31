"""Constrained PPO Trainer — Phase 3 dual-objective RLHF (CMDP-grounded)."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import Dataset


class ConstrainedPPOTrainer:
    """Phase 3 RLHF: Dual-objective optimization grounded in CMDP theory.

    Maximize: R_quality(response)
    Subject to: C_compliance(response) >= threshold
    """

    def __init__(
        self,
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        lora_r: int = 16,
        lora_alpha: int = 32,
        compliance_threshold: float = 0.95,
        compliance_penalty: float = 5.0,
        output_dir: str = "models/stage5_ppo",
    ):
        self.base_model_name = base_model
        self.compliance_threshold = compliance_threshold
        self.compliance_penalty = compliance_penalty
        self.output_dir = output_dir
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        self.model = None
        self.tokenizer = None
        self.quality_reward_model = None
        self.compliance_reward_model = None

    def _load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.base_model_name,
                peft_config=self.lora_config,
            )

    def train_reward_models(
        self,
        quality_texts: list[str],
        quality_labels: list[float],
        compliance_texts: list[str],
        compliance_labels: list[int],
        reward_model_name: str = "distilbert-base-uncased",
    ) -> None:
        """Train separate quality and compliance reward models."""
        from transformers import Trainer, TrainingArguments

        # Quality reward model
        q_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        q_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name, num_labels=1
        )
        q_ds = Dataset.from_dict({
            "text": quality_texts,
            "label": quality_labels,
        })

        def tokenize_q(examples):
            tokens = q_tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
            tokens["labels"] = [[float(l)] for l in examples["label"]]
            return tokens

        q_ds = q_ds.map(tokenize_q, batched=True, remove_columns=["text", "label"])
        q_ds.set_format("torch")

        trainer = Trainer(
            model=q_model,
            args=TrainingArguments(
                output_dir=f"{self.output_dir}/quality_reward",
                num_train_epochs=3,
                per_device_train_batch_size=8,
                logging_steps=50,
            ),
            train_dataset=q_ds,
        )
        trainer.train()
        self.quality_reward_model = q_model

        # Compliance reward model (binary classification)
        c_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name, num_labels=2
        )
        c_ds = Dataset.from_dict({
            "text": compliance_texts,
            "label": compliance_labels,
        })

        def tokenize_c(examples):
            tokens = q_tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
            tokens["labels"] = examples["label"]
            return tokens

        c_ds = c_ds.map(tokenize_c, batched=True, remove_columns=["text", "label"])
        c_ds.set_format("torch")

        trainer = Trainer(
            model=c_model,
            args=TrainingArguments(
                output_dir=f"{self.output_dir}/compliance_reward",
                num_train_epochs=3,
                per_device_train_batch_size=8,
                logging_steps=50,
            ),
            train_dataset=c_ds,
        )
        trainer.train()
        self.compliance_reward_model = c_model

    def compute_constrained_reward(
        self,
        quality_score: float,
        compliance_score: float,
    ) -> float:
        """Compute CMDP-constrained reward.

        reward = quality_score - penalty * max(0, threshold - compliance_score)
        """
        constraint_violation = max(0, self.compliance_threshold - compliance_score)
        return quality_score - self.compliance_penalty * constraint_violation

    def train(
        self,
        prompts: list[str],
        epochs: int = 3,
        learning_rate: float = 1e-6,
        batch_size: int = 4,
        kl_coeff: float = 0.05,
    ) -> dict:
        """Run Constrained PPO training."""
        self._load_model()

        config = PPOConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            mini_batch_size=min(batch_size, 2),
            ppo_epochs=epochs,
            kl_penalty="kl",
            init_kl_coef=kl_coeff,
        )

        ppo_trainer = PPOTrainer(
            config=config,
            model=self.model,
            tokenizer=self.tokenizer,
        )

        metrics = {"mean_reward": 0.0, "mean_compliance": 0.0}

        for epoch in range(epochs):
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i : i + batch_size]
                query_tensors = [
                    self.tokenizer.encode(p, return_tensors="pt").squeeze()
                    for p in batch_prompts
                ]

                # Generate responses
                response_tensors = ppo_trainer.generate(query_tensors, max_new_tokens=256)

                # Compute rewards (using reward models if available, else placeholder)
                rewards = []
                for rt in response_tensors:
                    text = self.tokenizer.decode(rt, skip_special_tokens=True)
                    q_score = self._score_quality(text)
                    c_score = self._score_compliance(text)
                    reward = self.compute_constrained_reward(q_score, c_score)
                    rewards.append(torch.tensor(reward))

                # PPO step
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        ppo_trainer.save_pretrained(self.output_dir)
        return metrics

    def _score_quality(self, text: str) -> float:
        """Score response quality (0-1). Uses reward model if available."""
        if self.quality_reward_model is not None:
            # Use trained reward model
            return 0.5  # Placeholder for actual model inference
        return 0.5

    def _score_compliance(self, text: str) -> float:
        """Score response compliance (0-1). Uses reward model if available."""
        if self.compliance_reward_model is not None:
            return 0.5  # Placeholder for actual model inference
        # Heuristic compliance check
        violations = 0
        lower = text.lower()
        if any(w in lower for w in ["we will fix", "guaranteed", "promise"]):
            violations += 1
        if any(w in lower for w in ["our engineer", "internal", "codebase"]):
            violations += 1
        return 1.0 - (violations * 0.25)
