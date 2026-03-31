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

        all_rewards = []
        all_quality = []
        all_compliance = []

        for epoch in range(epochs):
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i : i + batch_size]
                query_tensors = [
                    self.tokenizer.encode(p, return_tensors="pt").squeeze()
                    for p in batch_prompts
                ]

                # Generate responses
                response_tensors = ppo_trainer.generate(query_tensors, max_new_tokens=256)

                # Compute rewards using trained models or heuristic fallback
                rewards = []
                for rt in response_tensors:
                    text = self.tokenizer.decode(rt, skip_special_tokens=True)
                    q_score = self._score_quality(text)
                    c_score = self._score_compliance(text)
                    reward = self.compute_constrained_reward(q_score, c_score)
                    rewards.append(torch.tensor(reward))
                    all_quality.append(q_score)
                    all_compliance.append(c_score)
                    all_rewards.append(reward)

                # PPO step
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        ppo_trainer.save_pretrained(self.output_dir)

        metrics = {
            "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
            "mean_quality": sum(all_quality) / len(all_quality) if all_quality else 0.0,
            "mean_compliance": sum(all_compliance) / len(all_compliance) if all_compliance else 0.0,
            "compliance_violation_rate": sum(1 for c in all_compliance if c < self.compliance_threshold) / len(all_compliance) if all_compliance else 0.0,
            "n_responses_scored": len(all_rewards),
            "reward_model_used": self.quality_reward_model is not None,
        }
        return metrics

    def _score_quality(self, text: str) -> float:
        """Score response quality (0-1).

        Uses the trained reward model if available. Falls back to a
        multi-signal heuristic that evaluates length, specificity,
        empathy, and actionability markers.
        """
        # ---- Trained reward model inference ----
        if self.quality_reward_model is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                inputs = tokenizer(
                    text, padding="max_length", truncation=True,
                    max_length=256, return_tensors="pt",
                )
                device = next(self.quality_reward_model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = self.quality_reward_model(**inputs).logits
                # Regression model outputs single value; normalize to 0-1
                score = torch.sigmoid(logits.squeeze()).item()
                return score
            except Exception:
                pass  # Fall through to heuristic

        # ---- Heuristic fallback (when no reward model is trained) ----
        lower = text.lower()
        score = 0.0
        max_score = 0.0

        # Signal 1: Length (longer responses tend to be more helpful, up to a point)
        # 50-300 chars is the sweet spot
        length = len(text)
        max_score += 1.0
        if length < 20:
            score += 0.1
        elif length < 50:
            score += 0.3
        elif length <= 300:
            score += 1.0
        elif length <= 500:
            score += 0.8
        else:
            score += 0.6

        # Signal 2: Specificity — does it mention concrete details?
        specificity_markers = [
            "version", "v3.", "v2.", "update", "android", "ios", "device",
            "samsung", "pixel", "iphone", "crash", "login", "battery",
            "fix", "resolved", "identified", "issue", "bug",
        ]
        specific_count = sum(1 for m in specificity_markers if m in lower)
        max_score += 1.0
        score += min(1.0, specific_count * 0.2)

        # Signal 3: Empathy — does it acknowledge the user's frustration?
        empathy_markers = [
            "sorry", "apologize", "understand", "frustrating", "inconvenience",
            "appreciate", "thank you", "thank", "feedback", "patience",
        ]
        empathy_count = sum(1 for m in empathy_markers if m in lower)
        max_score += 1.0
        score += min(1.0, empathy_count * 0.3)

        # Signal 4: Actionability — does it suggest next steps?
        action_markers = [
            "please try", "you can", "we recommend", "update the app",
            "clear cache", "reinstall", "contact support", "settings",
            "next update", "working on", "will be", "check",
        ]
        action_count = sum(1 for m in action_markers if m in lower)
        max_score += 1.0
        score += min(1.0, action_count * 0.3)

        # Signal 5: Not a generic template
        generic_markers = [
            "thank you for your feedback",
            "we appreciate your review",
            "please contact us",
        ]
        max_score += 1.0
        is_generic = any(g in lower for g in generic_markers) and length < 80
        score += 0.2 if is_generic else 1.0

        return round(score / max_score, 4) if max_score > 0 else 0.5

    def _score_compliance(self, text: str) -> float:
        """Score response compliance (0-1).

        Uses the trained compliance model if available. Falls back to
        rule-based checking for 4 compliance dimensions: false promises,
        information leaks, tone violations, and legal risks.
        """
        # ---- Trained compliance model inference ----
        if self.compliance_reward_model is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                inputs = tokenizer(
                    text, padding="max_length", truncation=True,
                    max_length=256, return_tensors="pt",
                )
                device = next(self.compliance_reward_model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = self.compliance_reward_model(**inputs).logits
                # Binary classifier: class 1 = compliant
                probs = torch.softmax(logits, dim=-1)
                compliance_prob = probs[0][1].item()
                return compliance_prob
            except Exception:
                pass  # Fall through to heuristic

        # ---- Rule-based fallback ----
        import re

        lower = text.lower()
        violations = 0
        total_checks = 4

        # Check 1: No false promises
        promise_phrases = [
            "we will fix", "will be fixed", "guaranteed", "we promise",
            "we guarantee", "definitely will", "for sure",
            "will be resolved in the next", "we assure you",
        ]
        if any(p in lower for p in promise_phrases):
            violations += 1

        # Check 2: No information leaks
        leak_phrases = [
            "our engineer", "our developer", "internal system",
            "codebase", "source code", "backend server", "server-side",
            "our team member", "sprint", "jira", "slack channel",
            "production server", "database schema", "api key", "deployment pipeline",
            "jenkins", "docker", "kubernetes",
        ]
        if any(p in lower for p in leak_phrases):
            violations += 1

        # Check 3: Tone compliance
        tone_violations = [
            "that's your problem", "not our fault", "you should have",
            "obviously", "clearly you", "read the manual",
            "stop complaining", "deal with it", "too bad",
        ]
        if any(t in lower for t in tone_violations):
            violations += 1

        # Check 4: Legal safety (use word boundaries to avoid "sue" matching "issue")
        legal_patterns = [
            r"\bwe accept liability\b", r"\bwe are liable\b",
            r"\bcompensation\b", r"\bwe admit\b",
            r"\bour fault entirely\b", r"\bsue\b", r"\blawsuit\b",
            r"\bwe take full responsibility for the damage\b",
        ]
        if any(re.search(p, lower) for p in legal_patterns):
            violations += 1

        return round(1.0 - (violations / total_checks), 4)
