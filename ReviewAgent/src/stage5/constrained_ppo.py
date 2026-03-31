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

        # ---- Rule-based fallback (continuous scoring per dimension) ----
        import re

        lower = text.lower()

        # Each dimension scored 0.0 (major violation) to 1.0 (fully compliant)
        # with partial scores for soft/hedged violations

        # ---- Dimension 1: Promise Safety (0-1) ----
        promise_score = 1.0

        # Hard violations (definite promises)
        hard_promises = [
            "we guarantee", "guaranteed", "we promise", "100%",
            "definitely will", "for sure will", "we assure you",
        ]
        if any(p in lower for p in hard_promises):
            promise_score = 0.1

        # Medium violations (implicit promises with timelines)
        elif any(p in lower for p in [
            "will be fixed in", "will be resolved in", "fix will be included in",
            "releasing next week", "shipping in the next", "coming in version",
            "will be available in", "expect a fix by",
        ]):
            promise_score = 0.4

        # Soft violations (vague future commitments)
        elif any(p in lower for p in [
            "we will fix", "will be fixed", "will be resolved",
            "we will address", "will be included in the next update",
            "a fix will be", "we are going to fix",
        ]):
            promise_score = 0.6

        # Hedged language (acceptable but slightly risky)
        elif any(p in lower for p in [
            "working on a fix", "investigating", "looking into",
            "we aim to", "we hope to", "we plan to",
            "our team is working", "actively working",
        ]):
            promise_score = 0.85

        # ---- Dimension 2: Information Safety (0-1) ----
        info_score = 1.0

        # Hard violations (specific internal details)
        hard_leaks = [
            "production server", "database schema", "api key",
            "deployment pipeline", "jenkins", "docker", "kubernetes",
            "source code", "codebase", "git repo", "pull request",
            "null pointer", "stack trace", "exception in",
        ]
        if any(p in lower for p in hard_leaks):
            info_score = 0.1

        # Medium violations (team/process details)
        elif any(p in lower for p in [
            "our engineer", "our developer", "our backend team",
            "server-side", "our team member", "sprint",
            "jira", "slack channel", "internal",
        ]):
            info_score = 0.4

        # Soft violations (vague internal references)
        elif any(p in lower for p in [
            "our technical team", "our development team",
            "our team has identified", "our team found",
            "we identified a bug in", "root cause",
        ]):
            info_score = 0.75

        # Common acceptable phrases (not violations)
        # "our team is working on it" — ok
        # "we are aware of the issue" — ok

        # ---- Dimension 3: Tone Compliance (0-1) ----
        tone_score = 1.0

        # Hard tone violations
        hard_tone = [
            "that's your problem", "that is your problem",
            "not our fault", "not our problem",
            "you should have", "your fault",
            "stop complaining", "deal with it", "too bad",
            "read the manual", "figure it out", "not my problem",
        ]
        if any(t in lower for t in hard_tone):
            tone_score = 0.1

        # Medium tone issues (dismissive/curt)
        elif any(t in lower for t in [
            "obviously", "clearly you", "as i said",
            "i already told you", "not sure what you expect",
        ]):
            tone_score = 0.4

        # Slightly off-tone (too casual or too formal)
        elif len(text) < 30 and not any(w in lower for w in ["sorry", "thank", "apologize"]):
            tone_score = 0.7  # Very short response without empathy

        # Bonus: explicit empathy boosts tone score
        empathy_words = ["sorry", "apologize", "understand", "frustrating", "inconvenience"]
        if any(w in lower for w in empathy_words):
            tone_score = min(1.0, tone_score + 0.1)

        # ---- Dimension 4: Legal Safety (0-1) ----
        legal_score = 1.0

        # Hard legal violations
        legal_hard = [
            r"\bwe accept liability\b", r"\bwe are liable\b",
            r"\bwe admit fault\b", r"\bour fault entirely\b",
            r"\bwe take full responsibility for (?:the |any )?damage\b",
            r"\blawsuit\b", r"\bsue us\b",
        ]
        if any(re.search(p, lower) for p in legal_hard):
            legal_score = 0.1

        # Medium legal risks
        elif any(p in lower for p in [
            "compensation", "refund guaranteed", "we owe you",
            "we accept responsibility", "our liability",
            "our fault entirely", "we are at fault",
        ]):
            legal_score = 0.4

        # Soft legal caution
        elif any(p in lower for p in [
            "we take responsibility", "this is on us", "our mistake",
        ]):
            legal_score = 0.7

        # ---- Combination: min-weighted approach ----
        # A serious violation in ANY dimension should drop the score significantly.
        # Use: weighted_avg * min_penalty
        # where min_penalty = min(all_scores) ^ 0.5  (square root softens but still penalizes)
        dim_scores = [promise_score, info_score, tone_score, legal_score]
        weights = [0.35, 0.25, 0.20, 0.20]

        weighted_avg = sum(w * s for w, s in zip(weights, dim_scores))
        min_score = min(dim_scores)

        # If any dimension is severely violated (< 0.5), the min dominates
        # If all dimensions are clean (> 0.8), the weighted avg dominates
        if min_score < 0.5:
            # Severe violation: score drops to at most min_score * 1.2
            final_score = min(weighted_avg, min_score * 1.2)
        elif min_score < 0.8:
            # Moderate violation: blend weighted avg with min
            final_score = 0.5 * weighted_avg + 0.5 * min_score
        else:
            # Clean: use weighted average
            final_score = weighted_avg

        return round(final_score, 4)
