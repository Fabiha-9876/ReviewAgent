"""KTO Trainer — Phase 1 RLHF with binary feedback."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import KTOConfig, KTOTrainer


class KTOTrainerWrapper:
    """Phase 1 RLHF: KTO with binary good/bad signals (small data)."""

    def __init__(
        self,
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        lora_r: int = 16,
        lora_alpha: int = 32,
        output_dir: str = "models/stage5_kto",
    ):
        self.base_model_name = base_model
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

    def _load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
            self.model = get_peft_model(self.model, self.lora_config)

    def prepare_dataset(
        self, prompts: list[str], responses: list[str], labels: list[bool]
    ) -> Dataset:
        """Convert to KTO format: prompt, completion, label (desirable/undesirable)."""
        return Dataset.from_dict({
            "prompt": prompts,
            "completion": responses,
            "label": labels,
        })

    def train(
        self,
        prompts: list[str],
        responses: list[str],
        labels: list[bool],
        epochs: int = 3,
        learning_rate: float = 5e-6,
        batch_size: int = 4,
    ) -> dict:
        """Run KTO training."""
        self._load_model()
        dataset = self.prepare_dataset(prompts, responses, labels)

        config = KTOConfig(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            logging_steps=10,
            save_strategy="epoch",
        )

        trainer = KTOTrainer(
            model=self.model,
            args=config,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        result = trainer.train()
        trainer.save_model(self.output_dir)
        return result.metrics

    def save_model(self, path: str | None = None) -> None:
        if self.model:
            self.model.save_pretrained(path or self.output_dir)

    def load_model(self, path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path)
