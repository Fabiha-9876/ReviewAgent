"""Tests for KTO and DPO trainer wrappers — dataset preparation (no model loading)."""

import pytest
from datasets import Dataset

try:
    from src.stage5.kto_trainer import KTOTrainerWrapper
    from src.stage5.dpo_trainer import DPOTrainerWrapper
except ImportError:
    pytestmark = pytest.mark.skip(reason="trl not available")
    KTOTrainerWrapper = None
    DPOTrainerWrapper = None


# ============================================================
# KTO Trainer Tests
# ============================================================

class TestKTOTrainer:

    def test_default_params(self):
        trainer = KTOTrainerWrapper.__new__(KTOTrainerWrapper)
        trainer.__init__()
        assert trainer.base_model_name == "meta-llama/Llama-3.1-8B-Instruct"
        assert trainer.output_dir == "models/stage5_kto"
        assert trainer.model is None
        assert trainer.tokenizer is None

    def test_custom_params(self):
        trainer = KTOTrainerWrapper.__new__(KTOTrainerWrapper)
        trainer.__init__(base_model="test-model", lora_r=8, lora_alpha=16, output_dir="/tmp/kto")
        assert trainer.base_model_name == "test-model"
        assert trainer.output_dir == "/tmp/kto"
        assert trainer.lora_config.r == 8
        assert trainer.lora_config.lora_alpha == 16

    def test_prepare_dataset(self):
        trainer = KTOTrainerWrapper.__new__(KTOTrainerWrapper)
        trainer.__init__()
        ds = trainer.prepare_dataset(
            prompts=["What is this?", "Fix my app"],
            responses=["It's a bug", "Try updating"],
            labels=[True, False],
        )
        assert isinstance(ds, Dataset)
        assert len(ds) == 2
        assert ds[0]["prompt"] == "What is this?"
        assert ds[0]["completion"] == "It's a bug"
        assert ds[0]["label"] is True
        assert ds[1]["label"] is False

    def test_prepare_dataset_empty(self):
        trainer = KTOTrainerWrapper.__new__(KTOTrainerWrapper)
        trainer.__init__()
        ds = trainer.prepare_dataset([], [], [])
        assert len(ds) == 0

    def test_lora_config_target_modules(self):
        trainer = KTOTrainerWrapper.__new__(KTOTrainerWrapper)
        trainer.__init__()
        assert "q_proj" in trainer.lora_config.target_modules
        assert "v_proj" in trainer.lora_config.target_modules
        assert trainer.lora_config.task_type == "CAUSAL_LM"


# ============================================================
# DPO Trainer Tests
# ============================================================

class TestDPOTrainer:

    def test_default_params(self):
        trainer = DPOTrainerWrapper.__new__(DPOTrainerWrapper)
        trainer.__init__()
        assert trainer.base_model_name == "meta-llama/Llama-3.1-8B-Instruct"
        assert trainer.output_dir == "models/stage5_dpo"
        assert trainer.model is None

    def test_custom_params(self):
        trainer = DPOTrainerWrapper.__new__(DPOTrainerWrapper)
        trainer.__init__(base_model="test-model", output_dir="/tmp/dpo")
        assert trainer.base_model_name == "test-model"

    def test_prepare_dataset(self):
        trainer = DPOTrainerWrapper.__new__(DPOTrainerWrapper)
        trainer.__init__()
        ds = trainer.prepare_dataset(
            prompts=["Fix my login", "Add dark mode"],
            chosen_responses=["We found the bug", "Great idea, noted"],
            rejected_responses=["Try again", "No"],
        )
        assert isinstance(ds, Dataset)
        assert len(ds) == 2
        assert ds[0]["prompt"] == "Fix my login"
        assert ds[0]["chosen"] == "We found the bug"
        assert ds[0]["rejected"] == "Try again"

    def test_prepare_dataset_empty(self):
        trainer = DPOTrainerWrapper.__new__(DPOTrainerWrapper)
        trainer.__init__()
        ds = trainer.prepare_dataset([], [], [])
        assert len(ds) == 0

    def test_lora_config(self):
        trainer = DPOTrainerWrapper.__new__(DPOTrainerWrapper)
        trainer.__init__()
        assert trainer.lora_config.r == 16
        assert trainer.lora_config.lora_alpha == 32
        assert trainer.lora_config.lora_dropout == 0.05
