"""Stage 5 Pipeline: RLHF feedback loop with progressive training."""

from __future__ import annotations

from .feedback_collector import DualStreamFeedbackCollector
from .kto_trainer import KTOTrainerWrapper
from .dpo_trainer import DPOTrainerWrapper
from .constrained_ppo import ConstrainedPPOTrainer
from .feedback_propagator import FeedbackPropagator


class Stage5Pipeline:
    """Orchestrates the dual-objective RLHF feedback loop."""

    def __init__(
        self,
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        feedback_collector: DualStreamFeedbackCollector | None = None,
        propagator: FeedbackPropagator | None = None,
    ):
        self.base_model = base_model
        self.collector = feedback_collector or DualStreamFeedbackCollector()
        self.propagator = propagator or FeedbackPropagator()
        self.kto = KTOTrainerWrapper(base_model)
        self.dpo = DPOTrainerWrapper(base_model)
        self.ppo = ConstrainedPPOTrainer(base_model)
        self.iteration = 0

    def select_trainer(self) -> str:
        """Select RLHF method based on available data volume."""
        n_responses = len(self.collector.quality_ratings)
        if n_responses < 500:
            return "kto"
        elif n_responses < 1500:
            return "dpo"
        else:
            return "constrained_ppo"

    def run_iteration(
        self,
        prompts: list[str],
        responses: list[str],
    ) -> dict:
        """Run one RLHF iteration: select method, train, propagate feedback."""
        self.iteration += 1
        method = self.select_trainer()
        metrics = {}

        if method == "kto":
            kto_data = self.collector.export_kto_data_with_text()
            if kto_data:
                metrics = self.kto.train(
                    prompts=[d["prompt"] for d in kto_data],
                    responses=[d["response"] for d in kto_data],
                    labels=[d["label"] for d in kto_data],
                )
            elif prompts and responses:
                # Fallback: use provided prompts/responses with basic labels
                basic_labels = self.collector.export_kto_data()
                labels = [d["label"] for d in basic_labels]
                if labels:
                    metrics = self.kto.train(
                        prompts=prompts[: len(labels)],
                        responses=responses[: len(labels)],
                        labels=labels,
                    )

        elif method == "dpo":
            dpo_data = self.collector.export_dpo_data()
            if dpo_data:
                dpo_prompts = [d["prompt"] for d in dpo_data]
                dpo_chosen = [d["chosen"] for d in dpo_data]
                dpo_rejected = [d["rejected"] for d in dpo_data]
                metrics = self.dpo.train(
                    prompts=dpo_prompts,
                    chosen_responses=dpo_chosen,
                    rejected_responses=dpo_rejected,
                )

        elif method == "constrained_ppo":
            quality_data, compliance_data = self.collector.export_ppo_data()
            if quality_data and compliance_data:
                # Train reward models first
                q_texts = [prompts[0]] * len(quality_data) if prompts else []
                q_labels = [
                    sum(d["scores"].values()) / len(d["scores"])
                    for d in quality_data
                ]
                c_texts = q_texts
                c_labels = [1 if d["compliant"] else 0 for d in compliance_data]

                if q_texts:
                    self.ppo.train_reward_models(q_texts, q_labels, c_texts, c_labels)
                    metrics = self.ppo.train(prompts)

        # Propagate feedback backward
        self._propagate_feedback()

        metrics["iteration"] = self.iteration
        metrics["method"] = method
        return metrics

    def _propagate_feedback(self) -> None:
        """Propagate collected feedback to upstream stages."""
        # Stage 3: rubric scores
        rubric_data = [
            {"spec_id": q["response_id"], "scores": q["scores"]}
            for q in self.collector.quality_ratings
        ]
        if rubric_data:
            self.propagator.propagate_to_stage3(rubric_data)

        # Stage 4b: quality scores for RLHF
        quality_data = [
            {"response_id": q["response_id"], "scores": q["scores"]}
            for q in self.collector.quality_ratings
        ]
        if quality_data:
            self.propagator.propagate_to_stage4b(quality_data)
