from .feedback_collector import DualStreamFeedbackCollector
from .feedback_propagator import FeedbackPropagator

try:
    from .kto_trainer import KTOTrainerWrapper
    from .dpo_trainer import DPOTrainerWrapper
    from .constrained_ppo import ConstrainedPPOTrainer
    from .pipeline import Stage5Pipeline
except ImportError as _exc:
    # trl/peft may be absent, or present at an incompatible version (TRL 1.0 removed the
    # PPO API that constrained_ppo.py targets). Warn rather than degrade silently, so a
    # missing Stage5Pipeline is visible instead of surfacing later as a NameError.
    import warnings

    warnings.warn(
        f"src.stage5: trainer classes unavailable ({_exc}). "
        "KTOTrainerWrapper, DPOTrainerWrapper, ConstrainedPPOTrainer and Stage5Pipeline "
        "are not importable in this environment; install trl<1.0 and peft to enable them.",
        ImportWarning,
        stacklevel=2,
    )
