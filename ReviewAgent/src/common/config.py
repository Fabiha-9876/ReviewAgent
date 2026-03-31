"""Configuration loader using OmegaConf."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(config_name: str = "base", config_dir: str | None = None) -> DictConfig:
    """Load a YAML config file by name from the configs/ directory."""
    if config_dir is None:
        config_dir = Path(__file__).parent.parent.parent / "configs"
    else:
        config_dir = Path(config_dir)

    config_path = config_dir / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    base_cfg = OmegaConf.load(config_dir / "base.yaml") if (config_dir / "base.yaml").exists() else OmegaConf.create()
    stage_cfg = OmegaConf.load(config_path) if config_name != "base" else OmegaConf.create()

    return OmegaConf.merge(base_cfg, stage_cfg)
