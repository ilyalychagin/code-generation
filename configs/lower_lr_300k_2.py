from dataclasses import dataclass
from pathlib import Path

from configs.baseline_300k import Config as BaselineConfig
from configs.config import PROJECT_ROOT


@dataclass(frozen=True)
class Config(BaselineConfig):
    output_dir: Path = PROJECT_ROOT / "runs/lower_lr7e5_300k"
    learning_rate: float = 7e-5
    comet_experiment_name: str = "lower_lr7e5_300k"
