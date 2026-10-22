from dataclasses import dataclass
from pathlib import Path

from configs.baseline_100k import Config as BaselineConfig
from configs.config import PROJECT_ROOT


@dataclass(frozen=True)
class Config(BaselineConfig):
    output_dir: Path = PROJECT_ROOT / "runs/lower_lr_100k"
    learning_rate: float = 3e-4
    comet_experiment_name: str = "lower_lr_100k"
