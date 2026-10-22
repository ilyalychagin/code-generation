from dataclasses import dataclass
from pathlib import Path

from configs.baseline_1000k import Config as BaselineConfig
from configs.config import PROJECT_ROOT


@dataclass(frozen=True)
class Config(BaselineConfig):
    output_dir: Path = PROJECT_ROOT / "runs/lower_lr1e4_1000k"
    learning_rate: float = 1e-4
    comet_experiment_name: str = "lower_lr1e4_1000k"
