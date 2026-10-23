from dataclasses import dataclass
from pathlib import Path

from configs.baseline_1000k_bs8 import Config as BaseConfig
from configs.config import PROJECT_ROOT


@dataclass(frozen=True)
class Config(BaseConfig):
    output_dir: Path = PROJECT_ROOT / "runs/lower_lr5e5_1000k_bs8"
    learning_rate: float = 5e-5
    comet_experiment_name: str = "lower_lr5e5_1000k_bs8"
