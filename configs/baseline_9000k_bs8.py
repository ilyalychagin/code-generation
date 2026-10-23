from dataclasses import dataclass
from pathlib import Path

from configs.baseline_1000k_bs8 import Config as BaseConfig
from configs.config import PROJECT_ROOT


@dataclass(frozen=True)
class Config(BaseConfig):
    output_dir: Path = PROJECT_ROOT / "runs/baseline_9000k_bs8"
    max_train_samples: int | None = 9000000
    val_every_n_steps: int = 25000
    log_every_n_steps: int = 500
    log_code_every_n_steps: int | None = 25000
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    comet_experiment_name: str = "baseline_9000k_bs8"
