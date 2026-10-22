from dataclasses import dataclass
from pathlib import Path

from configs.config import Config as BaseConfig, PROJECT_ROOT


@dataclass(frozen=True)
class Config(BaseConfig):
    output_dir: Path = PROJECT_ROOT / "runs/baseline_1000k"
    max_train_samples: int | None = 1000000
    max_val_samples: int | None = 5000
    epochs: int = 1
    learning_rate: float = 1e-4
    comet_experiment_name: str = "baseline_1000k"
    log_every_n_steps: int = 100
    log_code_every_n_steps: int | None = 10000
    log_code_num_samples: int = 8
