from dataclasses import dataclass
from pathlib import Path

from configs.config import Config as BaseConfig, PROJECT_ROOT


@dataclass(frozen=True)
class Config(BaseConfig):
    max_train_samples: int | None = 128
    max_val_samples: int | None = 128
    epochs: int = 100
    learning_rate: float = 1e-3
    scheduler_patience: int = 100
    output_dir: Path = PROJECT_ROOT / "runs/test128_clip"
    max_grad_norm: float | None = 1.0

    comet_enabled: bool = True
    log_every_n_steps: int = 10
    log_code_every_n_steps: int | None = 100
    comet_experiment_name: str = "test128_clip"

