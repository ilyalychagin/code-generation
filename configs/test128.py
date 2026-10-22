from dataclasses import dataclass
from pathlib import Path

from configs.config import Config as BaseConfig, PROJECT_ROOT


@dataclass(frozen=True)
class Config(BaseConfig):
    output_dir: Path = PROJECT_ROOT / "runs/test128"
    max_train_samples: int | None = 128
    max_val_samples: int | None = 1
    epochs: int = 1
    learning_rate: float = 1e-3
    comet_experiment_name: str = "test128"
