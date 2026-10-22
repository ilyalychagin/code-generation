from dataclasses import dataclass
from pathlib import Path

from configs.config import Config as BaseConfig, PROJECT_ROOT


@dataclass(frozen=True)
class Config(BaseConfig):
    output_dir: Path = PROJECT_ROOT / "runs/baseline_100k"
    max_train_samples: int | None = 100_000
    max_val_samples: int | None = 5_000
    epochs: int = 1
    learning_rate: float = 1e-3
    comet_experiment_name: str = "baseline_100k"
