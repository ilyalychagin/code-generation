from dataclasses import dataclass
from pathlib import Path

from configs.baseline_1000k import Config as BaseConfig
from configs.config import PROJECT_ROOT


@dataclass(frozen=True)
class Config(BaseConfig):
    output_dir: Path = PROJECT_ROOT / "runs/baseline_1000k_bs8"
    batch_size: int = 8
    learning_rate: float = 7e-5
    val_every_n_steps: int = 12500
    log_code_every_n_steps: int | None = 5000
    comet_experiment_name: str = "baseline_1000k_bs8"
