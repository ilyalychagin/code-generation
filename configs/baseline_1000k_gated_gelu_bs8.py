from dataclasses import dataclass
from pathlib import Path

from configs.baseline_1000k_bs8 import Config as BaseConfig
from configs.config import PROJECT_ROOT


@dataclass(frozen=True)
class Config(BaseConfig):
    output_dir: Path = PROJECT_ROOT / "runs/baseline_1000k_gated_gelu_bs8"
    d_ff: int = 1408
    feed_forward_proj: str = "gated-gelu"
    comet_experiment_name: str = "baseline_1000k_gated_gelu_bs8"
