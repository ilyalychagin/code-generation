from dataclasses import dataclass
from pathlib import Path

from configs.baseline_9000k_bs8 import Config as BaseConfig
from configs.config import PROJECT_ROOT


@dataclass(frozen=True)
class Config(BaseConfig):
    output_dir: Path = PROJECT_ROOT / "runs/baseline_9000k_e6_d8_bs8"
    num_encoder_layers: int = 6
    num_decoder_layers: int = 8
    comet_experiment_name: str = "baseline_9000k_e6_d8_bs8"
