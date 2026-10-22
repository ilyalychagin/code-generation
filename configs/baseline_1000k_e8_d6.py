from dataclasses import dataclass
from pathlib import Path

from configs.baseline_1000k import Config as BaseConfig
from configs.config import PROJECT_ROOT


@dataclass(frozen=True)
class Config(BaseConfig):
    output_dir: Path = PROJECT_ROOT / "runs/baseline_1000k_e8_d6"
    num_encoder_layers: int = 8
    num_decoder_layers: int = 6
    batch_size: int = 4
    learning_rate: float = 7e-5
    comet_experiment_name: str = "baseline_1000k_e8_d6"
