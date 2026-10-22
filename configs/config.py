from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    seed: int = 42

    # Path
    raw_data_dir: Path = PROJECT_ROOT / "data/tinysms_qa_12m"
    tokenized_data_dir: Path = PROJECT_ROOT / "data/tinysms_bpe"
    tokenizer_dir: Path = PROJECT_ROOT / "artifacts/tokenizer"
    output_dir: Path = PROJECT_ROOT / "runs/baseline"

    #Tokenizer options
    vocab_size: int = 16000
    validation_size: float = 0.1
    batch_size: int = 4
    trunc_len: int | None = 512

    # Training
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    resume_from_checkpoint: Path | None = None
    epochs: int = 10
    learning_rate: float = 1e-3
    max_grad_norm: float | None = None
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2

    # Generation
    test_batch_size: int = 64
    max_new_tokens: int = 512

    # Comet logs locally; COMET_API_KEY is used only for uploading afterwards.
    comet_enabled: bool = True
    comet_upload_on_end: bool = True
    comet_upload_timeout: int = 60
    comet_project: str = "hw02-code-generation"
    comet_workspace: str | None = ""
    comet_experiment_name: str = "baseline"
    log_every_n_steps: int = 50
    log_code_every_n_steps: int | None = 1000
    log_code_num_samples: int = 8
