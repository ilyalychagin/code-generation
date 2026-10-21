from src.dataset.collate import collate
from src.models.model import TestModel
from torch.utils.data import DataLoader

from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim

from src.scripts.train import train
from src.scripts.inference import write_answer
from src.utils.utils import set_seed
from src.utils.logging import create_experiment, validate_logging_config
from configs.config import Config
from src.dataset.dataset import get_tokenized_dataset, get_raw_dataset
from src.models.tokenizer import get_tokenizer


def limit_dataset(dataset, max_samples, seed):
    if max_samples is None:
        return dataset
    if max_samples < 1:
        raise ValueError("Sample limits must be positive or None")
    if max_samples >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def main():
    config = Config()
    validate_logging_config(config)
    for limit in (config.max_train_samples, config.max_val_samples):
        if limit is not None and limit < 1:
            raise ValueError("Sample limits must be positive or None")
    if config.resume_from_checkpoint is not None and not config.resume_from_checkpoint.is_file():
        raise FileNotFoundError(config.resume_from_checkpoint)
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    raw_data_dir = config.raw_data_dir
    raw_dataset = get_raw_dataset(raw_data_dir)

    tokenizer = get_tokenizer(
        tokenizer_dir=config.tokenizer_dir,
        raw_dataset=raw_dataset,
        vocab_size=config.vocab_size,
    )

    dataset = get_tokenized_dataset(
        tokenized_data_dir=config.tokenized_data_dir,
        tokenizer=tokenizer,
        raw_data_path=config.raw_data_dir,
    )

    split = dataset["train"].train_test_split(
        test_size=config.validation_size,
        seed=config.seed,
    )

    train_dataset = limit_dataset(split["train"], config.max_train_samples, config.seed)
    val_dataset = limit_dataset(split["test"], config.max_val_samples, config.seed)
    print(
        f"Dataset sizes: train={len(train_dataset)}, validation={len(val_dataset)}, "
        f"test={len(dataset['test'])}",
        flush=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=partial(
            collate, 
            trunc_len=config.trunc_len, 
            padding_value=tokenizer.pad_token_id,
            end_token=tokenizer.eos_token_id,
            train=True
        ) 
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=partial(
            collate,
            trunc_len=config.trunc_len,
            padding_value=tokenizer.pad_token_id,
            end_token=tokenizer.eos_token_id,
            train=True,
        ),
    )

    test_loader = DataLoader(
        dataset['test'],
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=partial(
            collate, 
            trunc_len=config.trunc_len, 
            padding_value=tokenizer.pad_token_id,
            end_token=tokenizer.eos_token_id,
            train=False
        ) 
    )
    
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    model = TestModel(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
    ).to(device)

    optimizer = optim.Adam(model.model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    experiment = create_experiment(config, model, optimizer, scheduler, device)
    try:
        train(
            model, train_loader, val_loader, nn.CrossEntropyLoss(), optimizer,
            scheduler, config.epochs, device,
            experiment=experiment,
            log_every_n_steps=config.log_every_n_steps,
            checkpoint_dir=config.output_dir,
            resume_from_checkpoint=config.resume_from_checkpoint,
        )

        torch.save(model.state_dict(), config.output_dir / "checkpoint.pt")

        write_answer(
            model, test_loader, device, tokenizer,
            output_path=config.output_dir / "pass1_questions.jsonl",
            max_new_tokens=config.max_new_tokens,
        )
    finally:
        if experiment is not None:
            experiment.end()


if __name__ == "__main__":
    main()


