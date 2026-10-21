from pathlib import Path
import random

import numpy as np
import torch


def save_checkpoint(
    path, model, optimizer, scheduler, epoch, global_step,
    best_val_loss, train_losses, val_losses,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy_state = np.random.get_state()
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "train_losses": list(train_losses),
        "val_losses": list(val_losses),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": (
            numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]
        ),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(path, model, optimizer, scheduler):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    random.setstate(checkpoint["python_rng_state"])
    numpy_state = checkpoint["numpy_rng_state"]
    np.random.set_state((
        numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
        *numpy_state[2:],
    ))
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint["cuda_rng_states"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])

    return checkpoint
