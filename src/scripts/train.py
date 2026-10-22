import torch
from pathlib import Path
from tqdm.auto import tqdm
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.logging import gradient_norm

def train_one_epoch(
    model, train_loader, criterion, optimizer, device,
    experiment=None, epoch=0, start_step=0, log_every_n_steps=50,
    code_logger=None, log_code_every_n_steps=1_000,
):
    model.train()

    total_loss = 0.0
    total_tokens = 0

    batch_losses = []
    
    progress = tqdm(
        train_loader,
        desc=f"Train epoch {epoch}" if epoch else "Train",
        unit="batch",
        dynamic_ncols=True,
        mininterval=2.0,
    )
    for batch_index, batch in enumerate(progress, start=1):
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }
        output = model(**batch) # [B, L_output, vocab_size]

        loss = criterion(
            output.logits.transpose(1, 2), 
            batch['labels']
        )

        loss.backward()

        loss_value = loss.item()
        num_tokens = (batch["labels"] != -100).sum().item()
        total_loss += loss_value * num_tokens
        total_tokens += num_tokens

        batch_losses.append(loss_value)

        step = start_step + batch_index
        if experiment is not None and (step == 1 or step % log_every_n_steps == 0):
            experiment.log_metrics(
                {
                    "train/batch_loss": loss_value,
                    "train/grad_norm": gradient_norm(model),
                    **{
                        f"train/lr/group_{index}": group["lr"]
                        for index, group in enumerate(optimizer.param_groups)
                    },
                },
                step=step,
                epoch=epoch,
            )

        optimizer.step()
        optimizer.zero_grad()

        if code_logger is not None and (step == 1 or step % log_code_every_n_steps == 0):
            code_logger(model, step=step, epoch=epoch)

        progress.set_postfix(
            loss=f"{loss_value:.4f}",
            avg_loss=f"{total_loss / total_tokens:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            refresh=False,
        )
    
    return total_loss / total_tokens, batch_losses

@torch.no_grad()
def validation(model, val_loader, criterion, device, epoch=None):
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    
    progress = tqdm(
        val_loader,
        desc=f"Validation epoch {epoch}" if epoch is not None else "Validation",
        unit="batch",
        dynamic_ncols=True,
        mininterval=2.0,
    )
    for batch in progress:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }
        output = model(**batch)

        loss = criterion(
            output.logits.transpose(1, 2), 
            batch['labels']
        )

        num_tokens = (batch["labels"] != -100).sum().item()
        loss_value = loss.item()
        total_loss += loss_value * num_tokens
        total_tokens += num_tokens

        progress.set_postfix(
            loss=f"{loss_value:.4f}",
            avg_loss=f"{total_loss / total_tokens:.4f}",
            refresh=False,
        )

    
    return total_loss / total_tokens

def train(
    model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device,
    experiment=None, log_every_n_steps=50,
    checkpoint_dir=None, resume_from_checkpoint=None,
    code_logger=None, log_code_every_n_steps=1_000,
):

    if log_every_n_steps < 1:
        raise ValueError("log_every_n_steps must be positive")

    train_losses = []
    val_losses = []
    global_step = 0
    start_epoch = 0
    best_val_loss = float("inf")

    if resume_from_checkpoint is not None:
        checkpoint = load_checkpoint(resume_from_checkpoint, model, optimizer, scheduler)
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        best_val_loss = checkpoint["best_val_loss"]
        train_losses = list(checkpoint["train_losses"])
        val_losses = list(checkpoint["val_losses"])
        del checkpoint
        tqdm.write(f"Resumed after epoch {start_epoch}, step {global_step}")

    for epoch in range(start_epoch, epochs):
        
        epoch_loss, batch_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            experiment=experiment,
            epoch=epoch + 1,
            start_step=global_step,
            log_every_n_steps=log_every_n_steps,
            code_logger=code_logger,
            log_code_every_n_steps=log_code_every_n_steps,
        )
        global_step += len(batch_losses)
        train_losses.append(epoch_loss)

        val_loss = validation(model, val_loader, criterion, device, epoch=epoch + 1)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_state = dict(
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch + 1, global_step=global_step,
                best_val_loss=best_val_loss,
                train_losses=train_losses, val_losses=val_losses,
            )
            save_checkpoint(checkpoint_dir / "last.pt", **checkpoint_state)
            if is_best:
                save_checkpoint(checkpoint_dir / "best.pt", **checkpoint_state)
            tqdm.write(f"Checkpoint saved: {checkpoint_dir / 'last.pt'}")

        tqdm.write(
            f"Epoch {epoch + 1}/{epochs}: "
            f"train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}"
        )
        if experiment is not None:
            experiment.log_metrics(
                {
                    "train/epoch_loss": epoch_loss,
                    "validation/loss": val_loss,
                    **{
                        f"train/next_lr/group_{index}": group["lr"]
                        for index, group in enumerate(optimizer.param_groups)
                    },
                },
                step=global_step,
                epoch=epoch + 1,
            )

    if code_logger is not None and global_step > 1 and global_step % log_code_every_n_steps != 0:
        code_logger(model, step=global_step, epoch=len(train_losses))

    return train_losses, val_losses
