import torch
from src.utils.logging import gradient_norm

def train_one_epoch(
    model, train_loader, criterion, optimizer, device,
    experiment=None, epoch=0, start_step=0, log_every_n_steps=50,
):
    model.train()

    total_loss = 0.0
    total_tokens = 0

    batch_losses = []
    
    for batch_index, batch in enumerate(train_loader, start=1):
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
    
    return total_loss / total_tokens, batch_losses

@torch.no_grad()
def validation(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    
    for batch in val_loader:
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
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    
    return total_loss / total_tokens

def train(
    model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device,
    experiment=None, log_every_n_steps=50,
):

    if log_every_n_steps < 1:
        raise ValueError("log_every_n_steps must be positive")

    train_losses = []
    val_losses = []
    global_step = 0

    for epoch in range(epochs):
        
        epoch_loss, batch_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            experiment=experiment,
            epoch=epoch + 1,
            start_step=global_step,
            log_every_n_steps=log_every_n_steps,
        )
        global_step += len(batch_losses)
        train_losses.append(epoch_loss)

        val_loss = validation(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
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
    
    return train_losses, val_losses
