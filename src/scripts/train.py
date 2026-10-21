import torch

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_tokens = 0

    batch_losses = []
    
    for batch in train_loader:
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

        num_tokens = (batch["labels"] != -100).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

        batch_losses.append(loss.item())

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

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        
        epoch_loss, batch_losses = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(epoch_loss)

        val_loss = validation(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss) # нужно как-то смотреть, в зависимости от шедулера, что передавать
    
    return train_losses, val_losses
