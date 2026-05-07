import torch


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip_norm=None):
    """Run one training epoch and return avg loss."""
    model.train()
    epoch_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(loader)