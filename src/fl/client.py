"""Client-side local training."""

import copy

import torch


def train_local(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    epochs: int,
    max_grad_norm=None,
    use_amp: bool = False,
):
    """Train a local client model and return its state_dict and average loss."""
    model.to(device)
    model.train()
    amp_enabled = bool(use_amp and getattr(device, "type", str(device)) == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    total_loss = 0.0
    total_samples = 0

    for _ in range(epochs):
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(inputs)
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()

            if max_grad_norm is not None and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    continue

            scaler.step(optimizer)
            scaler.update()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    state_dict = {
        key: value.detach().cpu().clone()
        for key, value in copy.deepcopy(model.state_dict()).items()
    }
    return state_dict, avg_loss
