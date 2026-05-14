"""Metric helpers."""

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 classification accuracy."""
    if targets.numel() == 0:
        return 0.0

    predictions = logits.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    return correct / targets.numel()

