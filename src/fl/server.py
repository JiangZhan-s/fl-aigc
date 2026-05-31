"""FedAvg server orchestration."""

import copy
import random
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.fl.client import train_local
from src.fl.fedavg import fedavg_state_dicts
from src.utils.metrics import accuracy


@dataclass
class RoundMetrics:
    round: int
    train_loss: float
    test_loss: float
    test_accuracy: float
    num_clients: int


def evaluate(model, dataloader, criterion, device, use_amp: bool = False):
    """Evaluate model loss and accuracy."""
    model.to(device)
    model.eval()
    amp_enabled = bool(use_amp and getattr(device, "type", str(device)) == "cuda")

    total_loss = 0.0
    total_samples = 0
    total_correct = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(inputs)
                loss = criterion(logits, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            total_correct += accuracy(logits, targets) * batch_size

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def _make_optimizer(
    model,
    optimizer_name: str,
    learning_rate: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
):
    """Build a local optimizer."""
    name = optimizer_name.lower()
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


class FedAvgServer:
    """Minimal FedAvg server."""

    def __init__(
        self,
        model,
        train_dataset,
        test_dataset,
        client_indices: Sequence[Sequence[int]],
        device,
        batch_size: int = 64,
        local_epochs: int = 1,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        max_grad_norm: float = 10.0,
        client_fraction: float = 1.0,
        optimizer_name: str = "sgd",
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
        use_amp: bool = False,
        seed: int = 0,
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.client_indices = [list(indices) for indices in client_indices]
        self.device = device
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.client_fraction = client_fraction
        self.optimizer_name = optimizer_name
        self.num_workers = max(0, int(num_workers))
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers and self.num_workers > 0)
        self.prefetch_factor = prefetch_factor if self.num_workers > 0 else None
        self.use_amp = bool(use_amp)
        self.rng = random.Random(seed)
        self.criterion = nn.CrossEntropyLoss()

    def _loader_kwargs(self, shuffle: bool):
        """Return DataLoader kwargs for local and test loaders."""
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }
        if self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def _state_dict_is_finite(self, state_dict):
        """Return whether all floating-point tensors are finite."""
        for value in state_dict.values():
            if torch.is_floating_point(value) and not torch.isfinite(value).all():
                return False
        return True

    def sample_clients(self):
        available = [idx for idx, indices in enumerate(self.client_indices) if len(indices) > 0]
        if not available:
            raise ValueError("No non-empty clients are available")

        num_selected = max(1, int(round(self.client_fraction * len(available))))
        num_selected = min(num_selected, len(available))
        return self.rng.sample(available, num_selected)

    def train_round(self, round_id: int):
        selected_clients = self.sample_clients()
        local_states = []
        local_weights = []
        local_losses = []

        global_state = copy.deepcopy(self.model.state_dict())
        for client_id in selected_clients:
            local_model = copy.deepcopy(self.model)
            local_model.load_state_dict(global_state)
            optimizer = _make_optimizer(
                local_model,
                self.optimizer_name,
                self.learning_rate,
                self.momentum,
                self.weight_decay,
            )
            dataloader = DataLoader(
                Subset(self.train_dataset, self.client_indices[client_id]),
                **self._loader_kwargs(shuffle=True),
            )
            state_dict, train_loss = train_local(
                local_model,
                dataloader,
                optimizer,
                self.criterion,
                self.device,
                self.local_epochs,
                max_grad_norm=self.max_grad_norm,
                use_amp=self.use_amp,
            )
            if not self._state_dict_is_finite(state_dict) or not torch.isfinite(torch.tensor(train_loss)):
                continue
            local_states.append(state_dict)
            local_weights.append(len(self.client_indices[client_id]))
            local_losses.append(train_loss)

        if local_states:
            aggregated = fedavg_state_dicts(local_states, local_weights)
            self.model.load_state_dict(aggregated)

        test_loader = DataLoader(self.test_dataset, **self._loader_kwargs(shuffle=False))
        test_loss, test_acc = evaluate(self.model, test_loader, self.criterion, self.device, self.use_amp)
        weighted_train_loss = (
            sum(loss * weight for loss, weight in zip(local_losses, local_weights)) / sum(local_weights)
            if local_weights
            else 0.0
        )

        return RoundMetrics(
            round=round_id,
            train_loss=weighted_train_loss,
            test_loss=test_loss,
            test_accuracy=test_acc,
            num_clients=len(selected_clients),
        )

    def fit(self, rounds: int):
        """Run FedAvg for a fixed number of communication rounds."""
        history = []
        for round_id in range(1, rounds + 1):
            history.append(self.train_round(round_id))
        return history


def run_fedavg(
    model,
    train_dataset,
    test_dataset,
    client_indices,
    device,
    rounds: int,
    batch_size: int = 64,
    local_epochs: int = 1,
    learning_rate: float = 0.01,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    max_grad_norm: float = 10.0,
    client_fraction: float = 1.0,
    optimizer_name: str = "sgd",
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
    use_amp: bool = False,
    seed: int = 0,
):
    """Convenience function for running a FedAvg experiment."""
    server = FedAvgServer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        client_indices=client_indices,
        device=device,
        batch_size=batch_size,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        client_fraction=client_fraction,
        optimizer_name=optimizer_name,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        use_amp=use_amp,
        seed=seed,
    )
    return server.fit(rounds)
