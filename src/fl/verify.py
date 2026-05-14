"""Public-validation contribution scores for payment calibration."""

import copy
import math
from typing import Sequence

import torch


def flatten_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Flatten tensors into one CPU float vector."""
    if not tensors:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat([tensor.detach().float().cpu().reshape(-1) for tensor in tensors])


def state_dict_delta(base_state, updated_state):
    """Compute updated - base for floating-point entries in a state_dict."""
    delta = {}
    for key, base_value in base_state.items():
        updated_value = updated_state[key]
        if torch.is_floating_point(base_value):
            delta[key] = updated_value.detach().cpu() - base_value.detach().cpu()
    return delta


def flatten_state_dict_delta(delta_state) -> torch.Tensor:
    """Flatten a delta state_dict into one vector."""
    return flatten_tensors([delta_state[key] for key in sorted(delta_state)])


def apply_delta_to_model(model, delta_state, scale: float = 1.0):
    """Return a copied model with delta_state added to matching parameters."""
    updated = copy.deepcopy(model)
    state = updated.state_dict()
    for key, delta in delta_state.items():
        if key in state and torch.is_floating_point(state[key]):
            state[key] = state[key] + delta.to(state[key].device) * scale
    updated.load_state_dict(state)
    return updated


def validation_loss(model, dataloader, criterion, device) -> float:
    """Compute validation loss without updating model parameters."""
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return 0.0
    return total_loss / total_samples


def validation_loss_drop_scores(
    model,
    delta_states,
    dataloader,
    criterion,
    device,
) -> torch.Tensor:
    """Compute normalized positive validation-loss-drop scores."""
    base_loss = validation_loss(model, dataloader, criterion, device)
    drops = []
    for delta_state in delta_states:
        updated_model = apply_delta_to_model(model, delta_state)
        updated_loss = validation_loss(updated_model, dataloader, criterion, device)
        drops.append(max(base_loss - updated_loss, 0.0))

    scores = torch.tensor(drops, dtype=torch.float32)
    max_score = torch.max(scores) if scores.numel() > 0 else torch.tensor(0.0)
    if max_score <= 0:
        return torch.zeros_like(scores)
    return torch.clamp(scores / max_score, 0.0, 1.0)


def public_gradient_vector(model, dataloader, criterion, device) -> torch.Tensor:
    """Compute flattened gradient on the public validation set."""
    model.to(device)
    model.train()
    model.zero_grad(set_to_none=True)

    total_samples = 0
    total_loss = None
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        batch_size = targets.size(0)
        weighted_loss = loss * batch_size
        total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        total_samples += batch_size

    if total_samples == 0 or total_loss is None:
        return torch.empty(0, dtype=torch.float32)

    (total_loss / total_samples).backward()
    grads = []
    for parameter in model.parameters():
        if parameter.grad is not None:
            grads.append(parameter.grad.detach().cpu())
    model.zero_grad(set_to_none=True)
    return flatten_tensors(grads)


def geodesic_score_from_vectors(reference_gradient, delta_vector, eps: float = 1e-12) -> float:
    """Compute 1 - arccos(cos(-g_pub, delta_w)) / pi."""
    ref = reference_gradient.detach().float().cpu().reshape(-1)
    delta = delta_vector.detach().float().cpu().reshape(-1)
    ref_norm = torch.linalg.vector_norm(ref)
    delta_norm = torch.linalg.vector_norm(delta)
    if ref_norm <= eps or delta_norm <= eps:
        return 0.0

    cos = torch.dot(-ref, delta) / (ref_norm * delta_norm)
    cos = torch.clamp(cos, -1.0, 1.0)
    dist = torch.arccos(cos)
    score = 1.0 - dist / math.pi
    return float(torch.clamp(score, 0.0, 1.0))


def geodesic_scores(reference_gradient, delta_vectors) -> torch.Tensor:
    """Compute geodesic scores for multiple client update vectors."""
    return torch.tensor(
        [geodesic_score_from_vectors(reference_gradient, delta) for delta in delta_vectors],
        dtype=torch.float32,
    )


def geodesic_scores_from_model(
    model,
    delta_states,
    dataloader,
    criterion,
    device,
) -> torch.Tensor:
    """Compute public-gradient geodesic scores for state_dict deltas."""
    gradient = public_gradient_vector(model, dataloader, criterion, device)
    delta_vectors = [flatten_state_dict_delta(delta_state) for delta_state in delta_states]
    return geodesic_scores(gradient, delta_vectors)


def gamma_clip(psi, tau: float = 0.5):
    """Default Gamma(psi)=clip(psi/tau, 0, 1)."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    return torch.clamp(torch.as_tensor(psi, dtype=torch.float32) / tau, 0.0, 1.0)


def calibrate_rewards(theory_rewards, psi, tau: float = 0.5) -> torch.Tensor:
    """Scale theoretical rewards by Gamma(psi)."""
    rewards = torch.as_tensor(theory_rewards, dtype=torch.float32)
    gamma = gamma_clip(psi, tau=tau)
    return rewards * gamma
