"""FedAvg aggregation."""

import copy

import torch


def fedavg_state_dicts(state_dicts, weights):
    """Weighted-average model state_dicts by client sample counts."""
    if not state_dicts:
        raise ValueError("state_dicts must not be empty")
    if len(state_dicts) != len(weights):
        raise ValueError("state_dicts and weights must have the same length")

    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ValueError("sum(weights) must be positive")

    averaged = copy.deepcopy(state_dicts[0])
    for key in averaged:
        first_value = state_dicts[0][key]
        if torch.is_floating_point(first_value):
            avg_value = torch.zeros_like(first_value, dtype=first_value.dtype)
            for state_dict, weight in zip(state_dicts, weights):
                avg_value += state_dict[key].to(dtype=first_value.dtype) * (float(weight) / total_weight)
            averaged[key] = avg_value
        else:
            averaged[key] = first_value.clone()

    return averaged
