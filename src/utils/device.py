"""Device selection helpers."""

import torch


def get_device() -> torch.device:
    """Return CUDA device when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device = None) -> str:
    """Return a concise device diagnostic string for experiment logs."""
    resolved = device or get_device()
    parts = [
        f"torch_cuda_available={torch.cuda.is_available()}",
        f"selected_device={resolved}",
        f"torch_cuda_version={torch.version.cuda}",
        f"cuda_device_count={torch.cuda.device_count()}",
    ]
    if resolved.type == "cuda" and torch.cuda.is_available():
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        allocated_gb = torch.cuda.memory_allocated(index) / (1024**3)
        reserved_gb = torch.cuda.memory_reserved(index) / (1024**3)
        total_gb = props.total_memory / (1024**3)
        parts.extend(
            [
                f"cuda_index={index}",
                f"cuda_name={props.name}",
                f"cuda_capability={props.major}.{props.minor}",
                f"cuda_memory_allocated_gb={allocated_gb:.3f}",
                f"cuda_memory_reserved_gb={reserved_gb:.3f}",
                f"cuda_memory_total_gb={total_gb:.3f}",
            ]
        )
    return " ".join(parts)
