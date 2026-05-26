"""Non-IID dataset partitioning."""

from typing import List, Sequence

import numpy as np


def _to_numpy_labels(labels: Sequence[int]) -> np.ndarray:
    return np.asarray(labels, dtype=np.int64)


def _repair_min_size(client_indices: List[List[int]], min_size: int, rng) -> bool:
    """Move a minimal number of samples so every client reaches min_size."""
    if min_size <= 0:
        return True

    sizes = [len(indices) for indices in client_indices]
    deficient = [idx for idx, size in enumerate(sizes) if size < min_size]

    for target in deficient:
        while len(client_indices[target]) < min_size:
            donors = [
                idx
                for idx, indices in enumerate(client_indices)
                if idx != target and len(indices) > min_size
            ]
            if not donors:
                return False

            donor = max(donors, key=lambda idx: len(client_indices[idx]))
            move_pos = int(rng.integers(0, len(client_indices[donor])))
            sample_idx = client_indices[donor].pop(move_pos)
            client_indices[target].append(sample_idx)

    return True


def dirichlet_partition(
    labels: Sequence[int],
    num_clients: int,
    alpha: float,
    min_size: int = 10,
    seed: int = 0,
) -> List[List[int]]:
    """Partition sample indices using label-skew Dirichlet allocation."""
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    labels_array = _to_numpy_labels(labels)
    num_samples = len(labels_array)
    if num_samples < num_clients * min_size:
        raise ValueError("Not enough samples to satisfy min_size for all clients")

    classes = np.unique(labels_array)
    rng = np.random.default_rng(seed)

    for _ in range(100):
        client_indices = [[] for _ in range(num_clients)]

        for class_id in classes:
            class_indices = np.where(labels_array == class_id)[0]
            rng.shuffle(class_indices)

            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)

            for client_id, split in enumerate(np.split(class_indices, split_points)):
                client_indices[client_id].extend(split.tolist())

        if _repair_min_size(client_indices, min_size, rng):
            sizes = [len(indices) for indices in client_indices]
        else:
            sizes = [len(indices) for indices in client_indices]

        if min(sizes) >= min_size:
            for indices in client_indices:
                rng.shuffle(indices)
            return client_indices

    raise RuntimeError("Failed to create a valid Dirichlet partition after 100 attempts")
