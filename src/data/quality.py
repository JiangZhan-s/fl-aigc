"""Label-distribution quality metrics."""

from typing import List, Sequence

import numpy as np


def _to_numpy_labels(labels: Sequence[int]) -> np.ndarray:
    return np.asarray(labels, dtype=np.int64)


def label_distribution(
    labels: Sequence[int],
    indices: Sequence[int],
    num_classes: int,
) -> np.ndarray:
    """Compute normalized label distribution over selected indices."""
    labels_array = _to_numpy_labels(labels)
    selected = labels_array[np.asarray(indices, dtype=np.int64)]

    if len(selected) == 0:
        return np.zeros(num_classes, dtype=np.float64)

    counts = np.bincount(selected, minlength=num_classes).astype(np.float64)
    return counts[:num_classes] / counts.sum()


def global_label_distribution(labels: Sequence[int], num_classes: int) -> np.ndarray:
    """Compute normalized label distribution over all labels."""
    labels_array = _to_numpy_labels(labels)

    if len(labels_array) == 0:
        return np.zeros(num_classes, dtype=np.float64)

    counts = np.bincount(labels_array, minlength=num_classes).astype(np.float64)
    return counts[:num_classes] / counts.sum()


def tvd_lambda(client_dist: np.ndarray, global_dist: np.ndarray) -> float:
    """Compute total variation distance between two label distributions."""
    return float(0.5 * np.abs(client_dist - global_dist).sum())


def compute_lambdas(
    labels: Sequence[int],
    client_indices: Sequence[Sequence[int]],
    num_classes: int,
) -> List[float]:
    """Compute lambda_k values for each client's label distribution."""
    global_dist = global_label_distribution(labels, num_classes)
    return [
        tvd_lambda(label_distribution(labels, indices, num_classes), global_dist)
        for indices in client_indices
    ]
