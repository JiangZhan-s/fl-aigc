"""AIGC-proxy label-distribution augmentation.

This module does not train or sample from a real generative model. It simulates
AIGC-style augmentation by reusing global training-set indices from desired
classes, with replacement, so client label distributions move toward the global
label distribution according to q_k.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.data.quality import label_distribution, tvd_lambda


@dataclass
class AIGCProxyResult:
    """Result of index-level AIGC-proxy augmentation."""

    client_indices_aug: list
    added_counts: list
    augmented_lambdas: list
    target_distributions: list


def _to_numpy_labels(labels: Sequence[int]) -> np.ndarray:
    return np.asarray(labels, dtype=np.int64)


def _class_pools(labels: np.ndarray, num_classes: int) -> list:
    return [np.where(labels == class_id)[0] for class_id in range(num_classes)]


def _target_additions(
    counts: np.ndarray,
    target_dist: np.ndarray,
    num_original: int,
) -> np.ndarray:
    if num_original == 0:
        return np.zeros_like(counts, dtype=np.int64)

    positive = target_dist > 0
    if not np.any(positive):
        return np.zeros_like(counts, dtype=np.int64)

    target_total = num_original
    target_total = max(target_total, float(np.max(counts[positive] / target_dist[positive])))
    desired = np.ceil(target_total * target_dist).astype(np.int64)
    additions = np.maximum(desired - counts, 0)
    return additions


def aigc_proxy_augment(
    dataset,
    client_indices: Sequence[Sequence[int]],
    labels: Sequence[int],
    q_values: Sequence[float],
    lambda_values: Sequence[float],
    num_classes: int,
    global_dist: Sequence[float],
    seed: int = 0,
    eps: float = 1e-12,
) -> AIGCProxyResult:
    """Augment client index lists by moving label distributions toward global.

    Original samples are not modified. Added samples are indices drawn with
    replacement from same-label global pools, acting as synthetic/proxy samples.
    """
    labels_array = _to_numpy_labels(labels)
    global_dist_array = np.asarray(global_dist, dtype=np.float64)
    q_array = np.asarray(q_values, dtype=np.float64)
    lambda_array = np.asarray(lambda_values, dtype=np.float64)

    if len(dataset) != len(labels_array):
        raise ValueError("dataset and labels must have the same length")
    if len(client_indices) != len(q_array) or len(client_indices) != len(lambda_array):
        raise ValueError("client_indices, q_values, and lambda_values must align")
    if len(global_dist_array) != num_classes:
        raise ValueError("global_dist length must equal num_classes")

    rng = np.random.default_rng(seed)
    pools = _class_pools(labels_array, num_classes)

    client_indices_aug = []
    added_counts = []
    augmented_lambdas = []
    target_distributions = []

    for indices, q, lambda_k in zip(client_indices, q_array, lambda_array):
        original = list(indices)
        current_dist = label_distribution(labels_array, original, num_classes)
        improvement_ratio = float(np.clip(q / max(lambda_k, eps), 0.0, 1.0))
        target_dist = (1.0 - improvement_ratio) * current_dist + improvement_ratio * global_dist_array

        counts = np.bincount(
            labels_array[np.asarray(original, dtype=np.int64)],
            minlength=num_classes,
        ).astype(np.int64)[:num_classes]

        additions_by_class = _target_additions(counts, target_dist, len(original))
        synthetic_indices = []
        for class_id, num_to_add in enumerate(additions_by_class):
            if num_to_add <= 0:
                continue
            pool = pools[class_id]
            if len(pool) == 0:
                continue
            sampled = rng.choice(pool, size=int(num_to_add), replace=True)
            synthetic_indices.extend(sampled.astype(int).tolist())

        augmented = original + synthetic_indices
        augmented_dist = label_distribution(labels_array, augmented, num_classes)
        augmented_lambda = tvd_lambda(augmented_dist, global_dist_array)
        original_lambda = tvd_lambda(current_dist, global_dist_array)
        if augmented_lambda > original_lambda:
            augmented = original
            synthetic_indices = []
            augmented_lambda = original_lambda

        client_indices_aug.append(augmented)
        added_counts.append(len(synthetic_indices))
        augmented_lambdas.append(augmented_lambda)
        target_distributions.append(target_dist)

    return AIGCProxyResult(
        client_indices_aug=client_indices_aug,
        added_counts=added_counts,
        augmented_lambdas=augmented_lambdas,
        target_distributions=target_distributions,
    )


def augment_client_indices(*args, **kwargs) -> AIGCProxyResult:
    """Alias for aigc_proxy_augment."""
    return aigc_proxy_augment(*args, **kwargs)


def augment_with_aigc_proxy(*args, **kwargs) -> AIGCProxyResult:
    """Alias for aigc_proxy_augment."""
    return aigc_proxy_augment(*args, **kwargs)
