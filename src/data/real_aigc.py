"""Real AIGC image augmentation for FL clients.

This module consumes an offline, class-conditional synthetic image pool. It
does not call any external AIGC service during training. The mechanism q_k
still decides how much each client should move toward the global label
distribution; real generated images replace the index-resampling proxy.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms

from src.data.aigc_proxy import _target_additions
from src.data.quality import label_distribution, tvd_lambda


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class RealAIGCResult:
    """Result of real AIGC image augmentation."""

    dataset: Dataset
    client_indices_aug: list
    added_counts: list
    augmented_lambdas: list
    target_distributions: list
    synthetic_labels: list


class LabeledImagePoolDataset(Dataset):
    """Dataset backed by generated image file paths and integer labels."""

    def __init__(self, samples, transform=None, mode: str = "L"):
        self.samples = list(samples)
        self.transform = transform
        self.mode = mode

    def __len__(self):
        """Return number of synthetic samples."""
        return len(self.samples)

    def __getitem__(self, index):
        """Load one generated image and return image tensor plus label."""
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert(self.mode)
            if self.transform is not None:
                image = self.transform(image)
        return image, int(label)


def _fmnist_real_transform():
    """Return transform for generated FMNIST images."""
    return transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )


def _load_class_dirs(aigc_root: Path, dataset_name: str, num_classes: int):
    """Load class-id to directory mapping from metadata or numeric folders."""
    dataset_root = aigc_root / dataset_name
    metadata_path = dataset_root / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        classes = metadata.get("classes", [])
        mapping = {}
        for item in classes:
            class_id = int(item["id"])
            if 0 <= class_id < num_classes:
                mapping[class_id] = dataset_root / str(item["directory"])
        return mapping

    return {class_id: dataset_root / str(class_id) for class_id in range(num_classes)}


def _list_image_pool(aigc_root: Path, dataset_name: str, num_classes: int):
    """Return generated image path pools grouped by class id."""
    class_dirs = _load_class_dirs(aigc_root, dataset_name, num_classes)
    pools = []
    for class_id in range(num_classes):
        class_dir = class_dirs.get(class_id)
        if class_dir is None or not class_dir.exists():
            pools.append([])
            continue
        paths = [
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        pools.append(sorted(paths))
    return pools


def build_real_aigc_augmented_dataset(
    dataset,
    client_indices: Sequence[Sequence[int]],
    labels: Sequence[int],
    q_values: Sequence[float],
    lambda_values: Sequence[float],
    num_classes: int,
    global_dist: Sequence[float],
    aigc_root: str = "./aigc_imgs",
    dataset_name: str = "fmnist",
    seed: int = 0,
    eps: float = 1e-12,
    max_extra_ratio: float = 1.0,
) -> RealAIGCResult:
    """Append real generated images according to mechanism-selected q values."""
    normalized_name = dataset_name.lower()
    if normalized_name not in {"fmnist", "fashionmnist", "fashion_mnist"}:
        raise ValueError("Real AIGC image augmentation is currently implemented for FMNIST only")

    labels_array = np.asarray(labels, dtype=np.int64)
    q_array = np.asarray(q_values, dtype=np.float64)
    lambda_array = np.asarray(lambda_values, dtype=np.float64)
    global_dist_array = np.asarray(global_dist, dtype=np.float64)

    if len(dataset) != len(labels_array):
        raise ValueError("dataset and labels must have the same length")
    if len(client_indices) != len(q_array) or len(client_indices) != len(lambda_array):
        raise ValueError("client_indices, q_values, and lambda_values must align")
    if len(global_dist_array) != num_classes:
        raise ValueError("global_dist length must equal num_classes")
    if max_extra_ratio < 0:
        raise ValueError("max_extra_ratio must be non-negative")

    rng = np.random.default_rng(seed)
    pools = _list_image_pool(Path(aigc_root), "fmnist", num_classes)
    if any(len(pool) == 0 for pool in pools):
        missing = [str(class_id) for class_id, pool in enumerate(pools) if len(pool) == 0]
        raise ValueError(f"Missing real AIGC image pool for class ids: {', '.join(missing)}")

    synthetic_samples = []
    synthetic_labels = []
    client_indices_aug = []
    added_counts = []
    augmented_lambdas = []
    target_distributions = []

    for indices, q, lambda_k in zip(client_indices, q_array, lambda_array):
        original = list(indices)
        current_dist = label_distribution(labels_array, original, num_classes)
        improvement_ratio = float(np.clip(q / max(lambda_k, eps), 0.0, 1.0))
        target_dist = (1.0 - improvement_ratio) * current_dist + improvement_ratio * global_dist_array

        if improvement_ratio <= eps or max_extra_ratio <= 0:
            original_lambda = tvd_lambda(current_dist, global_dist_array)
            client_indices_aug.append(original)
            added_counts.append(0)
            augmented_lambdas.append(original_lambda)
            target_distributions.append(target_dist)
            continue

        counts = np.bincount(
            labels_array[np.asarray(original, dtype=np.int64)],
            minlength=num_classes,
        ).astype(np.int64)[:num_classes]

        additions_by_class = _target_additions(counts, target_dist, len(original))
        max_extra = int(np.ceil(max_extra_ratio * len(original)))
        total_requested = int(additions_by_class.sum())
        if total_requested > max_extra:
            if max_extra <= 0:
                additions_by_class = np.zeros_like(additions_by_class)
            else:
                probabilities = additions_by_class.astype(np.float64) / max(total_requested, 1)
                additions_by_class = rng.multinomial(max_extra, probabilities)

        synthetic_indices = []
        start = len(dataset) + len(synthetic_samples)
        for class_id, num_to_add in enumerate(additions_by_class):
            if num_to_add <= 0:
                continue
            pool = pools[class_id]
            sampled_positions = rng.choice(len(pool), size=int(num_to_add), replace=True)
            for position in sampled_positions:
                synthetic_samples.append((pool[int(position)], class_id))
                synthetic_labels.append(class_id)

        synthetic_indices.extend(range(start, len(dataset) + len(synthetic_samples)))
        augmented = original + synthetic_indices
        combined_labels = np.concatenate([labels_array, np.asarray(synthetic_labels, dtype=np.int64)])
        augmented_lambda = tvd_lambda(label_distribution(combined_labels, augmented, num_classes), global_dist_array)
        original_lambda = tvd_lambda(current_dist, global_dist_array)
        if augmented_lambda > original_lambda:
            synthetic_count = len(synthetic_indices)
            if synthetic_count > 0:
                del synthetic_samples[-synthetic_count:]
                del synthetic_labels[-synthetic_count:]
            augmented = original
            synthetic_indices = []
            augmented_lambda = original_lambda

        client_indices_aug.append(augmented)
        added_counts.append(len(synthetic_indices))
        augmented_lambdas.append(augmented_lambda)
        target_distributions.append(target_dist)

    synthetic_dataset = LabeledImagePoolDataset(
        synthetic_samples,
        transform=_fmnist_real_transform(),
        mode="L",
    )
    augmented_dataset = ConcatDataset([dataset, synthetic_dataset])

    return RealAIGCResult(
        dataset=augmented_dataset,
        client_indices_aug=client_indices_aug,
        added_counts=added_counts,
        augmented_lambdas=augmented_lambdas,
        target_distributions=target_distributions,
        synthetic_labels=synthetic_labels,
    )
