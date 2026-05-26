"""Dataset loading helpers."""

from typing import Optional

from torchvision import datasets, transforms


def _default_transform(name: str):
    if name == "fmnist":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )

    if name in {"cifar10", "cifar100"}:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616),
                ),
            ]
        )

    raise ValueError(f"Unsupported dataset: {name}")


def get_dataset(
    name: str,
    root: str,
    train: bool = True,
    download: bool = True,
    transform: Optional[object] = None,
):
    """Return a torchvision dataset by name."""
    normalized_name = name.lower()
    dataset_transform = transform or _default_transform(normalized_name)

    if normalized_name == "fmnist":
        return datasets.FashionMNIST(
            root=root,
            train=train,
            download=download,
            transform=dataset_transform,
        )

    if normalized_name == "cifar10":
        return datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=dataset_transform,
        )

    if normalized_name == "cifar100":
        return datasets.CIFAR100(
            root=root,
            train=train,
            download=download,
            transform=dataset_transform,
        )

    raise ValueError(f"Unsupported dataset: {name}")
