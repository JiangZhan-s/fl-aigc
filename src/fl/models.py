"""Model definitions for FedAvg experiments."""

import torch
from torch import nn
from torchvision import models


class SmallCNNFMNIST(nn.Module):
    """Small CNN for one-channel 28x28 FashionMNIST images."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class SmallCNNCIFAR(nn.Module):
    """Small CNN for three-channel CIFAR images."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """Return a non-pretrained ResNet-18 adapted for CIFAR-sized images."""
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_model(dataset_name: str, num_classes: int):
    """Build the default model for a supported dataset."""
    name = dataset_name.lower()
    if name in {"fmnist", "fashionmnist", "fashion_mnist"}:
        return SmallCNNFMNIST(num_classes=num_classes)
    if name in {"cifar10", "cifar100"}:
        return SmallCNNCIFAR(num_classes=num_classes)
    raise ValueError(f"Unsupported dataset for model construction: {dataset_name}")
