"""Visualize original dataset samples next to real AIGC samples."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import get_dataset
from src.experiments.run_fl import _extract_labels


FMNIST_CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616])
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _denormalize_fmnist(tensor):
    return tensor.squeeze(0).numpy() * 0.3530 + 0.2860


def _denormalize_cifar10(tensor):
    array = tensor.permute(1, 2, 0).numpy()
    return np.clip(array * CIFAR10_STD + CIFAR10_MEAN, 0.0, 1.0)


def _plot_row(axs, row, images, title, samples_per_class, cmap=None):
    axs[row, 0].set_ylabel(title, fontsize=9)
    for col, image in enumerate(images[:samples_per_class]):
        ax = axs[row, col]
        ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])


def _metadata_class_dirs(aigc_root, dataset_name, class_names):
    dataset_root = Path(aigc_root) / dataset_name
    metadata_path = dataset_root / "metadata.json"
    if not metadata_path.exists():
        return {class_id: dataset_root / name for class_id, name in enumerate(class_names)}

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    classes = metadata.get("classes", [])
    if classes and isinstance(classes[0], dict):
        return {
            int(item["id"]): dataset_root / str(item["directory"])
            for item in classes
        }
    return {class_id: dataset_root / str(name) for class_id, name in enumerate(classes)}


def _sample_aigc_file_images(aigc_root, dataset_name, class_names, samples_per_class, rng, size, mode):
    class_dirs = _metadata_class_dirs(aigc_root, dataset_name, class_names)
    resize = transforms.Resize(size)
    rows = []
    for class_id in range(len(class_names)):
        class_dir = class_dirs[class_id]
        paths = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        selected = rng.choice(paths, size=samples_per_class, replace=len(paths) < samples_per_class)
        images = []
        for path in selected:
            with Image.open(path) as image:
                image = resize(image.convert(mode))
                array = np.asarray(image, dtype=np.float32) / 255.0
                images.append(array)
        rows.append(images)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Visualize original samples and real AIGC samples")
    parser.add_argument("--dataset", default="fmnist", choices=["fmnist", "cifar10"])
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--aigc-cache", default="./aigc_imgs/fmnist/tensor_cache.pt")
    parser.add_argument("--aigc-root", default="./aigc_imgs")
    parser.add_argument("--output", default=None)
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    dataset = get_dataset(args.dataset, args.data_root, train=True, download=False)
    labels = _extract_labels(dataset)
    if args.dataset == "fmnist":
        class_names = FMNIST_CLASS_NAMES
        cache = torch.load(args.aigc_cache, map_location="cpu")
        aigc_images = cache["images"]
        aigc_labels = cache["labels"].numpy()
        original_image = lambda idx: _denormalize_fmnist(dataset[int(idx)][0])
        generated_rows = []
        for class_id in range(len(class_names)):
            pool = np.where(aigc_labels == class_id)[0]
            selected = rng.choice(pool, size=args.samples_per_class, replace=False)
            generated_rows.append([aigc_images[int(idx)].squeeze(0).numpy() / 255.0 for idx in selected])
        cmap = "gray"
        default_output = "./outputs/fmnist_real_aigc_samples.png"
    else:
        class_names = CIFAR10_CLASS_NAMES
        original_image = lambda idx: _denormalize_cifar10(dataset[int(idx)][0])
        generated_rows = _sample_aigc_file_images(
            args.aigc_root,
            "cifar10",
            class_names,
            args.samples_per_class,
            rng,
            (32, 32),
            "RGB",
        )
        cmap = None
        default_output = "./outputs/cifar10_real_aigc_samples.png"

    rows = len(class_names) * 2
    cols = args.samples_per_class
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 0.95))

    for class_id, class_name in enumerate(class_names):
        real_pool = np.where(labels == class_id)[0]
        real_ids = rng.choice(real_pool, size=args.samples_per_class, replace=False)
        real_images = [original_image(idx) for idx in real_ids]
        generated_images = generated_rows[class_id]

        _plot_row(axs, 2 * class_id, real_images, f"{class_name}\n{args.dataset.upper()}", args.samples_per_class, cmap)
        _plot_row(axs, 2 * class_id + 1, generated_images, "AIGC", args.samples_per_class, cmap)

    for ax in axs.flat:
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)

    fig.tight_layout(pad=0.2)
    output = Path(args.output or default_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    print(f"saved: {output}")
    print(f"saved: {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
