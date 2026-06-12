"""Build tensor caches for offline AIGC image pools."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.real_aigc import IMAGE_EXTENSIONS, default_cache_path


CIFAR10_CLASSES = [
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


CIFAR100_CLASSES = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]


def resolve_cifar10_root(aigc_root: Path) -> Path:
    """
    Support both:
    1. aigc_root/cifar10/airplane...
    2. aigc_root/airplane...
    3. aigc_root/cifar10_edm/airplane... when passed directly as aigc_root
    """
    if (aigc_root / "metadata.json").exists():
        return aigc_root

    if any((aigc_root / class_name).exists() for class_name in CIFAR10_CLASSES):
        return aigc_root

    return aigc_root / "cifar10"


def resolve_cifar100_root(aigc_root: Path) -> Path:
    """
    Support both:
    1. aigc_root/cifar100_styleganxl/apple...
    2. aigc_root/cifar100/apple...
    3. aigc_root/apple... when passed directly as aigc_root
    """
    if (aigc_root / "metadata.json").exists():
        return aigc_root

    if any((aigc_root / class_name).exists() for class_name in CIFAR100_CLASSES):
        return aigc_root

    styleganxl_root = aigc_root / "cifar100_styleganxl"
    if styleganxl_root.exists():
        return styleganxl_root

    return aigc_root / "cifar100"


def _load_class_dirs(dataset_root: Path, num_classes: int):
    metadata_path = dataset_root / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        class_dirs = {}
        classes = metadata.get("classes", [])
        for class_id, item in enumerate(classes):
            if isinstance(item, dict):
                mapped_id = int(item["id"])
                directory = str(item["directory"])
            else:
                mapped_id = class_id
                directory = str(item)
            if 0 <= mapped_id < num_classes:
                class_dirs[mapped_id] = dataset_root / directory
        return class_dirs, metadata

    return {class_id: dataset_root / str(class_id) for class_id in range(num_classes)}, {}


def build_fmnist_cache(aigc_root: Path, output_path: Path, num_classes: int = 10):
    """Build a uint8 [N, 1, 28, 28] FMNIST-style tensor cache."""
    dataset_root = aigc_root / "fmnist"
    class_dirs, metadata = _load_class_dirs(dataset_root, num_classes)
    resize = transforms.Resize((28, 28))

    images = []
    labels = []
    counts = {}

    for class_id in range(num_classes):
        class_dir = class_dirs.get(class_id)
        if class_dir is None or not class_dir.exists():
            raise FileNotFoundError(f"Missing AIGC class directory for class {class_id}: {class_dir}")

        paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        counts[class_id] = len(paths)
        for path in tqdm(paths, desc=f"class {class_id}", leave=False):
            with Image.open(path) as image:
                image = resize(image.convert("L"))
                tensor = torch.from_numpy(np.array(image, dtype="uint8")).unsqueeze(0)
            images.append(tensor)
            labels.append(class_id)

    if not images:
        raise RuntimeError(f"No AIGC images found under {dataset_root}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "dataset": "fmnist",
        "images": torch.stack(images, dim=0).contiguous(),
        "labels": torch.tensor(labels, dtype=torch.long),
        "counts": counts,
        "metadata": metadata,
        "format": "uint8_NCHW_1x28x28",
    }
    torch.save(cache, output_path)
    return cache


def build_cifar10_cache(aigc_root: Path, output_path: Path, num_classes: int = 10):
    """Build a uint8 [N, 3, 32, 32] CIFAR10-style tensor cache."""
    dataset_root = resolve_cifar10_root(aigc_root)
    class_dirs, metadata = _load_class_dirs(dataset_root, num_classes)
    resize = transforms.Resize((32, 32))

    images = []
    labels = []
    counts = {}

    for class_id in range(num_classes):
        class_dir = class_dirs.get(class_id)
        if class_dir is None or not class_dir.exists():
            class_name = CIFAR10_CLASSES[class_id] if class_id < len(CIFAR10_CLASSES) else str(class_id)
            fallback_dir = dataset_root / class_name
            if fallback_dir.exists():
                class_dir = fallback_dir
            else:
                raise FileNotFoundError(f"Missing AIGC class directory for class {class_id}: {class_dir}")

        paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        counts[class_id] = len(paths)
        for path in tqdm(paths, desc=f"class {class_id}", leave=False):
            with Image.open(path) as image:
                image = resize(image.convert("RGB"))
                array = np.array(image, dtype="uint8")
                tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
            images.append(tensor)
            labels.append(class_id)

    if not images:
        raise RuntimeError(f"No AIGC images found under {dataset_root}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "dataset": "cifar10",
        "images": torch.stack(images, dim=0).contiguous(),
        "labels": torch.tensor(labels, dtype=torch.long),
        "counts": counts,
        "metadata": metadata,
        "format": "uint8_NCHW_3x32x32",
    }
    torch.save(cache, output_path)
    return cache


def build_cifar100_cache(aigc_root: Path, output_path: Path, num_classes: int = 100):
    """Build a uint8 [N, 3, 32, 32] CIFAR100-style tensor cache."""
    dataset_root = resolve_cifar100_root(aigc_root)
    class_dirs, metadata = _load_class_dirs(dataset_root, num_classes)
    resize = transforms.Resize((32, 32))

    images = []
    labels = []
    counts = {}

    for class_id in range(num_classes):
        class_dir = class_dirs.get(class_id)
        if class_dir is None or not class_dir.exists():
            class_name = CIFAR100_CLASSES[class_id] if class_id < len(CIFAR100_CLASSES) else str(class_id)
            fallback_dir = dataset_root / class_name
            if fallback_dir.exists():
                class_dir = fallback_dir
            else:
                raise FileNotFoundError(f"Missing AIGC class directory for class {class_id}: {class_dir}")

        paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        counts[class_id] = len(paths)
        for path in tqdm(paths, desc=f"class {class_id}", leave=False):
            with Image.open(path) as image:
                image = resize(image.convert("RGB"))
                array = np.array(image, dtype="uint8")
                tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
            images.append(tensor)
            labels.append(class_id)

    if not images:
        raise RuntimeError(f"No AIGC images found under {dataset_root}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "dataset": "cifar100",
        "images": torch.stack(images, dim=0).contiguous(),
        "labels": torch.tensor(labels, dtype=torch.long),
        "counts": counts,
        "metadata": metadata,
        "format": "uint8_NCHW_3x32x32",
    }
    torch.save(cache, output_path)
    return cache


def main():
    parser = argparse.ArgumentParser(description="Build AIGC tensor cache")
    parser.add_argument("--dataset", default="fmnist", choices=["fmnist", "cifar10", "cifar100"])
    parser.add_argument("--aigc-root", type=Path, default=Path("./aigc_imgs"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or default_cache_path(str(args.aigc_root), args.dataset)
    if args.dataset == "fmnist":
        cache = build_fmnist_cache(args.aigc_root, output)
    elif args.dataset == "cifar10":
        cache = build_cifar10_cache(args.aigc_root, output)
    elif args.dataset == "cifar100":
        cache = build_cifar100_cache(args.aigc_root, output)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    print(f"saved: {output}")
    print(f"images: {tuple(cache['images'].shape)}")
    print(f"labels: {tuple(cache['labels'].shape)}")
    print(f"counts: {cache['counts']}")


if __name__ == "__main__":
    main()
