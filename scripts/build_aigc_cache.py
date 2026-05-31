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


def _load_class_dirs(dataset_root: Path, num_classes: int):
    metadata_path = dataset_root / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        class_dirs = {}
        for item in metadata.get("classes", []):
            class_id = int(item["id"])
            if 0 <= class_id < num_classes:
                class_dirs[class_id] = dataset_root / str(item["directory"])
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


def main():
    parser = argparse.ArgumentParser(description="Build AIGC tensor cache")
    parser.add_argument("--dataset", default="fmnist", choices=["fmnist"])
    parser.add_argument("--aigc-root", type=Path, default=Path("./aigc_imgs"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or default_cache_path(str(args.aigc_root), args.dataset)
    cache = build_fmnist_cache(args.aigc_root, output)
    print(f"saved: {output}")
    print(f"images: {tuple(cache['images'].shape)}")
    print(f"labels: {tuple(cache['labels'].shape)}")
    print(f"counts: {cache['counts']}")


if __name__ == "__main__":
    main()
