"""Visualize CIFAR10 training samples next to EDM synthetic samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--aigc-root", default="./aigc_imgs/cifar10_edm")
    parser.add_argument("--output", default="./outputs/cifar10_edm_samples.png")
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def denormalize_cifar10(tensor):
    import numpy as np

    array = tensor.permute(1, 2, 0).numpy()
    return np.clip(array * np.array(CIFAR10_STD) + np.array(CIFAR10_MEAN), 0.0, 1.0)


def extract_labels(dataset):
    import numpy as np

    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels, dtype=np.int64)
    raise ValueError("Dataset does not expose targets or labels")


def load_class_dirs(aigc_root: Path):
    metadata_path = aigc_root / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        classes = metadata.get("classes", [])
        if classes and isinstance(classes[0], dict):
            return {
                int(item["id"]): aigc_root / str(item["directory"])
                for item in classes
            }
    return {class_id: aigc_root / class_name for class_id, class_name in enumerate(CIFAR10_CLASSES)}


def sample_generated_rows(aigc_root: Path, samples_per_class: int, rng):
    import numpy as np
    from PIL import Image

    class_dirs = load_class_dirs(aigc_root)
    rows = []
    counts = {}
    for class_id, class_name in enumerate(CIFAR10_CLASSES):
        class_dir = class_dirs.get(class_id, aigc_root / class_name)
        paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not paths:
            raise FileNotFoundError(f"No generated images for class {class_id} {class_name}: {class_dir}")
        counts[class_id] = len(paths)
        selected = rng.choice(paths, size=samples_per_class, replace=len(paths) < samples_per_class)
        images = []
        for path in selected:
            with Image.open(path) as image:
                image = image.convert("RGB").resize((32, 32))
                images.append(np.asarray(image, dtype=np.float32) / 255.0)
        rows.append(images)
    return rows, counts


def plot(real_rows, generated_rows, output: Path, samples_per_class: int) -> None:
    import matplotlib.pyplot as plt

    rows = len(CIFAR10_CLASSES) * 2
    cols = samples_per_class
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 0.9))

    for class_id, class_name in enumerate(CIFAR10_CLASSES):
        for col in range(cols):
            axs[2 * class_id, col].imshow(real_rows[class_id][col])
            axs[2 * class_id + 1, col].imshow(generated_rows[class_id][col])
            axs[2 * class_id, col].set_xticks([])
            axs[2 * class_id, col].set_yticks([])
            axs[2 * class_id + 1, col].set_xticks([])
            axs[2 * class_id + 1, col].set_yticks([])
        axs[2 * class_id, 0].set_ylabel(f"{class_id} {class_name}\nCIFAR10", fontsize=8)
        axs[2 * class_id + 1, 0].set_ylabel("EDM", fontsize=8)

    for ax in axs.flat:
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)

    fig.tight_layout(pad=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    print(f"saved: {output}")
    print(f"saved: {output.with_suffix('.pdf')}")


def main() -> None:
    args = parse_args()

    import numpy as np
    from src.data.datasets import get_dataset

    rng = np.random.default_rng(args.seed)
    dataset = get_dataset("cifar10", args.data_root, train=True, download=args.download)
    labels = extract_labels(dataset)

    real_rows = []
    for class_id in range(len(CIFAR10_CLASSES)):
        pool = np.where(labels == class_id)[0]
        selected = rng.choice(pool, size=args.samples_per_class, replace=False)
        real_rows.append([denormalize_cifar10(dataset[int(idx)][0]) for idx in selected])

    generated_rows, counts = sample_generated_rows(Path(args.aigc_root), args.samples_per_class, rng)
    print(f"generated class counts: min={min(counts.values())}, max={max(counts.values())}")
    plot(real_rows, generated_rows, Path(args.output), args.samples_per_class)


if __name__ == "__main__":
    main()
