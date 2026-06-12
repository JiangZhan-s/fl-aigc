"""Visualize CIFAR100 training samples next to StyleGAN-XL synthetic samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


CIFAR100_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR100_STD = (0.2470, 0.2435, 0.2616)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--aigc-root", default="./aigc_imgs/cifar100_styleganxl")
    parser.add_argument("--output-dir", default="./outputs/cifar100_styleganxl_visual")
    parser.add_argument("--samples-per-class", type=int, default=6)
    parser.add_argument("--classes-per-page", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def denormalize_cifar100(tensor):
    import numpy as np

    array = tensor.permute(1, 2, 0).numpy()
    return np.clip(array * np.array(CIFAR100_STD) + np.array(CIFAR100_MEAN), 0.0, 1.0)


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
    return {class_id: aigc_root / class_name for class_id, class_name in enumerate(CIFAR100_CLASSES)}


def extract_labels(dataset):
    import numpy as np

    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels, dtype=np.int64)
    raise ValueError("Dataset does not expose targets or labels")


def sample_generated_rows(aigc_root: Path, samples_per_class: int, rng):
    import numpy as np
    from PIL import Image

    class_dirs = load_class_dirs(aigc_root)
    rows = []
    counts = {}
    for class_id, class_name in enumerate(CIFAR100_CLASSES):
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


def plot_page(
    page_id: int,
    class_ids: list[int],
    real_rows,
    generated_rows,
    output_dir: Path,
    samples_per_class: int,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = len(class_ids) * 2
    cols = samples_per_class
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 0.9))
    if rows == 2:
        axs = np.asarray(axs).reshape(rows, cols)

    for row_base, class_id in enumerate(class_ids):
        class_name = CIFAR100_CLASSES[class_id]
        for col in range(cols):
            axs[2 * row_base, col].imshow(real_rows[class_id][col])
            axs[2 * row_base + 1, col].imshow(generated_rows[class_id][col])
            axs[2 * row_base, col].set_xticks([])
            axs[2 * row_base, col].set_yticks([])
            axs[2 * row_base + 1, col].set_xticks([])
            axs[2 * row_base + 1, col].set_yticks([])
        axs[2 * row_base, 0].set_ylabel(f"{class_id:02d} {class_name}\nCIFAR100", fontsize=8)
        axs[2 * row_base + 1, 0].set_ylabel("StyleGAN-XL", fontsize=8)

    for ax in axs.flat:
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)

    fig.tight_layout(pad=0.25)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"cifar100_styleganxl_page_{page_id:02d}.png"
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")


def main() -> None:
    args = parse_args()

    import numpy as np
    from src.data.datasets import get_dataset

    rng = np.random.default_rng(args.seed)
    dataset = get_dataset("cifar100", args.data_root, train=True, download=args.download)
    labels = extract_labels(dataset)

    real_rows = []
    for class_id in range(len(CIFAR100_CLASSES)):
        pool = np.where(labels == class_id)[0]
        selected = rng.choice(pool, size=args.samples_per_class, replace=False)
        real_rows.append([denormalize_cifar100(dataset[int(idx)][0]) for idx in selected])

    generated_rows, counts = sample_generated_rows(Path(args.aigc_root), args.samples_per_class, rng)
    print(f"generated class counts: min={min(counts.values())}, max={max(counts.values())}")

    output_dir = Path(args.output_dir)
    class_ids = list(range(len(CIFAR100_CLASSES)))
    for page_id, start in enumerate(range(0, len(class_ids), args.classes_per_page)):
        plot_page(
            page_id=page_id,
            class_ids=class_ids[start : start + args.classes_per_page],
            real_rows=real_rows,
            generated_rows=generated_rows,
            output_dir=output_dir,
            samples_per_class=args.samples_per_class,
        )


if __name__ == "__main__":
    main()
