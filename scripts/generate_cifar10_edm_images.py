"""Generate CIFAR10 images with NVlabs EDM class-conditional generator.

This wraps the official NVlabs/edm generate.py. Clone/copy NVlabs/edm and
download edm-cifar10-32x32-cond-vp.pkl before running in an offline cluster.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


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


DEFAULT_NETWORK_URL = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edm-repo",
        required=True,
        help="Path to a local clone/copy of https://github.com/NVlabs/edm.",
    )
    parser.add_argument(
        "--network",
        default=DEFAULT_NETWORK_URL,
        help="EDM CIFAR10 class-conditional .pkl path or URL.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where class subfolders will be written.",
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=1000,
        help="Number of images to keep for each CIFAR10 class.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable from the EDM environment.",
    )
    return parser.parse_args()


def existing_count(class_dir: Path) -> int:
    return len(list(class_dir.glob("*.png")))


def seed_range(seed_start: int, count: int) -> str:
    if count <= 0:
        raise ValueError("count must be positive")
    return f"{seed_start}-{seed_start + count - 1}"


def save_metadata(output_dir: Path, args: argparse.Namespace) -> None:
    metadata = {
        "dataset": "cifar10",
        "generator": "nvlabs_edm",
        "network": args.network,
        "images_per_class": args.images_per_class,
        "total_images": args.images_per_class * len(CIFAR10_CLASSES),
        "steps": args.steps,
        "seed": args.seed,
        "classes": [
            {"id": class_id, "directory": class_name, "label": class_name}
            for class_id, class_name in enumerate(CIFAR10_CLASSES)
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_edm_generate(
    edm_repo: Path,
    python_exe: str,
    network: str,
    class_id: int,
    class_dir: Path,
    seeds: str,
    batch_size: int,
    steps: int,
) -> None:
    generate_py = edm_repo / "generate.py"
    if not generate_py.exists():
        raise FileNotFoundError(f"EDM generate.py not found: {generate_py}")

    command = [
        python_exe,
        str(generate_py),
        "--outdir",
        str(class_dir),
        "--seeds",
        seeds,
        "--batch",
        str(batch_size),
        "--steps",
        str(steps),
        "--class",
        str(class_id),
        "--network",
        network,
    ]
    subprocess.run(command, cwd=str(edm_repo), check=True)


def main() -> None:
    args = parse_args()
    edm_repo = Path(args.edm_repo).expanduser().resolve()
    if args.output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "aigc_imgs" / "cifar10"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(output_dir, args)

    for class_id, class_name in enumerate(CIFAR10_CLASSES):
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        start_index = existing_count(class_dir)
        if start_index >= args.images_per_class:
            continue

        count = args.images_per_class - start_index
        seed_start = args.seed + class_id * 1_000_000 + start_index
        run_edm_generate(
            edm_repo=edm_repo,
            python_exe=args.python,
            network=args.network,
            class_id=class_id,
            class_dir=class_dir,
            seeds=seed_range(seed_start, count),
            batch_size=args.batch_size,
            steps=args.steps,
        )

    print(f"Done. Images are saved under: {output_dir}")


if __name__ == "__main__":
    main()
