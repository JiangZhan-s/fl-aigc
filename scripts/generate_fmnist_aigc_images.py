"""Generate cached AIGC images for Fashion-MNIST with a pretrained model.

The script uses a pretrained text-to-image model, then converts images to
28x28 grayscale PNGs so they match the Fashion-MNIST input shape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FMNIST_CLASSES = [
    ("tshirt_top", "T-shirt/top"),
    ("trouser", "trouser"),
    ("pullover", "pullover sweater"),
    ("dress", "dress"),
    ("coat", "coat"),
    ("sandal", "sandal"),
    ("shirt", "shirt"),
    ("sneaker", "sneaker"),
    ("bag", "bag"),
    ("ankle_boot", "ankle boot"),
]


PROMPTS = {
    "tshirt_top": "a single plain T-shirt, centered product photo, clean white background",
    "trouser": "a single pair of trousers, centered product photo, clean white background",
    "pullover": "a single pullover sweater, centered product photo, clean white background",
    "dress": "a single dress, centered product photo, clean white background",
    "coat": "a single coat, centered product photo, clean white background",
    "sandal": "a single sandal, centered product photo, clean white background",
    "shirt": "a single shirt, centered product photo, clean white background",
    "sneaker": "a single sneaker shoe, centered product photo, clean white background",
    "bag": "a single handbag, centered product photo, clean white background",
    "ankle_boot": "a single ankle boot, centered product photo, clean white background",
}


NEGATIVE_PROMPT = (
    "person, mannequin, model wearing item, text, watermark, logo, label, cluttered background, "
    "multiple objects, cropped object, blurry, distorted, low quality"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default="runwayml/stable-diffusion-v1-5",
        help="Hugging Face model id or local model path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where class subfolders will be written.",
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=6000,
        help="Number of synthetic images to keep for each Fashion-MNIST class.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--target-size", type=int, default=28)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def existing_count(class_dir: Path) -> int:
    return len(list(class_dir.glob("*.png")))


def save_metadata(output_dir: Path, args: argparse.Namespace) -> None:
    metadata = {
        "dataset": "fmnist",
        "model_id": args.model_id,
        "images_per_class": args.images_per_class,
        "total_images": args.images_per_class * len(FMNIST_CLASSES),
        "height": args.height,
        "width": args.width,
        "target_size": args.target_size,
        "channels": 1,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "classes": [
            {"id": class_id, "directory": directory, "label": label}
            for class_id, (directory, label) in enumerate(FMNIST_CLASSES)
        ],
        "prompts": PROMPTS,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_pipeline(args: argparse.Namespace):
    import torch
    from diffusers import AutoPipelineForText2Image

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe = pipe.to(args.device)
    if args.device == "cuda":
        pipe.enable_attention_slicing()
    return pipe


def next_image_path(class_dir: Path, class_name: str, index: int) -> Path:
    return class_dir / f"{class_name}_{index:05d}.png"


def to_fmnist_image(image, target_size: int):
    from PIL import Image

    return image.convert("L").resize((target_size, target_size), Image.Resampling.LANCZOS)


def generate_class_images(
    pipe,
    class_name: str,
    class_dir: Path,
    args: argparse.Namespace,
    base_seed: int,
) -> None:
    class_dir.mkdir(parents=True, exist_ok=True)
    start_index = existing_count(class_dir)
    if start_index >= args.images_per_class:
        return

    prompt = PROMPTS[class_name]
    from tqdm import tqdm

    progress = tqdm(
        total=args.images_per_class - start_index,
        desc=f"generate {class_name}",
        leave=False,
    )
    image_index = start_index

    while image_index < args.images_per_class:
        import torch

        current_batch = min(args.batch_size, args.images_per_class - image_index)
        generators = [
            torch.Generator(device=args.device).manual_seed(base_seed + image_index + offset)
            for offset in range(current_batch)
        ]

        with torch.inference_mode():
            result = pipe(
                [prompt] * current_batch,
                negative_prompt=[NEGATIVE_PROMPT] * current_batch,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generators,
            )

        for image in result.images:
            fmnist_image = to_fmnist_image(image, args.target_size)
            fmnist_image.save(next_image_path(class_dir, class_name, image_index))
            image_index += 1
            progress.update(1)

    progress.close()


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    if args.output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "aigc_img" / "fmnist"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(output_dir, args)

    pipe = load_pipeline(args)
    for class_id, (class_name, _) in enumerate(FMNIST_CLASSES):
        generate_class_images(
            pipe=pipe,
            class_name=class_name,
            class_dir=output_dir / class_name,
            args=args,
            base_seed=args.seed + class_id * 100_000,
        )

    print(f"Done. Images are saved under: {output_dir}")


if __name__ == "__main__":
    main()
