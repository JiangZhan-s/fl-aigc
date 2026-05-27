"""Generate cached AIGC images with a pretrained text-to-image model.

The default target is CIFAR10. Images are generated at the model's native
resolution and then resized to 32x32 so they can be used as CIFAR-style
synthetic samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
from tqdm import tqdm


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


PROMPTS = {
    "airplane": "a small realistic airplane, centered object, clean background, natural light",
    "automobile": "a realistic car, centered object, clean background, natural light",
    "bird": "a realistic bird, centered object, clean background, natural light",
    "cat": "a realistic cat, centered object, clean background, natural light",
    "deer": "a realistic deer, centered object, clean background, natural light",
    "dog": "a realistic dog, centered object, clean background, natural light",
    "frog": "a realistic frog, centered object, clean background, natural light",
    "horse": "a realistic horse, centered object, clean background, natural light",
    "ship": "a realistic ship, centered object, clean background, natural light",
    "truck": "a realistic truck, centered object, clean background, natural light",
}


NEGATIVE_PROMPT = (
    "text, watermark, logo, label, blurry, distorted, duplicate object, cropped object, "
    "extra limbs, low quality, monochrome"
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
        default=500,
        help="Number of synthetic images to keep for each CIFAR10 class.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--target-size", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    return parser.parse_args()


def existing_count(class_dir: Path) -> int:
    return len(list(class_dir.glob("*.png")))


def save_metadata(output_dir: Path, args: argparse.Namespace) -> None:
    metadata_path = output_dir / "metadata.json"
    metadata = {
        "dataset": "cifar10",
        "model_id": args.model_id,
        "images_per_class": args.images_per_class,
        "height": args.height,
        "width": args.width,
        "target_size": args.target_size,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "classes": CIFAR10_CLASSES,
        "prompts": PROMPTS,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_pipeline(args: argparse.Namespace):
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
    progress = tqdm(
        total=args.images_per_class - start_index,
        desc=f"generate {class_name}",
        leave=False,
    )
    image_index = start_index

    while image_index < args.images_per_class:
        current_batch = min(args.batch_size, args.images_per_class - image_index)
        prompts = [prompt] * current_batch
        negative_prompts = [NEGATIVE_PROMPT] * current_batch
        generators = [
            torch.Generator(device=args.device).manual_seed(base_seed + image_index + offset)
            for offset in range(current_batch)
        ]

        with torch.inference_mode():
            result = pipe(
                prompts,
                negative_prompt=negative_prompts,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generators,
            )

        for image in result.images:
            resized = image.convert("RGB").resize(
                (args.target_size, args.target_size),
                Image.Resampling.LANCZOS,
            )
            resized.save(next_image_path(class_dir, class_name, image_index))
            image_index += 1
            progress.update(1)

    progress.close()


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "aigc_img" / "cifar10"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(output_dir, args)

    pipe = load_pipeline(args)
    for class_id, class_name in enumerate(CIFAR10_CLASSES):
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
