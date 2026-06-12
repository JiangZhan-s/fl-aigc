"""Generate CIFAR100 images with StyleGAN-XL ImageNet32.

This script uses a public StyleGAN-XL ImageNet32 checkpoint for offline,
class-conditional generation. It does not train, fine-tune, or use prompts.
Each CIFAR100 fine class is mapped to a semantically close ImageNet class id
and generated into its own class directory as 32x32 RGB PNG files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


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


# CIFAR100 fine-label id -> ImageNet-1k class id. The ids follow the ILSVRC2012
# class index order used by ImageNet conditional generators.
CIFAR100_TO_IMAGENET32 = {
    0: 948,   # apple -> Granny Smith
    1: 1,     # aquarium_fish -> goldfish
    2: 982,   # baby -> person proxy
    3: 294,   # bear -> brown bear
    4: 337,   # beaver -> beaver
    5: 431,   # bed -> studio couch/bed-like indoor furniture proxy
    6: 309,   # bee -> bee
    7: 306,   # beetle -> rhinoceros beetle
    8: 444,   # bicycle -> bicycle-built-for-two
    9: 898,   # bottle -> water bottle
    10: 659,  # bowl -> mixing bowl
    11: 981,  # boy -> ballplayer/person proxy
    12: 839,  # bridge -> suspension bridge
    13: 779,  # bus -> school bus
    14: 323,  # butterfly -> monarch
    15: 354,  # camel -> Arabian camel
    16: 473,  # can -> can opener/can proxy
    17: 483,  # castle -> castle
    18: 314,  # caterpillar -> cockroach/insect proxy
    19: 345,  # cattle -> ox
    20: 559,  # chair -> folding chair
    21: 367,  # chimpanzee -> chimpanzee
    22: 892,  # clock -> wall clock
    23: 980,  # cloud -> volcano/sky-scene proxy
    24: 314,  # cockroach -> cockroach
    25: 831,  # couch -> studio couch
    26: 118,  # crab -> Dungeness crab
    27: 49,   # crocodile -> African crocodile
    28: 968,  # cup -> cup
    29: 51,   # dinosaur -> triceratops
    30: 148,  # dolphin -> killer whale/cetacean proxy
    31: 386,  # elephant -> African elephant
    32: 391,  # flatfish -> coho/fish proxy
    33: 970,  # forest -> alp/landscape proxy
    34: 280,  # fox -> grey fox
    35: 981,  # girl -> ballplayer/person proxy
    36: 333,  # hamster -> hamster
    37: 660,  # house -> mobile home/house proxy
    38: 104,  # kangaroo -> wallaby
    39: 508,  # keyboard -> computer keyboard
    40: 846,  # lamp -> table lamp
    41: 621,  # lawn_mower -> lawn mower
    42: 288,  # leopard -> leopard
    43: 291,  # lion -> lion
    44: 46,   # lizard -> green lizard
    45: 122,  # lobster -> American lobster
    46: 982,  # man -> groom/person proxy
    47: 991,  # maple_tree -> tree/fungus proxy, no maple class
    48: 670,  # motorcycle -> motor scooter
    49: 970,  # mountain -> alp
    50: 673,  # mouse -> mouse/mousetrap proxy
    51: 947,  # mushroom -> mushroom
    52: 988,  # oak_tree -> acorn/oak proxy
    53: 950,  # orange -> orange
    54: 986,  # orchid -> flower proxy
    55: 360,  # otter -> otter
    56: 976,  # palm_tree -> promontory/tropical landscape proxy
    57: 950,  # pear -> fruit proxy
    58: 717,  # pickup_truck -> pickup
    59: 970,  # pine_tree -> mountain/evergreen landscape proxy
    60: 978,  # plain -> seashore/open landscape proxy
    61: 923,  # plate -> plate
    62: 985,  # poppy -> daisy/flower proxy
    63: 334,  # porcupine -> porcupine
    64: 106,  # possum -> wombat/marsupial proxy
    65: 330,  # rabbit -> wood rabbit
    66: 372,  # raccoon -> raccoon
    67: 6,    # ray -> stingray
    68: 757,  # road -> recreational vehicle/road scene proxy
    69: 744,  # rocket -> missile
    70: 985,  # rose -> flower proxy
    71: 978,  # sea -> seashore
    72: 150,  # seal -> sea lion
    73: 2,    # shark -> great white shark
    74: 299,  # shrew -> mongoose/small mammal proxy
    75: 361,  # skunk -> skunk
    76: 821,  # skyscraper -> steel arch bridge/building proxy
    77: 113,  # snail -> snail
    78: 52,   # snake -> thunder snake
    79: 72,   # spider -> garden spider
    80: 335,  # squirrel -> fox squirrel
    81: 829,  # streetcar -> streetcar
    82: 985,  # sunflower -> flower proxy
    83: 945,  # sweet_pepper -> bell pepper
    84: 532,  # table -> dining table
    85: 847,  # tank -> tank
    86: 528,  # telephone -> dial telephone
    87: 851,  # television -> television
    88: 292,  # tiger -> tiger
    89: 866,  # tractor -> tractor
    90: 820,  # train -> steam locomotive
    91: 0,    # trout -> tench/fish proxy
    92: 986,  # tulip -> flower proxy
    93: 33,   # turtle -> loggerhead turtle
    94: 894,  # wardrobe -> wardrobe
    95: 147,  # whale -> grey whale
    96: 970,  # willow_tree -> landscape/tree proxy
    97: 269,  # wolf -> timber wolf
    98: 982,  # woman -> groom/person proxy
    99: 111,  # worm -> nematode
}


DEFAULT_NETWORK_URL = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet32.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--styleganxl-repo",
        required=True,
        help="Path to a local clone/copy of https://github.com/autonomousvision/stylegan-xl.",
    )
    parser.add_argument(
        "--network",
        default=DEFAULT_NETWORK_URL,
        help="StyleGAN-XL ImageNet32 .pkl path or URL.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where CIFAR100 class subfolders will be written.",
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=500,
        help="Number of images to keep for each CIFAR100 class.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for StyleGAN-XL generation.",
    )
    return parser.parse_args()


def existing_count(class_dir: Path) -> int:
    return len(list(class_dir.glob("*.png")))


def save_metadata(output_dir: Path, args: argparse.Namespace) -> None:
    metadata = {
        "dataset": "cifar100",
        "generator": "styleganxl_imagenet32",
        "network": args.network,
        "images_per_class": args.images_per_class,
        "total_images": args.images_per_class * len(CIFAR100_CLASSES),
        "seed": args.seed,
        "classes": [
            {"id": class_id, "directory": class_name, "label": class_name}
            for class_id, class_name in enumerate(CIFAR100_CLASSES)
        ],
        "cifar100_to_imagenet32_mapping": {
            str(class_id): {
                "cifar100_class": CIFAR100_CLASSES[class_id],
                "imagenet_class_id": int(imagenet_id),
            }
            for class_id, imagenet_id in CIFAR100_TO_IMAGENET32.items()
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_generator(styleganxl_repo: Path, network: str):
    sys.path.insert(0, str(styleganxl_repo))

    import dnnlib
    import legacy
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with dnnlib.util.open_url(network) as handle:
        generator = legacy.load_network_pkl(handle)["G_ema"]
    generator = generator.eval().requires_grad_(False).to(device)
    if getattr(generator, "c_dim", 0) <= 0:
        raise ValueError("The loaded StyleGAN-XL network is not class-conditional")
    return generator, device


def generate_batch(generator, device, imagenet_class_id: int, seeds: list[int]):
    import numpy as np
    import torch

    z = np.stack([np.random.RandomState(seed).randn(generator.z_dim) for seed in seeds])
    z = torch.from_numpy(z).to(device=device, dtype=torch.float32)
    c = torch.zeros([len(seeds), generator.c_dim], device=device)
    if not 0 <= imagenet_class_id < generator.c_dim:
        raise ValueError(f"ImageNet class id {imagenet_class_id} is outside generator c_dim={generator.c_dim}")
    c[:, imagenet_class_id] = 1

    with torch.no_grad():
        images = generator(z, c, truncation_psi=1.0, noise_mode="const")
    images = (images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    return images


def save_pngs(images, paths: list[Path]) -> None:
    from PIL import Image

    for image, path in zip(images, paths):
        Image.fromarray(image, "RGB").save(path)


def validate_output(output_dir: Path, images_per_class: int) -> None:
    from PIL import Image

    missing = []
    bad_counts = {}
    bad_images = []
    for class_name in CIFAR100_CLASSES:
        class_dir = output_dir / class_name
        if not class_dir.exists():
            missing.append(class_name)
            continue
        paths = sorted(class_dir.glob("*.png"))
        if len(paths) != images_per_class:
            bad_counts[class_name] = len(paths)
        if paths:
            with Image.open(paths[0]) as image:
                if image.mode != "RGB" or image.size != (32, 32):
                    bad_images.append((class_name, image.mode, image.size))

    if missing or bad_counts or bad_images:
        raise RuntimeError(
            "StyleGAN-XL CIFAR100 output validation failed: "
            f"missing={missing}, bad_counts={bad_counts}, bad_images={bad_images}"
        )


def generate_class_images(
    generator,
    device,
    class_id: int,
    class_name: str,
    class_dir: Path,
    args: argparse.Namespace,
) -> None:
    class_dir.mkdir(parents=True, exist_ok=True)
    start_index = existing_count(class_dir)
    if start_index >= args.images_per_class:
        return

    from tqdm import tqdm

    imagenet_class_id = CIFAR100_TO_IMAGENET32[class_id]
    image_index = start_index
    progress = tqdm(
        total=args.images_per_class - start_index,
        desc=f"{class_id:03d} {class_name}->{imagenet_class_id}",
        leave=False,
    )

    while image_index < args.images_per_class:
        current_batch = min(args.batch_size, args.images_per_class - image_index)
        seed_start = args.seed + class_id * 1_000_000 + image_index
        seeds = list(range(seed_start, seed_start + current_batch))
        paths = [class_dir / f"{class_name}_{idx:05d}.png" for idx in range(image_index, image_index + current_batch)]
        images = generate_batch(generator, device, imagenet_class_id, seeds)
        save_pngs(images, paths)
        image_index += current_batch
        progress.update(current_batch)

    progress.close()


def main() -> None:
    args = parse_args()
    requested_python = Path(shutil.which(args.python) or args.python).expanduser().resolve()
    current_python = Path(sys.executable).resolve()
    if (
        str(requested_python) != str(current_python)
        and os.environ.get("STYLEGANXL_WRAPPER_REEXEC") != "1"
    ):
        env = os.environ.copy()
        env["STYLEGANXL_WRAPPER_REEXEC"] = "1"
        subprocess.run([str(requested_python), str(Path(__file__).resolve()), *sys.argv[1:]], check=True, env=env)
        return

    styleganxl_repo = Path(args.styleganxl_repo).expanduser().resolve()
    if not (styleganxl_repo / "legacy.py").exists():
        raise FileNotFoundError(f"StyleGAN-XL legacy.py not found under: {styleganxl_repo}")

    if args.output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "aigc_imgs" / "cifar100_styleganxl"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(output_dir, args)

    generator, device = load_generator(styleganxl_repo, args.network)
    for class_id, class_name in enumerate(CIFAR100_CLASSES):
        generate_class_images(
            generator=generator,
            device=device,
            class_id=class_id,
            class_name=class_name,
            class_dir=output_dir / class_name,
            args=args,
        )

    validate_output(output_dir, args.images_per_class)
    print(f"Done. Images are saved under: {output_dir}")


if __name__ == "__main__":
    main()
