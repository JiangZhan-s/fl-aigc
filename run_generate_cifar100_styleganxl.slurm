#!/bin/bash
#SBATCH -J GEN_C100_SGXL
#SBATCH -p gpuA800
#SBATCH --qos=normal
#SBATCH -A a_xmwang
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH -o logs/job_%j.out
#SBATCH -e logs/job_%j.err

set -eo pipefail

cd $SLURM_SUBMIT_DIR
mkdir -p logs
mkdir -p styleganxl_models
mkdir -p aigc_imgs/cifar100_styleganxl

module purge
module load cuda/12.3

source ~/.bashrc
source activate aigc39 || conda activate aigc39

set -u

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

STYLEGANXL_REPO="${STYLEGANXL_REPO:-${SLURM_SUBMIT_DIR}/stylegan-xl}"
STYLEGANXL_MODEL="${STYLEGANXL_MODEL:-${SLURM_SUBMIT_DIR}/styleganxl_models/imagenet32.pkl}"
OUT_DIR="${OUT_DIR:-${SLURM_SUBMIT_DIR}/aigc_imgs/cifar100_styleganxl}"
CACHE_PATH="${CACHE_PATH:-${OUT_DIR}/tensor_cache.pt}"

IMAGES_PER_CLASS=${IMAGES_PER_CLASS:-500}
BATCH_SIZE=${BATCH_SIZE:-128}
SEED=${SEED:-42}
PYTHON_BIN=${PYTHON_BIN:-python}

echo "=== CIFAR100 StyleGAN-XL ImageNet32 generation ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Workdir: $(pwd)"
echo "STYLEGANXL_REPO: ${STYLEGANXL_REPO}"
echo "STYLEGANXL_MODEL: ${STYLEGANXL_MODEL}"
echo "OUT_DIR: ${OUT_DIR}"
echo "CACHE_PATH: ${CACHE_PATH}"
echo "IMAGES_PER_CLASS: ${IMAGES_PER_CLASS}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "SEED: ${SEED}"
echo "PYTHON_BIN: ${PYTHON_BIN}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
nvidia-smi
echo "=================================================="

if [ ! -f "${STYLEGANXL_REPO}/legacy.py" ]; then
  echo "ERROR: missing StyleGAN-XL repo at ${STYLEGANXL_REPO}"
  echo "Prepare it first: git clone https://github.com/autonomousvision/stylegan-xl.git ${STYLEGANXL_REPO}"
  exit 1
fi

if [ ! -f "${STYLEGANXL_MODEL}" ]; then
  echo "ERROR: missing StyleGAN-XL ImageNet32 checkpoint at ${STYLEGANXL_MODEL}"
  echo "Prepare it first:"
  echo "wget -O ${STYLEGANXL_MODEL} https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet32.pkl"
  exit 1
fi

${PYTHON_BIN} - <<'PY'
import importlib
checks = {
    "torch": "torch",
    "numpy": "numpy",
    "PIL": "pillow",
    "tqdm": "tqdm",
    "dill": "dill",
    "click": "click",
    "scipy": "scipy",
    "requests": "requests",
    "ninja": "ninja",
    "imageio": "imageio",
    "psutil": "psutil",
    "regex": "regex",
    "pyspng": "pyspng",
    "timm": "timm",
}
missing = []
for module_name, package_name in checks.items():
    try:
        importlib.import_module(module_name)
    except Exception:
        missing.append(package_name)
if missing:
    raise SystemExit(
        "Missing Python packages: "
        + ", ".join(missing)
        + "\nInstall in the active conda env, e.g. pip install "
        + " ".join(dict.fromkeys(missing))
    )
print("dependency check: OK")
import timm
if timm.__version__ != "0.4.12":
    raise SystemExit(
        "Incompatible timm version: "
        + timm.__version__
        + "\nStyleGAN-XL official environment uses timm==0.4.12."
        + "\nInstall with: pip uninstall -y timm && pip install timm==0.4.12"
    )
print("timm version check: OK, timm==0.4.12")
PY

${PYTHON_BIN} -u scripts/generate_cifar100_styleganxl_images.py \
  --styleganxl-repo "${STYLEGANXL_REPO}" \
  --network "${STYLEGANXL_MODEL}" \
  --output-dir "${OUT_DIR}" \
  --images-per-class "${IMAGES_PER_CLASS}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --python "${PYTHON_BIN}"

${PYTHON_BIN} -u scripts/build_aigc_cache.py \
  --dataset cifar100 \
  --aigc-root "${OUT_DIR}" \
  --output "${CACHE_PATH}"

${PYTHON_BIN} - <<PY
import torch
cache_path = "${CACHE_PATH}"
cache = torch.load(cache_path, map_location="cpu")
print("cache:", cache_path)
print("dataset:", cache["dataset"])
print("images.shape:", tuple(cache["images"].shape))
print("labels.shape:", tuple(cache["labels"].shape))
counts = {int(k): int(v) for k, v in cache["counts"].items()}
print("num_classes:", len(counts))
print("min_count:", min(counts.values()))
print("max_count:", max(counts.values()))
assert cache["dataset"] == "cifar100"
assert tuple(cache["images"].shape) == (int("${IMAGES_PER_CLASS}") * 100, 3, 32, 32)
assert tuple(cache["labels"].shape) == (int("${IMAGES_PER_CLASS}") * 100,)
assert len(counts) == 100
assert all(v == int("${IMAGES_PER_CLASS}") for v in counts.values())
print("cache validation: OK")
PY

echo "Job End Time: $(date)"
