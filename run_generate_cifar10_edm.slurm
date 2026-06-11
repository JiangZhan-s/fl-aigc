#!/bin/bash
#SBATCH -J GEN_C10_EDM
#SBATCH -p gpuA800
#SBATCH --qos=normal
#SBATCH -A a_xmwang
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o logs/job_%j.out
#SBATCH -e logs/job_%j.err

cd $SLURM_SUBMIT_DIR
mkdir -p logs
mkdir -p edm_models

module purge
module load cuda/12.3

source ~/.bashrc
source activate aigc39 || conda activate aigc39

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

EDM_REPO="${SLURM_SUBMIT_DIR}/edm"
EDM_MODEL="${SLURM_SUBMIT_DIR}/edm_models/edm-cifar10-32x32-cond-vp.pkl"
OUT_DIR="${SLURM_SUBMIT_DIR}/aigc_imgs/cifar10_edm"
CACHE_PATH="${OUT_DIR}/tensor_cache.pt"

IMAGES_PER_CLASS=${IMAGES_PER_CLASS:-6000}
BATCH_SIZE=${BATCH_SIZE:-256}
STEPS=${STEPS:-18}

echo "=== CIFAR10 EDM generation ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Workdir: $(pwd)"
echo "EDM_REPO: ${EDM_REPO}"
echo "EDM_MODEL: ${EDM_MODEL}"
echo "OUT_DIR: ${OUT_DIR}"
echo "IMAGES_PER_CLASS: ${IMAGES_PER_CLASS}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "STEPS: ${STEPS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
nvidia-smi
echo "=============================="

if [ ! -f "${EDM_REPO}/generate.py" ]; then
  echo "ERROR: missing EDM repo at ${EDM_REPO}"
  echo "Prepare it first: git clone https://github.com/NVlabs/edm.git ${EDM_REPO}"
  exit 1
fi

if [ ! -f "${EDM_MODEL}" ]; then
  echo "ERROR: missing EDM model at ${EDM_MODEL}"
  echo "Prepare it first:"
  echo "wget -O ${EDM_MODEL} https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl"
  exit 1
fi

python -u scripts/generate_cifar10_edm_images.py \
  --edm-repo "${EDM_REPO}" \
  --network "${EDM_MODEL}" \
  --output-dir "${OUT_DIR}" \
  --images-per-class "${IMAGES_PER_CLASS}" \
  --batch-size "${BATCH_SIZE}" \
  --steps "${STEPS}"

python scripts/build_aigc_cache.py \
  --dataset cifar10 \
  --aigc-root "${OUT_DIR}" \
  --output "${CACHE_PATH}"

echo "Job End Time: $(date)"
