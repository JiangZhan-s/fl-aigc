#!/bin/bash
#SBATCH -J FL_CIFAR100_REAL
#SBATCH -p gpuA800
#SBATCH --qos=normal
#SBATCH -A a_xmwang
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH -o logs/job_%j.out
#SBATCH -e logs/job_%j.err

cd $SLURM_SUBMIT_DIR
mkdir -p logs

module purge
module load cuda/12.3

source ~/.bashrc
source activate aigc39 || conda activate aigc39

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

echo "=== FL-AIGC CIFAR100 real AIGC sweep ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Workdir: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
nvidia-smi
echo "========================================"

DATASET=cifar100
ROUNDS=300
TAGS="0p3 0p5 0p7 1"
METHODS="no_aigc random_incentive binary_aigc fixed_price data_size_proportional quality_gap_proportional proposed_active_set"

AIGC_ROOT=./aigc_imgs
AIGC_CACHE=${AIGC_CACHE:-./aigc_imgs/cifar100_styleganxl/tensor_cache.pt}

# Use the prebuilt StyleGAN-XL tensor cache. Do not silently rebuild from image
# files here, because generation and cache validation should be completed first.
if [ ! -f "${AIGC_CACHE}" ]; then
  echo "ERROR: missing CIFAR100 StyleGAN-XL AIGC cache: ${AIGC_CACHE}"
  echo "Generate/build it first with run_generate_cifar100_styleganxl.slurm"
  echo "or submit with: sbatch --export=ALL,AIGC_CACHE=/absolute/path/tensor_cache.pt run_aigc_cifar100_real.slurm"
  exit 1
fi

MAX_JOBS=4
NUM_WORKERS=4
PREFETCH_FACTOR=4
BATCH_SIZE=128

echo "DATASET=${DATASET}"
echo "TAGS=${TAGS}"
echo "METHODS=${METHODS}"
echo "MAX_JOBS=${MAX_JOBS}, NUM_WORKERS=${NUM_WORKERS}, BATCH_SIZE=${BATCH_SIZE}"
echo "AIGC_CACHE=${AIGC_CACHE}"

for tag in ${TAGS}; do
  for method in ${METHODS}; do
    while [ "$(jobs -rp | wc -l)" -ge "${MAX_JOBS}" ]; do
      sleep 10
    done

    echo "[$(date)] launch real-aigc dataset=${DATASET} alpha=${tag} method=${method}"
    python -m src.experiments.run_fl \
      --config outputs/config_sweeps/${DATASET}/alpha_${tag}.yaml \
      --method ${method} \
      --rounds ${ROUNDS} \
      --clients 50 \
      --subset-size 0 \
      --output-dir outputs/${DATASET}_styleganxl_aigc_alpha_sweep_0612/alpha_${tag}/fl/${method} \
      --real-aigc \
      --aigc-root "${AIGC_ROOT}" \
      --aigc-cache "${AIGC_CACHE}" \
      --fast-gpu \
      --num-workers ${NUM_WORKERS} \
      --prefetch-factor ${PREFETCH_FACTOR} \
      --batch-size ${BATCH_SIZE} \
      --progress \
      > logs/${DATASET}_styleganxl_real_${tag}_${method}_${SLURM_JOB_ID}.out \
      2> logs/${DATASET}_styleganxl_real_${tag}_${method}_${SLURM_JOB_ID}.err &
  done
done

wait
echo "Job End Time: $(date)"
