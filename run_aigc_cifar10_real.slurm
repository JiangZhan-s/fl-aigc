#!/bin/bash
#SBATCH -J FL_CIFAR10_REAL
#SBATCH -p gpuA800
#SBATCH --qos=normal
#SBATCH -A a_xmwang
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
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

echo "=== FL-AIGC CIFAR10 real AIGC sweep ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Workdir: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
nvidia-smi
echo "======================================="

DATASET=cifar10
ROUNDS=200
TAGS="0p3 0p5 0p7 1"
METHODS="no_aigc random_incentive binary_aigc fixed_price data_size_proportional quality_gap_proportional proposed_active_set"

AIGC_ROOT=./aigc_imgs
AIGC_CACHE=./aigc_imgs/cifar10/tensor_cache.pt

# Build the tensor cache once if it does not exist. This avoids repeatedly
# opening tens of thousands of generated image files during training.
if [ ! -f "${AIGC_CACHE}" ]; then
  echo "[$(date)] building AIGC tensor cache: ${AIGC_CACHE}"
  python scripts/build_aigc_cache.py \
    --dataset cifar10 \
    --aigc-root "${AIGC_ROOT}" \
    --output "${AIGC_CACHE}"
fi

# A800 80G can usually run several lightweight FL jobs concurrently. If the
# job hits file/process limits, reduce MAX_JOBS or NUM_WORKERS first.
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
      --output-dir outputs/${DATASET}_real_aigc_alpha_sweep/alpha_${tag}/fl/${method} \
      --real-aigc \
      --aigc-root "${AIGC_ROOT}" \
      --aigc-cache "${AIGC_CACHE}" \
      --fast-gpu \
      --num-workers ${NUM_WORKERS} \
      --prefetch-factor ${PREFETCH_FACTOR} \
      --batch-size ${BATCH_SIZE} \
      --progress \
      > logs/${DATASET}_real_${tag}_${method}_${SLURM_JOB_ID}.out \
      2> logs/${DATASET}_real_${tag}_${method}_${SLURM_JOB_ID}.err &
  done
done

wait
echo "Job End Time: $(date)"
