#!/bin/bash
#SBATCH -J FL_FMNIST_NEW
#SBATCH -p gpuA800
#SBATCH --qos=normal
#SBATCH -A a_xmwang
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
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

echo "=== FL-AIGC FMNIST new sweep ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Workdir: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
nvidia-smi
echo "================================"

DATASET=fmnist
ROUNDS=100
TAGS="0p3 0p5"
METHODS="proposed_active_set no_aigc random_incentive binary_aigc fixed_price data_size_proportional quality_gap_proportional"

MAX_JOBS=4
NUM_WORKERS=4
PREFETCH_FACTOR=4
BATCH_SIZE=128

for tag in ${TAGS}; do
  for method in ${METHODS}; do
    while [ "$(jobs -rp | wc -l)" -ge "${MAX_JOBS}" ]; do
      sleep 10
    done

    echo "[$(date)] launch dataset=${DATASET} alpha=${tag} method=${method}"
    python -m src.experiments.run_fl \
      --config outputs/config_sweeps/${DATASET}/alpha_${tag}.yaml \
      --method ${method} \
      --rounds ${ROUNDS} \
      --clients 50 \
      --subset-size 0 \
      --output-dir outputs/${DATASET}_alpha_sweep_new/alpha_${tag}/fl/${method} \
      --fast-gpu \
      --num-workers ${NUM_WORKERS} \
      --prefetch-factor ${PREFETCH_FACTOR} \
      --batch-size ${BATCH_SIZE} \
      > logs/${DATASET}_new_${tag}_${method}_${SLURM_JOB_ID}.out \
      2> logs/${DATASET}_new_${tag}_${method}_${SLURM_JOB_ID}.err &
  done
done

wait
echo "Job End Time: $(date)"
