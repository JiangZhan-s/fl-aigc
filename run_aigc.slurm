#!/bin/bash
#SBATCH -J DRL_FL_Job           # 作业名
#SBATCH -p gpuA800              # [截图确认] 分区名，必须填这个
#SBATCH --qos=normal            # [截图确认] 服务质量，保持默认
#SBATCH -A a_xmwang             # [截图确认] 账户名 (非常重要！缺少这个可能导致权限拒绝)
#SBATCH -N 1                    # 节点数
#SBATCH -n 1                    # 任务数
#SBATCH --gres=gpu:1            # 单张 A800；通过同卡多进程并发提高利用率
#SBATCH --cpus-per-task=32      # 多个 FL 进程 + DataLoader workers 需要更多 CPU
#SBATCH --mem=128G             # [建议] 内存也稍微给大点，防止多线程数据加载 OOM
#SBATCH --time=48:00:00         # [修改] 截图默认30分钟太短，这里改为48小时
#SBATCH -o logs/job_%j.out      # 标准输出日志
#SBATCH -e logs/job_%j.err      # 错误日志

# 1. 确保当前目录是提交脚本的目录 (项目根目录)
cd $SLURM_SUBMIT_DIR
# 2. 准备日志文件夹 (防止报错)
mkdir -p logs
mkdir -p save
mkdir -p ppo_tensorboard_cifar100
mkdir -p evaluation_results

# 3. 加载环境
module purge                   # 清除默认模块
module load cuda/12.3         # 加载 CUDA (版本需匹配你的 PyTorch)

source ~/.bashrc

# 4. 激活虚拟环境
# 有些集群需要先 source conda.sh，如果 source activate 报错，尝试用 conda activate
source activate aigc39 || conda activate aigc39
# =========================================================
# [核心修复] 配置离线模式与缓存路径
# 告诉 HuggingFace 直接去项目目录下的 hf_cache 找文件，不要联网
# =========================================================
export HF_HOME=$SLURM_SUBMIT_DIR/hf_cache
export HF_DATASETS_CACHE=$SLURM_SUBMIT_DIR/hf_cache
export HF_HUB_OFFLINE=1  # 强制离线模式，双重保险
# 5. 调试信息
echo "=== 运行环境信息 ==="
echo "节点: $(hostname)"
echo "目录: $(pwd)"
echo "显卡: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "===================="

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

# =========================
# A800 并发 FL sweep 配置
# =========================
DATASET=cifar10
ROUNDS=200
TAGS="0p7"
METHODS="proposed_active_set no_aigc random_incentive binary_aigc fixed_price data_size_proportional quality_gap_proportional"

# A800 80G 可以从 4 开始试；如果 nvidia-smi 仍然很空，可改成 6 或 8。
MAX_JOBS=4

# 每个 Python 进程的 DataLoader worker 数。MAX_JOBS * NUM_WORKERS 不要明显超过 CPU 核数。
NUM_WORKERS=4
BATCH_SIZE=128
PREFETCH_FACTOR=4

echo "DATASET=${DATASET}"
echo "TAGS=${TAGS}"
echo "METHODS=${METHODS}"
echo "MAX_JOBS=${MAX_JOBS}, NUM_WORKERS=${NUM_WORKERS}, BATCH_SIZE=${BATCH_SIZE}"

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
      --output-dir outputs/${DATASET}_alpha_sweep/alpha_${tag}/fl/${method} \
      --fast-gpu \
      --num-workers ${NUM_WORKERS} \
      --prefetch-factor ${PREFETCH_FACTOR} \
      --batch-size ${BATCH_SIZE} \
      > logs/${DATASET}_${tag}_${method}_${SLURM_JOB_ID}.out \
      2> logs/${DATASET}_${tag}_${method}_${SLURM_JOB_ID}.err &
  done
done

wait


echo "Job End Time: $(date)"

# # ... (激活环境代码之后)

# # [关键] 设置环境变量指向你刚才解压的位置
# # $SLURM_SUBMIT_DIR 就是你提交作业的目录，即项目根目录
# export HF_HOME=$SLURM_SUBMIT_DIR/hf_cache
# export HF_DATASETS_CACHE=$SLURM_SUBMIT_DIR/hf_cache

# # 强制离线模式 (可选，双重保险)
# export HF_HUB_OFFLINE=1

# echo "Using HF Cache at: $HF_HOME"

# python -u main_drl.py ...
