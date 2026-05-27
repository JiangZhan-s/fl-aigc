# FL-AIGC 激励机制实验框架使用文档

本项目实现了一个用于 FL-AIGC 激励机制实验的框架，包含机制优化、baseline 对比、AIGC-proxy 数据增强、FedAvg 训练、公共验证集支付校准和结果绘图工具。

注意：active-set 求解器是启发式算法。它成功时返回一个可行的、模式稳定的解，但不声称全局最优。exact enum 只用于小规模客户端数量。

## 环境

推荐环境：

```text
Python 3.9.24
torch 2.8.0
torchvision 0.23.0
```

在项目目录安装依赖：

```bash
pip install -r requirements.txt
```

如果使用已有 conda 环境：

```bash
conda activate zwy
```

## 项目结构

```text
configs/                  YAML 配置文件
src/data/                 数据加载、Non-IID 划分、lambda 计算、AIGC-proxy
src/mechanism/            激励机制、支付函数和求解器
src/fl/                   FedAvg、模型、验证贡献分数
src/experiments/          可运行实验入口
scripts/plot_results.py   绘图脚本
tests/                    pytest 测试
```

## 快速检查

运行全部测试：

```bash
pytest -q
```

在 `zwy` conda 环境中运行：

```bash
conda run -n zwy python -m pytest -q
```

## 配置文件

主要配置示例：

```bash
configs/fmnist.yaml
```

重要字段：

```yaml
dataset:
  name: fmnist          # fmnist, cifar10, cifar100
  root: ./data
  download: true
  num_clients: 20
  dirichlet_alpha: 0.5
  min_size: 10
  num_classes: 10

mechanism:
  lambda_base: 1.0
  alpha_range: [0.01, 0.1]
  beta_range: [0.01, 0.1]
  S_range: [0.01, 0.1]
  budget_ratio: 0.3
  exact_max_clients: 12
  active_set_max_iter: 30

fl:
  model: smallcnn       # smallcnn 或 resnet18_cifar
  rounds: 20
  local_epochs: 1
  batch_size: 64
  lr: 0.01
  momentum: 0.0
  client_fraction: 1.0

output:
  dir: ./outputs
```

## 机制层实验

该实验只验证机制优化，不运行 FL 训练。流程包括数据划分、lambda 估计、机制参数生成、active-set 求解，以及在小规模 `N <= exact_max_clients` 时运行 exact enum。

```bash
python -m src.experiments.run_mechanism --config configs/fmnist.yaml
```

指定输出目录：

```bash
python -m src.experiments.run_mechanism \
  --config configs/fmnist.yaml \
  --output-dir outputs/mechanism
```

输出文件：

```text
outputs/
  mechanism_summary.json
  mechanism_clients.csv
  mechanism_active_set_clients.csv
  mechanism_exact_enum_clients.csv     # 仅 exact enum 运行时生成
```

核心 summary 指标：

```text
server_utility
budget_used
budget_utilization
participation_rate
avg_q
corr_lambda_q
ir_violations
ic_regret_max
ic_regret_mean
runtime
```

## Baseline 实验

运行全部机制 baseline，并输出统一格式的 CSV/JSON：

```bash
python -m src.experiments.run_baselines --config configs/fmnist.yaml
```

已实现 baseline：

```text
NoAIGC
RandomIncentive
BinaryAIGC
FixedPrice
DataSizeProportional
QualityGapProportional
ProposedActiveSet
```

输出文件：

```text
outputs/baselines/
  baseline_clients.csv
  baseline_summary.json
```

## 端到端 FL 实验

该实验将机制输出连接到 AIGC-proxy 数据增强，然后运行 FedAvg。

```bash
python -m src.experiments.run_fl \
  --config configs/fmnist.yaml \
  --method proposed_active_set
```

CLI 默认 smoke run 参数：

```text
rounds=2
clients=5
subset_size=2000
```

可以手动覆盖：

```bash
python -m src.experiments.run_fl \
  --config configs/fmnist.yaml \
  --method proposed_active_set \
  --rounds 5 \
  --clients 10 \
  --subset-size 5000
```

支持的方法：

```text
no_aigc
random_incentive
binary_aigc
fixed_price
data_size_proportional
quality_gap_proportional
proposed_active_set
```

支持的模型：

```text
smallcnn
resnet18_cifar
```

FMNIST 使用 `smallcnn`。`resnet18_cifar` 适用于 3 通道 CIFAR 输入。

输出文件：

```text
outputs/
  fl_metrics.csv
  fl_<method>_rounds.csv
```

FL 每轮指标：

```text
round
test_accuracy
test_loss
train_loss
avg_lambda_before
avg_lambda_after
budget_used
participation_rate
```

## 绘图

绘图脚本会从已有 CSV/JSON 读取结果。缺少某类输入时，会跳过对应图。

```bash
python scripts/plot_results.py \
  --fl-csv outputs/fl_metrics.csv \
  --baseline-csv outputs/baselines/baseline_clients.csv \
  --baseline-json outputs/baselines/baseline_summary.json \
  --mechanism-csv outputs/mechanism_clients.csv \
  --mechanism-json outputs/mechanism_summary.json \
  --output-dir outputs/figures
```

生成的图片会同时保存为 PNG 和 PDF：

```text
accuracy_vs_rounds
test_loss_vs_rounds
train_loss_vs_rounds
server_utility_vs_budget_ratio
q_vs_lambda
runtime_vs_n
optimality_gap_n_le_12
lambda_before_after
budget_utilization
server_utility_comparison
```

## 论文实验命令清单

以下命令均假设当前目录为项目根目录：

```bash
cd /home/wuyi/newModel/fl_aigc
```

### 1. 环境与代码正确性检查

```bash
conda activate zwy
python --version
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__)"
pytest -q
```

如果不激活环境，也可以使用：

```bash
conda run -n zwy python -m pytest -q
```

### 2. 快速 smoke run

用于确认整条实验链路可以跑通。

```bash
python -m src.experiments.run_mechanism \
  --config configs/fmnist.yaml \
  --output-dir outputs/smoke/mechanism

python -m src.experiments.run_baselines \
  --config configs/fmnist.yaml \
  --output-dir outputs/smoke/baselines

python -m src.experiments.run_fl \
  --config configs/fmnist.yaml \
  --method proposed_active_set \
  --rounds 2 \
  --clients 5 \
  --subset-size 2000 \
  --output-dir outputs/smoke/fl/proposed_active_set
```

### 3. 三个数据集的机制层实验

只跑机制优化，不跑 FL。适合生成 `server utility`、`budget utilization`、`q-lambda`、IR/IC 等表格。

```bash
python -m src.experiments.run_mechanism \
  --config configs/fmnist.yaml \
  --output-dir outputs/fmnist/mechanism

python -m src.experiments.run_mechanism \
  --config configs/cifar10.yaml \
  --output-dir outputs/cifar10/mechanism

python -m src.experiments.run_mechanism \
  --config configs/cifar100.yaml \
  --output-dir outputs/cifar100/mechanism
```

### 4. 三个数据集的 baseline 实验

运行 `NoAIGC`、`RandomIncentive`、`BinaryAIGC`、`FixedPrice`、`DataSizeProportional`、`QualityGapProportional` 和 `ProposedActiveSet`。

```bash
python -m src.experiments.run_baselines \
  --config configs/fmnist.yaml \
  --output-dir outputs/fmnist/baselines

python -m src.experiments.run_baselines \
  --config configs/cifar10.yaml \
  --output-dir outputs/cifar10/baselines

python -m src.experiments.run_baselines \
  --config configs/cifar100.yaml \
  --output-dir outputs/cifar100/baselines
```

### 5. 端到端 FL：ProposedActiveSet

用于论文中的主方法端到端精度曲线。

```bash
python -m src.experiments.run_fl \
  --config configs/fmnist.yaml \
  --method proposed_active_set \
  --rounds 100 \
  --clients 50 \
  --subset-size 0 \
  --output-dir outputs/fmnist/fl/proposed_active_set

python -m src.experiments.run_fl \
  --config configs/cifar10.yaml \
  --method proposed_active_set \
  --rounds 200 \
  --clients 50 \
  --subset-size 0 \
  --output-dir outputs/cifar10/fl/proposed_active_set

python -m src.experiments.run_fl \
  --config configs/cifar100.yaml \
  --method proposed_active_set \
  --rounds 300 \
  --clients 50 \
  --subset-size 0 \
  --output-dir outputs/cifar100/fl/proposed_active_set
```

说明：`--subset-size 0` 表示使用完整训练集。

### 6. 端到端 FL：所有方法对比

FMNIST：

```bash
python -m src.experiments.run_fl --config configs/fmnist.yaml --method no_aigc --rounds 100 --clients 50 --subset-size 0 --output-dir outputs/fmnist/fl/no_aigc
python -m src.experiments.run_fl --config configs/fmnist.yaml --method random_incentive --rounds 100 --clients 50 --subset-size 0 --output-dir outputs/fmnist/fl/random_incentive
python -m src.experiments.run_fl --config configs/fmnist.yaml --method binary_aigc --rounds 100 --clients 50 --subset-size 0 --output-dir outputs/fmnist/fl/binary_aigc
python -m src.experiments.run_fl --config configs/fmnist.yaml --method fixed_price --rounds 100 --clients 50 --subset-size 0 --output-dir outputs/fmnist/fl/fixed_price
python -m src.experiments.run_fl --config configs/fmnist.yaml --method data_size_proportional --rounds 100 --clients 50 --subset-size 0 --output-dir outputs/fmnist/fl/data_size_proportional
python -m src.experiments.run_fl --config configs/fmnist.yaml --method quality_gap_proportional --rounds 100 --clients 50 --subset-size 0 --output-dir outputs/fmnist/fl/quality_gap_proportional
python -m src.experiments.run_fl --config configs/fmnist.yaml --method proposed_active_set --rounds 100 --clients 50 --subset-size 0 --output-dir outputs/fmnist/fl/proposed_active_set
```

CIFAR10：

```bash
python -m src.experiments.run_fl --config configs/cifar10.yaml --method no_aigc --rounds 200 --clients 50 --subset-size 0 --output-dir outputs/cifar10/fl/no_aigc
python -m src.experiments.run_fl --config configs/cifar10.yaml --method random_incentive --rounds 200 --clients 50 --subset-size 0 --output-dir outputs/cifar10/fl/random_incentive
python -m src.experiments.run_fl --config configs/cifar10.yaml --method binary_aigc --rounds 200 --clients 50 --subset-size 0 --output-dir outputs/cifar10/fl/binary_aigc
python -m src.experiments.run_fl --config configs/cifar10.yaml --method fixed_price --rounds 200 --clients 50 --subset-size 0 --output-dir outputs/cifar10/fl/fixed_price
python -m src.experiments.run_fl --config configs/cifar10.yaml --method data_size_proportional --rounds 200 --clients 50 --subset-size 0 --output-dir outputs/cifar10/fl/data_size_proportional
python -m src.experiments.run_fl --config configs/cifar10.yaml --method quality_gap_proportional --rounds 200 --clients 50 --subset-size 0 --output-dir outputs/cifar10/fl/quality_gap_proportional
python -m src.experiments.run_fl --config configs/cifar10.yaml --method proposed_active_set --rounds 200 --clients 50 --subset-size 0 --output-dir outputs/cifar10/fl/proposed_active_set
```

CIFAR100：

```bash
python -m src.experiments.run_fl --config configs/cifar100.yaml --method no_aigc --rounds 200 --clients 50 --subset-size 0 --output-dir outputs/cifar100/fl/no_aigc
python -m src.experiments.run_fl --config configs/cifar100.yaml --method random_incentive --rounds 200 --clients 50 --subset-size 0 --output-dir outputs/cifar100/fl/random_incentive
python -m src.experiments.run_fl --config configs/cifar100.yaml --method binary_aigc --rounds 200 --clients 50 --subset-size 0 --output-dir outputs/cifar100/fl/binary_aigc
python -m src.experiments.run_fl --config configs/cifar100.yaml --method fixed_price --rounds 200 --clients 50 --subset-size 0 --output-dir outputs/cifar100/fl/fixed_price
python -m src.experiments.run_fl --config configs/cifar100.yaml --method data_size_proportional --rounds 200 --clients 50 --subset-size 0 --output-dir outputs/cifar100/fl/data_size_proportional
python -m src.experiments.run_fl --config configs/cifar100.yaml --method quality_gap_proportional --rounds 200 --clients 50 --subset-size 0 --output-dir outputs/cifar100/fl/quality_gap_proportional
python -m src.experiments.run_fl --config configs/cifar100.yaml --method proposed_active_set --rounds 200 --clients 50 --subset-size 0 --output-dir outputs/cifar100/fl/proposed_active_set
```

### 7. 绘制 FMNIST 论文图

该命令会生成 accuracy 曲线、train/test loss 曲线、lambda before/after 图，以及可用的机制图。

```bash
python scripts/plot_results.py \
  --fl-csv outputs/fmnist/fl/no_aigc/fl_metrics.csv \
  --fl-csv outputs/fmnist/fl/random_incentive/fl_metrics.csv \
  --fl-csv outputs/fmnist/fl/binary_aigc/fl_metrics.csv \
  --fl-csv outputs/fmnist/fl/fixed_price/fl_metrics.csv \
  --fl-csv outputs/fmnist/fl/data_size_proportional/fl_metrics.csv \
  --fl-csv outputs/fmnist/fl/quality_gap_proportional/fl_metrics.csv \
  --fl-csv outputs/fmnist/fl/proposed_active_set/fl_metrics.csv \
  --baseline-csv outputs/fmnist/baselines/baseline_clients.csv \
  --baseline-json outputs/fmnist/baselines/baseline_summary.json \
  --mechanism-csv outputs/fmnist/mechanism/mechanism_clients.csv \
  --mechanism-json outputs/fmnist/mechanism/mechanism_summary.json \
  --output-dir outputs/fmnist/figures
```

### 8. 绘制 CIFAR10 论文图

该命令会生成 accuracy 曲线、train/test loss 曲线、lambda before/after 图，以及可用的机制图。

```bash
python scripts/plot_results.py \
  --fl-csv outputs/cifar10/fl/no_aigc/fl_metrics.csv \
  --fl-csv outputs/cifar10/fl/random_incentive/fl_metrics.csv \
  --fl-csv outputs/cifar10/fl/binary_aigc/fl_metrics.csv \
  --fl-csv outputs/cifar10/fl/fixed_price/fl_metrics.csv \
  --fl-csv outputs/cifar10/fl/data_size_proportional/fl_metrics.csv \
  --fl-csv outputs/cifar10/fl/quality_gap_proportional/fl_metrics.csv \
  --fl-csv outputs/cifar10/fl/proposed_active_set/fl_metrics.csv \
  --baseline-csv outputs/cifar10/baselines/baseline_clients.csv \
  --baseline-json outputs/cifar10/baselines/baseline_summary.json \
  --mechanism-csv outputs/cifar10/mechanism/mechanism_clients.csv \
  --mechanism-json outputs/cifar10/mechanism/mechanism_summary.json \
  --output-dir outputs/cifar10/figures
```

### 9. 绘制 CIFAR100 论文图

该命令会生成 accuracy 曲线、train/test loss 曲线、lambda before/after 图，以及可用的机制图。

```bash
python scripts/plot_results.py \
  --fl-csv outputs/cifar100/fl/no_aigc/fl_metrics.csv \
  --fl-csv outputs/cifar100/fl/random_incentive/fl_metrics.csv \
  --fl-csv outputs/cifar100/fl/binary_aigc/fl_metrics.csv \
  --fl-csv outputs/cifar100/fl/fixed_price/fl_metrics.csv \
  --fl-csv outputs/cifar100/fl/data_size_proportional/fl_metrics.csv \
  --fl-csv outputs/cifar100/fl/quality_gap_proportional/fl_metrics.csv \
  --fl-csv outputs/cifar100/fl/proposed_active_set/fl_metrics.csv \
  --baseline-csv outputs/cifar100/baselines/baseline_clients.csv \
  --baseline-json outputs/cifar100/baselines/baseline_summary.json \
  --mechanism-csv outputs/cifar100/mechanism/mechanism_clients.csv \
  --mechanism-json outputs/cifar100/mechanism/mechanism_summary.json \
  --output-dir outputs/cifar100/figures
```

### 10. 输出文件检查

```bash
find outputs -maxdepth 4 -type f | sort
```

查看 summary：

```bash
python -m json.tool outputs/fmnist/mechanism/mechanism_summary.json
python -m json.tool outputs/fmnist/baselines/baseline_summary.json
```

查看 FL 指标前几行：

```bash
python - <<'PY'
import pandas as pd
print(pd.read_csv("outputs/fmnist/fl/proposed_active_set/fl_metrics.csv").head())
PY
```

### 11. Dirichlet alpha 消融实验

该实验用于展示 Non-IID 强度变化对机制和 FL 性能的影响。`dirichlet_alpha` 越小，客户端标签分布越偏；推荐至少跑：

```text
0.05, 0.1, 0.3, 0.5
```

下面命令不会修改原始 `configs/*.yaml`，而是在 `outputs/config_sweeps/` 下生成临时配置。

注意：`0.01` 是极强 Non-IID。若同时使用 `N=50` 和 `min_size=10`，Dirichlet 重采样可能失败。默认命令使用更稳定的 `0.05`；如果必须跑 `0.01`，建议在临时配置中把 `dataset.min_size` 调低到 `1` 或 `2`。

#### 11.1 生成不同 alpha 的临时配置

FMNIST：

```bash
python - <<'PY'
from pathlib import Path
import yaml

base_path = Path("configs/fmnist.yaml")
out_dir = Path("outputs/config_sweeps/fmnist")
out_dir.mkdir(parents=True, exist_ok=True)

for alpha in [0.01, 0.1, 0.3, 0.5]:
    cfg = yaml.safe_load(base_path.read_text())
    cfg.setdefault("dataset", {})["dirichlet_alpha"] = alpha
    tag = str(alpha).replace(".", "p")
    (out_dir / f"alpha_{tag}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
```

CIFAR10：

```bash
python - <<'PY'
from pathlib import Path
import yaml

base_path = Path("configs/cifar10.yaml")
out_dir = Path("outputs/config_sweeps/cifar10")
out_dir.mkdir(parents=True, exist_ok=True)

for alpha in [0.01, 0.05, 0.1, 0.5]:
    cfg = yaml.safe_load(base_path.read_text())
    cfg.setdefault("dataset", {})["dirichlet_alpha"] = alpha
    tag = str(alpha).replace(".", "p")
    (out_dir / f"alpha_{tag}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
```

CIFAR100：

```bash
python - <<'PY'
from pathlib import Path
import yaml

base_path = Path("configs/cifar100.yaml")
out_dir = Path("outputs/config_sweeps/cifar100")
out_dir.mkdir(parents=True, exist_ok=True)

for alpha in [0.01, 0.05, 0.1, 0.3, 0.5]:
    cfg = yaml.safe_load(base_path.read_text())
    cfg.setdefault("dataset", {})["dirichlet_alpha"] = alpha
    tag = str(alpha).replace(".", "p")
    (out_dir / f"alpha_{tag}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
```

#### 11.2 机制层 alpha sweep

这组命令只跑机制和 baseline，不跑 FL，速度较快。适合生成 `server utility`、`budget utilization`、`avg_q`、`lambda-q` 等消融图和表格。

FMNIST：

```bash
for tag in 0p05 0p1 0p3 0p5; do
  python -m src.experiments.run_baselines \
    --config outputs/config_sweeps/fmnist/alpha_${tag}.yaml \
    --output-dir outputs/fmnist_alpha_sweep/alpha_${tag}/baselines
done
```

CIFAR10：

```bash
for tag in 0p05 0p1 0p3 0p5; do
  python -m src.experiments.run_baselines \
    --config outputs/config_sweeps/cifar10/alpha_${tag}.yaml \
    --output-dir outputs/cifar10_alpha_sweep/alpha_${tag}/baselines
done
```

CIFAR100：

```bash
for tag in 0p05 0p1 0p3 0p5; do
  python -m src.experiments.run_baselines \
    --config outputs/config_sweeps/cifar100/alpha_${tag}.yaml \
    --output-dir outputs/cifar100_alpha_sweep/alpha_${tag}/baselines
done
```

绘制机制层 alpha sweep 图：

```bash
python scripts/plot_results.py \
  --baseline-json outputs/fmnist_alpha_sweep/alpha_0p05/baselines/baseline_summary.json \
  --baseline-json outputs/fmnist_alpha_sweep/alpha_0p1/baselines/baseline_summary.json \
  --baseline-json outputs/fmnist_alpha_sweep/alpha_0p3/baselines/baseline_summary.json \
  --baseline-json outputs/fmnist_alpha_sweep/alpha_0p5/baselines/baseline_summary.json \
  --baseline-csv outputs/fmnist_alpha_sweep/alpha_0p3/baselines/baseline_clients.csv \
  --output-dir outputs/fmnist_alpha_sweep/figures

python scripts/plot_results.py \
  --baseline-json outputs/cifar10_alpha_sweep/alpha_0p05/baselines/baseline_summary.json \
  --baseline-json outputs/cifar10_alpha_sweep/alpha_0p1/baselines/baseline_summary.json \
  --baseline-json outputs/cifar10_alpha_sweep/alpha_0p3/baselines/baseline_summary.json \
  --baseline-json outputs/cifar10_alpha_sweep/alpha_0p5/baselines/baseline_summary.json \
  --baseline-csv outputs/cifar10_alpha_sweep/alpha_0p3/baselines/baseline_clients.csv \
  --output-dir outputs/cifar10_alpha_sweep/figures

python scripts/plot_results.py \
  --baseline-json outputs/cifar100_alpha_sweep/alpha_0p05/baselines/baseline_summary.json \
  --baseline-json outputs/cifar100_alpha_sweep/alpha_0p1/baselines/baseline_summary.json \
  --baseline-json outputs/cifar100_alpha_sweep/alpha_0p3/baselines/baseline_summary.json \
  --baseline-json outputs/cifar100_alpha_sweep/alpha_0p5/baselines/baseline_summary.json \
  --baseline-csv outputs/cifar100_alpha_sweep/alpha_0p3/baselines/baseline_clients.csv \
  --output-dir outputs/cifar100_alpha_sweep/figures
```

注意：当前 `plot_results.py` 的 `server_utility_vs_budget_ratio` 横轴固定是 `budget_ratio`。如果用于 alpha sweep，该图不能直接解释为 alpha 横轴；alpha sweep 的机制结果更建议先用 summary 表格汇总，或后续单独扩展绘图脚本。

#### 11.3 端到端 FL alpha sweep：完整方法对比

这组命令用于展示不同 Non-IID 强度下的完整 FL 方法对比。每个 `dirichlet_alpha` 都跑五个方法：

```text
no_aigc
random_incentive
binary_aigc
fixed_price
data_size_proportional
quality_gap_proportional
proposed_active_set
```

为了节省时间，可以先跑 CIFAR10；论文主图稳定后再补 FMNIST 和 CIFAR100。

FMNIST：

```bash
for tag in 0p01 0p05 0p1 0p3 0p5; do
  for method in no_aigc random_incentive binary_aigc fixed_price data_size_proportional quality_gap_proportional proposed_active_set; do
    python -m src.experiments.run_fl \
      --config outputs/config_sweeps/fmnist/alpha_${tag}.yaml \
      --method ${method} \
      --rounds 100 \
      --clients 50 \
      --subset-size 0 \
      --output-dir outputs/fmnist_alpha_sweep/alpha_${tag}/fl/${method}
  done
done
```

CIFAR10：

```bash
for tag in 0p01 0p05 0p5; do
  for method in no_aigc random_incentive binary_aigc fixed_price data_size_proportional quality_gap_proportional proposed_active_set; do
    python -m src.experiments.run_fl \
      --config outputs/config_sweeps/cifar10/alpha_${tag}.yaml \
      --method ${method} \
      --rounds 200 \
      --clients 50 \
      --subset-size 0 \
      --output-dir outputs/cifar10_alpha_sweep/alpha_${tag}/fl/${method}
  done
done
```

CIFAR100：

先确认已经生成对应临时配置：

```bash
find outputs/config_sweeps/cifar100 -maxdepth 1 -name "alpha_*.yaml" | sort
```

如果没有配置文件，先运行 11.1 中的 CIFAR100 配置生成命令。

```bash
for tag in 0p05 0p1 0p3 0p5; do
  for method in no_aigc random_incentive binary_aigc fixed_price data_size_proportional quality_gap_proportional proposed_active_set; do
    python -m src.experiments.run_fl \
      --config outputs/config_sweeps/cifar100/alpha_${tag}.yaml \
      --method ${method} \
      --rounds 300 \
      --clients 50 \
      --subset-size 0 \
      --output-dir outputs/cifar100_alpha_sweep/alpha_${tag}/fl/${method}
  done
done
```

#### 11.4 绘制每个 alpha 下的完整 FL 对比图

每个 alpha 生成一张完整方法对比图。该图适合回答：在相同 Non-IID 强度下，Proposed 是否优于 baseline。

FMNIST：

```bash
for tag in 0p01 0p05 0p1 0p3 0p5; do
  python scripts/plot_results.py \
    --fl-csv outputs/fmnist_alpha_sweep/alpha_${tag}/fl/no_aigc/fl_metrics.csv \
    --fl-csv outputs/fmnist_alpha_sweep/alpha_${tag}/fl/random_incentive/fl_metrics.csv \
    --fl-csv outputs/fmnist_alpha_sweep/alpha_${tag}/fl/binary_aigc/fl_metrics.csv \
    --fl-csv outputs/fmnist_alpha_sweep/alpha_${tag}/fl/fixed_price/fl_metrics.csv \
    --fl-csv outputs/fmnist_alpha_sweep/alpha_${tag}/fl/data_size_proportional/fl_metrics.csv \
    --fl-csv outputs/fmnist_alpha_sweep/alpha_${tag}/fl/quality_gap_proportional/fl_metrics.csv \
    --fl-csv outputs/fmnist_alpha_sweep/alpha_${tag}/fl/proposed_active_set/fl_metrics.csv \
    --baseline-csv outputs/fmnist_alpha_sweep/alpha_${tag}/baselines/baseline_clients.csv \
    --baseline-json outputs/fmnist_alpha_sweep/alpha_${tag}/baselines/baseline_summary.json \
    --output-dir outputs/fmnist_alpha_sweep/alpha_${tag}/figures_fl_compare
done
```

CIFAR10：

```bash
for tag in 0p01 0p05 0p5; do
  python scripts/plot_results.py \
    --fl-csv outputs/cifar10_alpha_sweep/alpha_${tag}/fl/no_aigc/fl_metrics.csv \
    --fl-csv outputs/cifar10_alpha_sweep/alpha_${tag}/fl/random_incentive/fl_metrics.csv \
    --fl-csv outputs/cifar10_alpha_sweep/alpha_${tag}/fl/binary_aigc/fl_metrics.csv \
    --fl-csv outputs/cifar10_alpha_sweep/alpha_${tag}/fl/fixed_price/fl_metrics.csv \
    --fl-csv outputs/cifar10_alpha_sweep/alpha_${tag}/fl/data_size_proportional/fl_metrics.csv \
    --fl-csv outputs/cifar10_alpha_sweep/alpha_${tag}/fl/quality_gap_proportional/fl_metrics.csv \
    --fl-csv outputs/cifar10_alpha_sweep/alpha_${tag}/fl/proposed_active_set/fl_metrics.csv \
    --baseline-csv outputs/cifar10_alpha_sweep/alpha_${tag}/baselines/baseline_clients.csv \
    --baseline-json outputs/cifar10_alpha_sweep/alpha_${tag}/baselines/baseline_summary.json \
    --output-dir outputs/cifar10_alpha_sweep/alpha_${tag}/figures_fl_compare
done
```

CIFAR100：

同样需要先确认 `outputs/config_sweeps/cifar100/alpha_${tag}.yaml` 已存在。

```bash
for tag in 0p05 0p01 0p3 0p5; do
  python scripts/plot_results.py \
    --fl-csv outputs/cifar100_alpha_sweep/alpha_${tag}/fl/no_aigc/fl_metrics.csv \
    --fl-csv outputs/cifar100_alpha_sweep/alpha_${tag}/fl/random_incentive/fl_metrics.csv \
    --fl-csv outputs/cifar100_alpha_sweep/alpha_${tag}/fl/binary_aigc/fl_metrics.csv \
    --fl-csv outputs/cifar100_alpha_sweep/alpha_${tag}/fl/fixed_price/fl_metrics.csv \
    --fl-csv outputs/cifar100_alpha_sweep/alpha_${tag}/fl/data_size_proportional/fl_metrics.csv \
    --fl-csv outputs/cifar100_alpha_sweep/alpha_${tag}/fl/quality_gap_proportional/fl_metrics.csv \
    --fl-csv outputs/cifar100_alpha_sweep/alpha_${tag}/fl/proposed_active_set/fl_metrics.csv \
    --baseline-csv outputs/cifar100_alpha_sweep/alpha_${tag}/baselines/baseline_clients.csv \
    --baseline-json outputs/cifar100_alpha_sweep/alpha_${tag}/baselines/baseline_summary.json \
    --output-dir outputs/cifar100_alpha_sweep/alpha_${tag}/figures_fl_compare
done
```

生成的核心图：

```text
outputs/<dataset>_alpha_sweep/alpha_<tag>/figures_fl_compare/accuracy_vs_rounds.pdf
outputs/<dataset>_alpha_sweep/alpha_<tag>/figures_fl_compare/test_loss_vs_rounds.pdf
outputs/<dataset>_alpha_sweep/alpha_<tag>/figures_fl_compare/train_loss_vs_rounds.pdf
outputs/<dataset>_alpha_sweep/alpha_<tag>/figures_fl_compare/lambda_before_after.pdf
outputs/<dataset>_alpha_sweep/alpha_<tag>/figures_fl_compare/server_utility_comparison.pdf
outputs/<dataset>_alpha_sweep/alpha_<tag>/figures_fl_compare/budget_utilization.pdf
```

#### 11.5 绘制不同 alpha 下同一方法的精度曲线

先把每个 alpha 的 CSV 复制成带有清晰 `method` 名称的绘图副本：

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

for dataset in ["fmnist", "cifar10", "cifar100"]:
    for method in ["no_aigc", "random_incentive", "binary_aigc", "fixed_price", "data_size_proportional", "quality_gap_proportional", "proposed_active_set"]:
        for tag, alpha in [("0p05", "0.05"), ("0p1", "0.1"), ("0p3", "0.3"), ("0p5", "0.5")]:
            src = Path(f"outputs/{dataset}_alpha_sweep/alpha_{tag}/fl/{method}/fl_metrics.csv")
            if not src.exists():
                continue
            out_dir = Path(f"outputs/{dataset}_alpha_sweep/plot_inputs/{method}")
            out_dir.mkdir(parents=True, exist_ok=True)
            df = pd.read_csv(src)
            df["method"] = f"{method} alpha={alpha}"
            df.to_csv(out_dir / f"{method}_alpha_{tag}.csv", index=False)
PY
```

绘制 ProposedActiveSet 在不同 alpha 下的 FMNIST 精度曲线：

```bash
python scripts/plot_results.py \
  --fl-csv outputs/fmnist_alpha_sweep/plot_inputs/proposed_active_set/proposed_active_set_alpha_0p05.csv \
  --fl-csv outputs/fmnist_alpha_sweep/plot_inputs/proposed_active_set/proposed_active_set_alpha_0p1.csv \
  --fl-csv outputs/fmnist_alpha_sweep/plot_inputs/proposed_active_set/proposed_active_set_alpha_0p3.csv \
  --fl-csv outputs/fmnist_alpha_sweep/plot_inputs/proposed_active_set/proposed_active_set_alpha_0p5.csv \
  --output-dir outputs/fmnist_alpha_sweep/figures_proposed_by_alpha
```

绘制 ProposedActiveSet 在不同 alpha 下的 CIFAR10 精度曲线：

```bash
python scripts/plot_results.py \
  --fl-csv outputs/cifar10_alpha_sweep/plot_inputs/proposed_active_set/proposed_active_set_alpha_0p05.csv \
  --fl-csv outputs/cifar10_alpha_sweep/plot_inputs/proposed_active_set/proposed_active_set_alpha_0p1.csv \
  --fl-csv outputs/cifar10_alpha_sweep/plot_inputs/proposed_active_set/proposed_active_set_alpha_0p3.csv \
  --fl-csv outputs/cifar10_alpha_sweep/plot_inputs/proposed_active_set/proposed_active_set_alpha_0p5.csv \
  --output-dir outputs/cifar10_alpha_sweep/figures_proposed_by_alpha
```

绘制 ProposedActiveSet 在不同 alpha 下的 CIFAR100 精度曲线：

```bash
python scripts/plot_results.py \
  --fl-csv outputs/cifar100_alpha_sweep/plot_inputs/proposed_active_set/proposed_active_set_alpha_0p05.csv \
  --fl-csv outputs/cifar100_alpha_sweep/plot_inputs/proposed_active_set/proposed_active_set_alpha_0p1.csv \
  --fl-csv outputs/cifar100_alpha_sweep/plot_inputs/proposed_active_set/proposed_active_set_alpha_0p3.csv \
  --fl-csv outputs/cifar100_alpha_sweep/plot_inputs/proposed_active_set/proposed_active_set_alpha_0p5.csv \
  --output-dir outputs/cifar100_alpha_sweep/figures_proposed_by_alpha
```

生成的核心图：

```text
outputs/<dataset>_alpha_sweep/figures_proposed_by_alpha/accuracy_vs_rounds.pdf
outputs/<dataset>_alpha_sweep/figures_proposed_by_alpha/test_loss_vs_rounds.pdf
outputs/<dataset>_alpha_sweep/figures_proposed_by_alpha/train_loss_vs_rounds.pdf
outputs/<dataset>_alpha_sweep/figures_proposed_by_alpha/lambda_before_after.pdf
```

如果想画其他方法随 alpha 变化的曲线，只需要把上面命令中的 `proposed_active_set` 替换为对应方法名，例如 `fixed_price`。

#### 11.6 推荐论文呈现方式

建议至少保留以下三类图：

```text
1. 每个 Dirichlet alpha 下的完整 FL 方法对比图
2. ProposedActiveSet 在不同 Dirichlet alpha 下的精度曲线
3. Lambda before/after under different Dirichlet alpha
4. Server utility / avg_q table under different Dirichlet alpha
```

如果时间有限，优先跑：

```text
CIFAR10 alpha sweep: 0.05, 0.1, 0.3, 0.5
FMNIST alpha sweep: 0.05, 0.1, 0.3, 0.5
CIFAR100 只跑机制层 alpha sweep，端到端 FL 可作为补充
```

### 12. 论文中常用表格字段

机制表格建议从这些文件提取：

```text
outputs/<dataset>/mechanism/mechanism_summary.json
outputs/<dataset>/baselines/baseline_summary.json
```

FL 精度表格建议从这些文件提取：

```text
outputs/<dataset>/fl/<method>/fl_metrics.csv
```

客户端明细和散点图建议从这些文件提取：

```text
outputs/<dataset>/mechanism/mechanism_clients.csv
outputs/<dataset>/baselines/baseline_clients.csv
```

## 数据集说明

支持的数据集：

```text
fmnist
cifar10
cifar100
```

第一次真实运行时，如果配置中 `download: true`，数据会下载到 `dataset.root`，默认是 `./data`。

推荐较完整实验参数：

```text
FMNIST:   rounds=100
CIFAR10:  rounds=200
CIFAR100: rounds=300
N=50
dirichlet_alpha=0.3
budget_ratio=0.4
local_epochs=5
batch_size=32
```

## 概念说明

实验中的 `lambda_k` 是从实际数据划分后计算得到的 Non-IID 程度估计值。真实系统中它应被理解为估计量，而不是客户端自报的真实值。

机制求解阶段和支付验证阶段是分开的：

```text
理论支付 -> 公共验证集贡献分数 -> 最终支付
```

不要直接使用客户端自报的 `q` 或 `lambda` 作为最终支付依据。

## 最小验收命令

在项目目录下运行：

```bash
python -m src.experiments.run_mechanism --config configs/fmnist.yaml
python -m src.experiments.run_fl --config configs/fmnist.yaml --method proposed_active_set
pytest -q
```
