# FL-AIGC Incentive Mechanism Experiments

This project implements an experimental framework for FL-AIGC incentive
mechanisms. It includes mechanism optimization, baseline comparison,
AIGC-proxy data augmentation, FedAvg training, public-validation payment
calibration, and plotting utilities.

The active-set solver is a heuristic. It returns a feasible mode-stable
solution when successful, but it does not claim global optimality. Exact
enumeration is only intended for small client counts.

## Environment

Recommended environment:

```text
Python 3.9.24
torch 2.8.0
torchvision 0.23.0
```

Install dependencies from the project directory:

```bash
pip install -r requirements.txt
```

If you use the provided conda environment:

```bash
conda activate zwy
```

## Project Layout

```text
configs/                  YAML configs
src/data/                 dataset loading, partitioning, lambda, AIGC-proxy
src/mechanism/            incentive mechanism solvers and payments
src/fl/                   FedAvg, models, verification scores
src/experiments/          runnable experiment entry points
scripts/plot_results.py   plotting script
tests/                    pytest suite
```

## Quick Check

Run all tests:

```bash
pytest -q
```

In the `zwy` conda environment:

```bash
conda run -n zwy python -m pytest -q
```

## Configuration

Main config example:

```bash
configs/fmnist.yaml
```

Important fields:

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
  model: smallcnn       # smallcnn or resnet18_cifar
  rounds: 20
  local_epochs: 1
  batch_size: 64
  lr: 0.01
  momentum: 0.0
  client_fraction: 1.0

output:
  dir: ./outputs
```

## Mechanism-Only Experiment

This runs data partitioning, lambda estimation, mechanism parameter generation,
active-set solving, and exact enumeration when `N <= exact_max_clients`. It does
not run FL training.

```bash
python -m src.experiments.run_mechanism --config configs/fmnist.yaml
```

Optional output directory:

```bash
python -m src.experiments.run_mechanism \
  --config configs/fmnist.yaml \
  --output-dir outputs/mechanism
```

Outputs:

```text
outputs/
  mechanism_summary.json
  mechanism_clients.csv
  mechanism_active_set_clients.csv
  mechanism_exact_enum_clients.csv     # only when exact enum runs
```

Key summary metrics:

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

## Baseline Experiment

Run all mechanism baselines with unified CSV/JSON output:

```bash
python -m src.experiments.run_baselines --config configs/fmnist.yaml
```

Baselines:

```text
NoAIGC
RandomIncentive
FixedPrice
DataSizeProportional
ProposedActiveSet
```

Outputs:

```text
outputs/baselines/
  baseline_clients.csv
  baseline_summary.json
```

## End-to-End FL Experiment

This connects mechanism output to AIGC-proxy augmentation and then runs FedAvg.

```bash
python -m src.experiments.run_fl \
  --config configs/fmnist.yaml \
  --method proposed_active_set
```

Smoke run defaults from CLI:

```text
rounds=2
clients=5
subset_size=2000
```

You can override them:

```bash
python -m src.experiments.run_fl \
  --config configs/fmnist.yaml \
  --method proposed_active_set \
  --rounds 5 \
  --clients 10 \
  --subset-size 5000
```

Supported methods:

```text
no_aigc
random_incentive
fixed_price
data_size_proportional
proposed_active_set
```

Supported models:

```text
smallcnn
resnet18_cifar
```

For FMNIST, use `smallcnn`. `resnet18_cifar` expects 3-channel CIFAR inputs.

Outputs:

```text
outputs/
  fl_metrics.csv
  fl_<method>_rounds.csv
```

FL metrics:

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

## Plot Results

Use the plotting script with whichever outputs you have. Missing inputs are
skipped gracefully.

```bash
python scripts/plot_results.py \
  --fl-csv outputs/fl_metrics.csv \
  --baseline-csv outputs/baselines/baseline_clients.csv \
  --baseline-json outputs/baselines/baseline_summary.json \
  --mechanism-csv outputs/mechanism_clients.csv \
  --mechanism-json outputs/mechanism_summary.json \
  --output-dir outputs/figures
```

Generated figures are saved as both PNG and PDF:

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
social_welfare_comparison
```

## Dataset Notes

Supported datasets:

```text
fmnist
cifar10
cifar100
```

The first real run may download data into `dataset.root`, usually `./data`.

Recommended longer-run defaults:

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

## Conceptual Notes

`lambda_k` is computed from the realized data partition in these experiments.
It should be interpreted as an estimated Non-IID degree, not as a client-reported
ground-truth value in a real deployment.

Mechanism solving and payment verification are separate:

```text
theory_reward -> public validation contribution score -> final_reward
```

Do not use client-reported `q` or `lambda` directly as the final payment signal.

## Minimal Acceptance Commands

From the project directory:

```bash
python -m src.experiments.run_mechanism --config configs/fmnist.yaml
python -m src.experiments.run_fl --config configs/fmnist.yaml --method proposed_active_set
pytest -q
```
