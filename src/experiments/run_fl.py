"""Run end-to-end FL experiments with mechanism-driven AIGC-proxy indices."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from torch.utils.data import Subset

from src.data.aigc_proxy import aigc_proxy_augment
from src.data.datasets import get_dataset
from src.data.partition import dirichlet_partition
from src.data.quality import compute_lambdas, global_label_distribution
from src.data.real_aigc import build_real_aigc_augmented_dataset
from src.experiments.run_baselines import (
    baseline_binary_aigc,
    baseline_data_size_proportional,
    baseline_fixed_price,
    baseline_no_aigc,
    baseline_proposed_active_set,
    baseline_quality_gap_proportional,
    baseline_random_incentive,
)
from src.experiments.run_mechanism import build_mechanism_params, compute_budget
from src.experiments.run_mechanism import budget_ratio_from_config, dataset_config, dataset_name_from_config
from src.fl.models import SmallCNNCIFAR, build_model, resnet18_cifar
from src.fl.server import run_fedavg
from src.mechanism.client_response import best_response_q, client_utility
from src.utils.device import get_device
from src.utils.seed import set_seed


def _dataset_key(name: str) -> str:
    normalized = name.lower()
    if normalized in {"fashionmnist", "fashion_mnist", "fmnist"}:
        return "fmnist"
    if normalized == "cifar10":
        return "cifar10"
    if normalized == "cifar100":
        return "cifar100"
    raise ValueError(f"Unsupported dataset: {name}")


def _num_classes(dataset_name: str) -> int:
    return 100 if _dataset_key(dataset_name) == "cifar100" else 10


def _extract_labels(dataset):
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels, dtype=np.int64)
    raise ValueError("Dataset does not expose targets or labels")


def _maybe_subset(dataset, labels, subset_size):
    if subset_size is None or subset_size <= 0 or subset_size >= len(dataset):
        return dataset, labels
    indices = list(range(int(subset_size)))
    return Subset(dataset, indices), np.asarray(labels, dtype=np.int64)[indices]


def _build_model(dataset_name: str, model_name: str, num_classes: int):
    normalized = model_name.lower()
    if normalized == "smallcnn":
        return build_model(dataset_name, num_classes)
    if normalized == "resnet18_cifar":
        if _dataset_key(dataset_name) == "fmnist":
            raise ValueError("resnet18_cifar expects 3-channel CIFAR inputs")
        return resnet18_cifar(num_classes=num_classes)
    if normalized == "smallcnn_cifar":
        return SmallCNNCIFAR(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")


def _corr_lambda_q(lambda_values, q_values):
    """Return corr(lambda, q), or nan when either side is constant."""
    lambda_array = np.asarray(lambda_values, dtype=np.float64)
    q_array = np.asarray(q_values, dtype=np.float64)
    if len(lambda_array) < 2 or np.std(lambda_array) == 0 or np.std(q_array) == 0:
        return float("nan")
    return float(np.corrcoef(lambda_array, q_array)[0, 1])


def _mode_counts(modes):
    """Return a compact mode-count dictionary."""
    return {mode: int(modes.count(mode)) for mode in sorted(set(modes))}


def validate_mechanism_result(params, result, B, tol: float = 1e-6):
    """Print mechanism sanity warnings without interrupting FL."""
    q = np.asarray(result.q, dtype=np.float64)
    p = np.asarray(result.p, dtype=np.float64)
    lambda_k = np.asarray(params.lambda_k, dtype=np.float64)
    warnings = []
    max_response_error = 0.0
    max_ic_regret = 0.0

    if np.any(q < -tol) or np.any(q > lambda_k + tol):
        warnings.append("q is outside [0, lambda_k]")
    if np.any(p < -tol):
        warnings.append("p contains negative values")
    if result.total_cost > B + tol:
        warnings.append(f"total_cost exceeds budget: {result.total_cost:.6f} > {B:.6f}")

    for idx, mode in enumerate(result.mode):
        if mode == "exit":
            continue

        utility = client_utility(
            q[idx],
            p[idx],
            params.d[idx],
            params.lambda_k[idx],
            params.S[idx],
            params.alpha[idx],
            params.beta[idx],
            params.lambda_base,
        )
        if utility < -tol:
            warnings.append(f"client {idx} violates IR: utility={utility:.6e}")

        br_q = best_response_q(
            p[idx],
            params.d[idx],
            params.lambda_k[idx],
            params.alpha[idx],
            params.beta[idx],
            params.lambda_base,
        )
        response_error = abs(float(br_q) - q[idx])
        max_response_error = max(max_response_error, response_error)
        br_utility = client_utility(
            br_q,
            p[idx],
            params.d[idx],
            params.lambda_k[idx],
            params.S[idx],
            params.alpha[idx],
            params.beta[idx],
            params.lambda_base,
        )
        max_ic_regret = max(max_ic_regret, max(float(br_utility - utility), 0.0))

    if max_response_error > tol:
        warnings.append(f"max_response_error={max_response_error:.6e}")
    if max_ic_regret > tol:
        warnings.append(f"max_ic_regret={max_ic_regret:.6e}")

    if warnings:
        print("[mechanism warning] " + " | ".join(warnings))

    return {
        "max_response_error": max_response_error,
        "max_ic_regret": max_ic_regret,
        "warnings": warnings,
    }


def _print_mechanism_debug(method, B, result, lambda_before, avg_lambda_before):
    """Print pre-augmentation mechanism diagnostics."""
    q = np.asarray(result.q, dtype=np.float64)
    lambda_array = np.asarray(lambda_before, dtype=np.float64)
    participation_rate = float(np.mean(result.x > 0))
    print("[mechanism debug] before AIGC-proxy")
    print(f"  method: {method}")
    print(f"  B: {B:.6f}")
    print(f"  budget_used/total_cost: {result.total_cost:.6f}")
    print(f"  participation_rate: {participation_rate:.6f}")
    print(f"  q_min/q_mean/q_max: {q.min():.6f} / {q.mean():.6f} / {q.max():.6f}")
    print(
        "  lambda_min/lambda_mean/lambda_max: "
        f"{lambda_array.min():.6f} / {lambda_array.mean():.6f} / {lambda_array.max():.6f}"
    )
    print(f"  corr(lambda_k, q): {_corr_lambda_q(lambda_array, q)}")
    print(f"  mode_counts: {_mode_counts(result.mode)}")
    print(f"  avg_lambda_before: {avg_lambda_before:.6f}")


def _print_augmentation_debug(augmentation, avg_lambda_before, avg_lambda_after):
    """Print post-augmentation mechanism diagnostics."""
    sizes = np.asarray([len(indices) for indices in augmentation.client_indices_aug], dtype=np.float64)
    print("[mechanism debug] after AIGC-proxy")
    print(f"  avg_lambda_after: {avg_lambda_after:.6f}")
    print(f"  lambda_reduction: {avg_lambda_before - avg_lambda_after:.6f}")
    print(f"  total_augmented_samples: {int(np.sum(augmentation.added_counts))}")
    print(
        "  augmented_size_min/mean/max: "
        f"{sizes.min():.0f} / {sizes.mean():.2f} / {sizes.max():.0f}"
    )


def _run_method(params, budget, method: str, seed: int, config=None):
    """Run one mechanism or baseline method and return its output."""
    normalized = method.lower()
    mechanism_cfg = (config or {}).get("mechanism", {})
    if normalized in {"no_aigc", "noaigc"}:
        return baseline_no_aigc(params, budget)
    if normalized in {"random", "random_incentive", "randomincentive"}:
        return baseline_random_incentive(params, budget, seed=seed)
    if normalized in {"binary", "binary_aigc", "binaryaigc"}:
        return baseline_binary_aigc(
            params,
            budget,
            rho=float(mechanism_cfg.get("binary_rho", 0.5)),
        )
    if normalized in {"fixed_price", "fixedprice"}:
        return baseline_fixed_price(params, budget)
    if normalized in {"data_size", "data_size_proportional", "datasizeproportional"}:
        return baseline_data_size_proportional(params, budget)
    if normalized in {"quality_gap", "quality_gap_proportional", "qualitygapproportional"}:
        return baseline_quality_gap_proportional(params, budget)
    if normalized in {"proposed", "active_set", "proposed_active_set", "proposedactiveset"}:
        return baseline_proposed_active_set(
            params,
            budget,
            max_iter=int(mechanism_cfg.get("active_set_max_iter", 50)),
            tol=float(mechanism_cfg.get("tol", 1e-8)),
        )
    raise ValueError(f"Unsupported method: {method}")


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_end_to_end(
    config,
    method: str = "proposed_active_set",
    output_dir: Path = None,
    debug_mechanism: bool = False,
):
    """Run mechanism/baseline selection, AIGC-proxy augmentation, and FedAvg."""
    seed = int(config.get("seed", 0))
    set_seed(seed)

    data_cfg = dataset_config(config)
    fl_cfg = config.get("fl", {})
    if bool(fl_cfg.get("cuda_benchmark", False)) and get_device().type == "cuda":
        import torch

        torch.backends.cudnn.benchmark = True
    dataset_name = _dataset_key(dataset_name_from_config(config))
    num_classes = int(data_cfg.get("num_classes", _num_classes(dataset_name)))

    download = bool(data_cfg.get("download", True))
    train_dataset_full = get_dataset(dataset_name, data_cfg.get("root", "./data"), train=True, download=download)
    test_dataset = get_dataset(dataset_name, data_cfg.get("root", "./data"), train=False, download=download)
    labels_full = _extract_labels(train_dataset_full)
    train_dataset, labels = _maybe_subset(
        train_dataset_full,
        labels_full,
        data_cfg.get("subset_size"),
    )

    client_indices = dirichlet_partition(
        labels=labels,
        num_clients=int(data_cfg.get("num_clients", 20)),
        alpha=float(data_cfg.get("dirichlet_alpha", 0.5)),
        min_size=int(data_cfg.get("min_size", 10)),
        seed=seed,
    )
    lambda_before = np.asarray(compute_lambdas(labels, client_indices, num_classes), dtype=np.float64)
    params = build_mechanism_params(labels, client_indices, config)
    budget = compute_budget(params, budget_ratio_from_config(config))
    mechanism_output = _run_method(params, budget, method, seed, config)
    validate_mechanism_result(params, mechanism_output.result, budget)

    global_dist = global_label_distribution(labels, num_classes)
    avg_lambda_before = float(np.mean(lambda_before))
    if debug_mechanism:
        _print_mechanism_debug(method, budget, mechanism_output.result, lambda_before, avg_lambda_before)

    aigc_cfg = config.get("aigc_proxy", {})
    use_real_aigc = bool(aigc_cfg.get("use_real", False)) or aigc_cfg.get("backend", "proxy") == "real"
    if use_real_aigc:
        augmentation = build_real_aigc_augmented_dataset(
            dataset=train_dataset,
            client_indices=client_indices,
            labels=labels,
            q_values=mechanism_output.result.q,
            lambda_values=lambda_before,
            num_classes=num_classes,
            global_dist=global_dist,
            aigc_root=aigc_cfg.get("real_root", "./aigc_imgs"),
            dataset_name=dataset_name,
            seed=seed,
            max_extra_ratio=float(aigc_cfg.get("max_extra_ratio", 1.0)),
            cache_path=aigc_cfg.get("cache_path"),
        )
        train_dataset_for_fl = augmentation.dataset
    else:
        augmentation = aigc_proxy_augment(
            dataset=train_dataset,
            client_indices=client_indices,
            labels=labels,
            q_values=mechanism_output.result.q,
            lambda_values=lambda_before,
            num_classes=num_classes,
            global_dist=global_dist,
            seed=seed,
            max_extra_ratio=float(aigc_cfg.get("max_extra_ratio", 1.0)),
        )
        train_dataset_for_fl = train_dataset
    avg_lambda_after = float(np.mean(augmentation.augmented_lambdas))

    if debug_mechanism:
        _print_augmentation_debug(augmentation, avg_lambda_before, avg_lambda_after)

    model = _build_model(dataset_name, fl_cfg.get("model", "smallcnn"), num_classes)
    history = run_fedavg(
        model=model,
        train_dataset=train_dataset_for_fl,
        test_dataset=test_dataset,
        client_indices=augmentation.client_indices_aug,
        device=get_device(),
        rounds=int(fl_cfg.get("rounds", 2)),
        batch_size=int(fl_cfg.get("batch_size", 64)),
        local_epochs=int(fl_cfg.get("local_epochs", 1)),
        learning_rate=float(fl_cfg.get("lr", fl_cfg.get("learning_rate", 0.01))),
        momentum=float(fl_cfg.get("momentum", 0.0)),
        weight_decay=float(fl_cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(fl_cfg.get("max_grad_norm", 10.0)),
        client_fraction=float(fl_cfg.get("client_fraction", 1.0)),
        optimizer_name=fl_cfg.get("optimizer", "sgd"),
        num_workers=int(fl_cfg.get("num_workers", 0)),
        pin_memory=bool(fl_cfg.get("pin_memory", False)),
        persistent_workers=bool(fl_cfg.get("persistent_workers", False)),
        prefetch_factor=(
            int(fl_cfg["prefetch_factor"])
            if fl_cfg.get("prefetch_factor") is not None
            else None
        ),
        use_amp=bool(fl_cfg.get("amp", False)),
        seed=seed,
    )

    participation_rate = float(np.mean(mechanism_output.result.x > 0))
    rows = [
        {
            "round": metrics.round,
            "test_accuracy": metrics.test_accuracy,
            "test_loss": metrics.test_loss,
            "train_loss": metrics.train_loss,
            "avg_lambda_before": avg_lambda_before,
            "avg_lambda_after": avg_lambda_after,
            "budget_used": mechanism_output.result.total_cost,
            "participation_rate": participation_rate,
            "method": method,
        }
        for metrics in history
    ]
    df = pd.DataFrame(rows)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / f"fl_{method}_rounds.csv", index=False)
        df.to_csv(output_dir / "fl_metrics.csv", index=False)

    return df


def run_from_config(config):
    """Backward-compatible wrapper for the default No-AIGC run."""
    return run_end_to_end(config, method="no_aigc")[
        ["round", "train_loss", "test_loss", "test_accuracy"]
    ]


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end FL experiment")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--method", default="proposed_active_set")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--subset-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--client-fraction", type=float, default=None)
    parser.add_argument("--model", default=None, choices=["smallcnn", "resnet18_cifar", "smallcnn_cifar"])
    parser.add_argument("--debug-mechanism", action="store_true")
    parser.add_argument("--real-aigc", action="store_true")
    parser.add_argument("--aigc-root", default=None)
    parser.add_argument("--aigc-cache", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cuda-benchmark", action="store_true")
    parser.add_argument(
        "--fast-gpu",
        action="store_true",
        help="Enable CUDA-friendly defaults: AMP, pinned memory, persistent workers, and cuDNN benchmark.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("fl", {})["rounds"] = args.rounds
    config.setdefault("dataset", config.get("data", {}))["num_clients"] = args.clients
    config.setdefault("dataset", config.get("data", {}))["subset_size"] = args.subset_size
    if args.model is not None:
        config.setdefault("fl", {})["model"] = args.model
    if args.real_aigc:
        config.setdefault("aigc_proxy", {})["use_real"] = True
        config.setdefault("aigc_proxy", {})["backend"] = "real"
    if args.aigc_root is not None:
        config.setdefault("aigc_proxy", {})["real_root"] = args.aigc_root
    if args.aigc_cache is not None:
        config.setdefault("aigc_proxy", {})["cache_path"] = args.aigc_cache
    fl_cfg = config.setdefault("fl", {})
    if args.batch_size is not None:
        fl_cfg["batch_size"] = args.batch_size
    if args.client_fraction is not None:
        fl_cfg["client_fraction"] = args.client_fraction
    if args.fast_gpu:
        fl_cfg["amp"] = True
        fl_cfg["pin_memory"] = True
        fl_cfg["persistent_workers"] = True
        fl_cfg["cuda_benchmark"] = True
        fl_cfg.setdefault("num_workers", 8)
        fl_cfg.setdefault("prefetch_factor", 4)
    if args.num_workers is not None:
        fl_cfg["num_workers"] = args.num_workers
    if args.pin_memory:
        fl_cfg["pin_memory"] = True
    if args.persistent_workers:
        fl_cfg["persistent_workers"] = True
    if args.prefetch_factor is not None:
        fl_cfg["prefetch_factor"] = args.prefetch_factor
    if args.amp:
        fl_cfg["amp"] = True
    if args.cuda_benchmark:
        fl_cfg["cuda_benchmark"] = True

    output_dir = args.output_dir or Path(config.get("output", {}).get("dir", "./outputs"))
    df = run_end_to_end(
        config,
        method=args.method,
        output_dir=output_dir,
        debug_mechanism=args.debug_mechanism,
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
