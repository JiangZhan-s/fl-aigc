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
from src.experiments.run_baselines import (
    baseline_data_size_proportional,
    baseline_fixed_price,
    baseline_no_aigc,
    baseline_proposed_active_set,
    baseline_random_incentive,
)
from src.experiments.run_mechanism import build_mechanism_params, compute_budget
from src.experiments.run_mechanism import budget_ratio_from_config, dataset_config, dataset_name_from_config
from src.fl.models import SmallCNNCIFAR, build_model, resnet18_cifar
from src.fl.server import run_fedavg
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


def _run_method(params, budget, method: str, seed: int, config=None):
    """Run one mechanism or baseline method and return its output."""
    normalized = method.lower()
    mechanism_cfg = (config or {}).get("mechanism", {})
    if normalized in {"no_aigc", "noaigc"}:
        return baseline_no_aigc(params, budget)
    if normalized in {"random", "random_incentive", "randomincentive"}:
        return baseline_random_incentive(params, budget, seed=seed)
    if normalized in {"fixed_price", "fixedprice"}:
        return baseline_fixed_price(params, budget)
    if normalized in {"data_size", "data_size_proportional", "datasizeproportional"}:
        return baseline_data_size_proportional(params, budget)
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


def run_end_to_end(config, method: str = "proposed_active_set", output_dir: Path = None):
    """Run mechanism/baseline selection, AIGC-proxy augmentation, and FedAvg."""
    seed = int(config.get("seed", 0))
    set_seed(seed)

    data_cfg = dataset_config(config)
    fl_cfg = config.get("fl", {})
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

    global_dist = global_label_distribution(labels, num_classes)
    augmentation = aigc_proxy_augment(
        dataset=train_dataset,
        client_indices=client_indices,
        labels=labels,
        q_values=mechanism_output.result.q,
        lambda_values=lambda_before,
        num_classes=num_classes,
        global_dist=global_dist,
        seed=seed,
    )

    model = _build_model(dataset_name, fl_cfg.get("model", "smallcnn"), num_classes)
    history = run_fedavg(
        model=model,
        train_dataset=train_dataset,
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
        seed=seed,
    )

    avg_lambda_before = float(np.mean(lambda_before))
    avg_lambda_after = float(np.mean(augmentation.augmented_lambdas))
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
    parser.add_argument("--model", default=None, choices=["smallcnn", "resnet18_cifar", "smallcnn_cifar"])
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("fl", {})["rounds"] = args.rounds
    config.setdefault("dataset", config.get("data", {}))["num_clients"] = args.clients
    config.setdefault("dataset", config.get("data", {}))["subset_size"] = args.subset_size
    if args.model is not None:
        config.setdefault("fl", {})["model"] = args.model

    output_dir = args.output_dir or Path(config.get("output", {}).get("dir", "./outputs"))
    df = run_end_to_end(config, method=args.method, output_dir=output_dir)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
