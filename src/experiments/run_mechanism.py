"""Run mechanism-only experiments without FL training."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.datasets import get_dataset
from src.data.partition import dirichlet_partition
from src.data.quality import compute_lambdas
from src.mechanism.active_set import solve_active_set
from src.mechanism.client_response import best_response_q, client_utility
from src.mechanism.exact_enum import solve_exact_enum
from src.mechanism.params import MechanismParams
from src.mechanism.payment import cost_upper, min_payment
from src.utils.seed import set_seed


def dataset_config(config):
    """Return dataset config, accepting both dataset.* and legacy data.*."""
    return config.get("dataset", config.get("data", {}))


def dataset_name_from_config(config) -> str:
    """Return normalized dataset name from config."""
    data_cfg = dataset_config(config)
    return data_cfg.get("name", data_cfg.get("dataset", "fmnist"))


def service_cost_range_from_config(mechanism_cfg):
    """Return S_k sampling range, accepting S_range and legacy service_cost_range."""
    return mechanism_cfg.get("S_range", mechanism_cfg.get("service_cost_range", [0.01, 0.1]))


def budget_ratio_from_config(config) -> float:
    """Return budget ratio, accepting mechanism.budget_ratio and legacy budget.ratio."""
    mechanism_cfg = config.get("mechanism", {})
    if "budget_ratio" in mechanism_cfg:
        return float(mechanism_cfg["budget_ratio"])
    return float(config.get("budget", {}).get("ratio", 0.3))


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


def _sample_uniform(rng, value_range, size):
    low, high = value_range
    return rng.uniform(float(low), float(high), size=size)


def build_mechanism_params(labels, client_indices, config):
    """Build MechanismParams from partition statistics and config ranges."""
    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)
    data_cfg = dataset_config(config)
    mechanism_cfg = config.get("mechanism", {})

    num_classes = int(data_cfg.get("num_classes", _num_classes(dataset_name_from_config(config))))
    # lambda_values are estimated from the realized partition, not client-reported truth.
    lambda_values = np.asarray(compute_lambdas(labels, client_indices, num_classes), dtype=np.float64)
    d = np.asarray([len(indices) for indices in client_indices], dtype=np.float64)
    d = d / max(float(np.sum(d)), 1.0)

    value_scale = float(mechanism_cfg.get("value_scale", 1.0))
    params = MechanismParams(
        d=d,
        lambda_k=lambda_values,
        S=_sample_uniform(rng, service_cost_range_from_config(mechanism_cfg), len(d)),
        alpha=_sample_uniform(rng, mechanism_cfg.get("alpha_range", [0.01, 0.1]), len(d)),
        beta=_sample_uniform(rng, mechanism_cfg.get("beta_range", [0.01, 0.1]), len(d)),
        V=value_scale * d.copy(),
        lambda_base=float(mechanism_cfg.get("lambda_base", 1.0)),
    )
    params.validate()
    return params


def compute_budget(params: MechanismParams, budget_ratio: float) -> float:
    """Compute budget from ratio times finite upper-bound costs."""
    costs = []
    for idx in range(len(params.d)):
        cost, _ = cost_upper(
            params.d[idx],
            params.lambda_k[idx],
            params.S[idx],
            params.alpha[idx],
            params.beta[idx],
            params.lambda_base,
        )
        if np.isfinite(cost):
            costs.append(cost)
    return float(budget_ratio * np.sum(costs))


def _client_rows(params: MechanismParams, result):
    rows = []
    for idx in range(len(params.d)):
        cost, _, _ = min_payment(
            result.q[idx],
            params.d[idx],
            params.lambda_k[idx],
            params.S[idx],
            params.alpha[idx],
            params.beta[idx],
            params.lambda_base,
        ) if result.mode[idx] != "exit" else (0.0, 0.0, "exit")
        utility = client_utility(
            result.q[idx],
            result.p[idx],
            params.d[idx],
            params.lambda_k[idx],
            params.S[idx],
            params.alpha[idx],
            params.beta[idx],
            params.lambda_base,
        ) if result.mode[idx] != "exit" else 0.0
        rows.append(
            {
                "client_id": idx,
                "lambda_k": params.lambda_k[idx],
                "q": result.q[idx],
                "p": result.p[idx],
                "x": result.x[idx],
                "mode": result.mode[idx],
                "cost": cost,
                "utility": utility,
            }
        )
    return rows


def _diagnostics(params: MechanismParams, result, tol: float = 1e-7):
    regrets = []
    ir_violations = 0
    for idx, mode in enumerate(result.mode):
        if mode == "exit":
            continue
        utility = client_utility(
            result.q[idx],
            result.p[idx],
            params.d[idx],
            params.lambda_k[idx],
            params.S[idx],
            params.alpha[idx],
            params.beta[idx],
            params.lambda_base,
        )
        if utility < -tol:
            ir_violations += 1

        br_q = best_response_q(
            result.p[idx],
            params.d[idx],
            params.lambda_k[idx],
            params.alpha[idx],
            params.beta[idx],
            params.lambda_base,
        )
        br_utility = client_utility(
            br_q,
            result.p[idx],
            params.d[idx],
            params.lambda_k[idx],
            params.S[idx],
            params.alpha[idx],
            params.beta[idx],
            params.lambda_base,
        )
        regrets.append(max(float(br_utility - utility), 0.0))

    return {
        "ir_violations": ir_violations,
        "ic_regret_max": max(regrets) if regrets else 0.0,
        "ic_regret_mean": float(np.mean(regrets)) if regrets else 0.0,
    }


def _corr_lambda_q(lambda_values, q_values):
    if len(lambda_values) < 2:
        return 0.0
    if np.std(lambda_values) == 0 or np.std(q_values) == 0:
        return 0.0
    return float(np.corrcoef(lambda_values, q_values)[0, 1])


def summarize(params: MechanismParams, result, budget: float, runtime: float, solver_name: str):
    diagnostics = _diagnostics(params, result)
    participating = result.x > 0
    summary = {
        "solver": solver_name,
        "server_utility": result.server_utility,
        "budget_used": result.total_cost,
        "budget_utilization": result.total_cost / budget if budget > 0 else 0.0,
        "participation_rate": float(np.mean(participating)),
        "avg_q": float(np.mean(result.q[participating])) if np.any(participating) else 0.0,
        "corr_lambda_q": _corr_lambda_q(params.lambda_k, result.q),
        "runtime": runtime,
    }
    summary.update(diagnostics)
    return summary


def run_mechanism_experiment(config, output_dir: Path):
    """Run exact-enum when available and active-set for all N."""
    start = time.perf_counter()
    seed = int(config.get("seed", 0))
    set_seed(seed)

    data_cfg = dataset_config(config)
    dataset_name = _dataset_key(dataset_name_from_config(config))
    data_root = data_cfg.get("root", "./data")
    num_clients = int(data_cfg.get("num_clients", 20))
    min_size = int(data_cfg.get("min_size", 10))
    dirichlet_alpha = float(data_cfg.get("dirichlet_alpha", 0.5))

    dataset = get_dataset(dataset_name, data_root, train=True, download=bool(data_cfg.get("download", True)))
    labels = _extract_labels(dataset)
    client_indices = dirichlet_partition(
        labels=labels,
        num_clients=num_clients,
        alpha=dirichlet_alpha,
        min_size=min_size,
        seed=seed,
    )
    params = build_mechanism_params(labels, client_indices, config)
    budget = compute_budget(params, budget_ratio_from_config(config))

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}

    active_start = time.perf_counter()
    mechanism_cfg = config.get("mechanism", {})
    active = solve_active_set(
        params,
        budget,
        max_iter=int(mechanism_cfg.get("active_set_max_iter", 50)),
        tol=float(mechanism_cfg.get("tol", 1e-8)),
    )
    active_runtime = time.perf_counter() - active_start
    if not active.feasible:
        raise RuntimeError("active-set failed to produce a feasible solution")

    active_df = pd.DataFrame(_client_rows(params, active.result))
    active_csv = output_dir / "mechanism_active_set_clients.csv"
    active_df.to_csv(active_csv, index=False)
    active_df.to_csv(output_dir / "mechanism_clients.csv", index=False)
    summaries["active_set"] = summarize(params, active.result, budget, active_runtime, "active_set")
    summaries["active_set"]["iterations"] = active.iterations
    summaries["active_set"]["logs"] = active.logs

    max_exact = int(mechanism_cfg.get("exact_max_clients", mechanism_cfg.get("max_clients_for_exact", 12)))
    if len(params.d) <= max_exact:
        exact_start = time.perf_counter()
        exact = solve_exact_enum(params, budget, max_clients_for_exact=max_exact)
        exact_runtime = time.perf_counter() - exact_start
        if exact.feasible:
            exact_df = pd.DataFrame(_client_rows(params, exact.best_result))
            exact_csv = output_dir / "mechanism_exact_enum_clients.csv"
            exact_df.to_csv(exact_csv, index=False)
            summaries["exact_enum"] = summarize(params, exact.best_result, budget, exact_runtime, "exact_enum")
            summaries["exact_enum"]["searched_modes"] = exact.searched_modes
            summaries["exact_enum"]["feasible_modes"] = exact.feasible_modes

    summaries["meta"] = {
        "dataset": dataset_name,
        "num_clients": num_clients,
        "budget": budget,
        "budget_ratio": budget_ratio_from_config(config),
        "total_runtime": time.perf_counter() - start,
    }

    summary_path = output_dir / "mechanism_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)

    return summaries


def load_config(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main():
    parser = argparse.ArgumentParser(description="Run mechanism-only experiment")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir or Path(config.get("output", {}).get("dir", "./outputs"))
    summaries = run_mechanism_experiment(config, output_dir)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
