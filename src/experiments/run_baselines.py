"""Run mechanism baselines with unified CSV/JSON outputs."""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.datasets import get_dataset
from src.data.partition import dirichlet_partition
from src.experiments.run_mechanism import (
    _corr_lambda_q,
    _dataset_key,
    _diagnostics,
    _extract_labels,
    build_mechanism_params,
    compute_budget,
    budget_ratio_from_config,
    dataset_config,
    dataset_name_from_config,
)
from src.mechanism.active_set import solve_active_set
from src.mechanism.client_response import best_response_q, client_utility, phi
from src.mechanism.fixed_mode_solver import FixedModeResult
from src.mechanism.params import MechanismParams
from src.mechanism.payment import cost_zero, min_payment
from src.utils.seed import set_seed


@dataclass
class BaselineOutput:
    name: str
    result: FixedModeResult
    costs: np.ndarray
    runtime: float
    extra: dict


def _empty_arrays(params: MechanismParams):
    n = len(params.d)
    return (
        np.zeros(n, dtype=np.float64),
        np.zeros(n, dtype=np.float64),
        np.zeros(n, dtype=np.float64),
        np.zeros(n, dtype=np.float64),
        ["exit"] * n,
    )


def _make_result(params: MechanismParams, q, p, x, costs, modes) -> FixedModeResult:
    # Server objective for the budget-constrained mechanism. Payments are
    # reported as total_cost and constrained by B, not subtracted again here.
    server_utility = float(np.sum(x * params.V * phi(params.lambda_k, q, params.lambda_base)))
    return FixedModeResult(
        feasible=True,
        q=q,
        p=p,
        x=x,
        total_cost=float(np.sum(costs)),
        server_utility=server_utility,
        mode=modes,
    )


def _mode_from_q(q, lambda_k, tol=1e-8):
    if q <= tol:
        return "zero"
    if abs(q - lambda_k) <= tol:
        return "upper"
    return "internal"


def _try_min_payment(params: MechanismParams, idx: int, q: float):
    return min_payment(
        q,
        params.d[idx],
        params.lambda_k[idx],
        params.S[idx],
        params.alpha[idx],
        params.beta[idx],
        params.lambda_base,
    )


def baseline_no_aigc(params: MechanismParams, B: float) -> BaselineOutput:
    start = time.perf_counter()
    q, p, x, costs, modes = _empty_arrays(params)
    candidates = []
    for idx in range(len(params.d)):
        cost, price = cost_zero(
            params.d[idx],
            params.lambda_k[idx],
            params.S[idx],
            params.alpha[idx],
            params.lambda_base,
        )
        if np.isfinite(cost):
            value = params.V[idx] * phi(params.lambda_k[idx], 0.0, params.lambda_base)
            candidates.append((value / max(cost, 1e-12), value, idx, cost, price))

    spent = 0.0
    for _, _, idx, cost, price in sorted(candidates, reverse=True):
        if spent + cost <= B + 1e-9:
            p[idx] = price
            x[idx] = 1.0
            costs[idx] = cost
            modes[idx] = "zero"
            spent += cost

    result = _make_result(params, q, p, x, costs, modes)
    return BaselineOutput("NoAIGC", result, costs, time.perf_counter() - start, {})


def baseline_random_incentive(params: MechanismParams, B: float, seed: int = 0) -> BaselineOutput:
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    q, p, x, costs, modes = _empty_arrays(params)
    order = rng.permutation(len(params.d))
    spent = 0.0

    for idx in order:
        q_candidate = float(rng.uniform(0.0, params.lambda_k[idx]))
        cost, price, mode = _try_min_payment(params, idx, q_candidate)
        if not np.isfinite(cost) or spent + cost > B + 1e-9:
            continue
        q[idx] = q_candidate
        p[idx] = price
        x[idx] = 1.0
        costs[idx] = cost
        modes[idx] = mode
        spent += cost

    result = _make_result(params, q, p, x, costs, modes)
    return BaselineOutput("RandomIncentive", result, costs, time.perf_counter() - start, {})


def baseline_binary_aigc(params: MechanismParams, B: float, rho: float = 0.5) -> BaselineOutput:
    """Binary AIGC baseline: each selected client uses q_bar=rho*lambda_k."""
    start = time.perf_counter()
    q, p, x, costs, modes = _empty_arrays(params)
    rho = float(np.clip(rho, 0.0, 1.0))
    candidates = []

    for idx in range(len(params.d)):
        q_candidate = float(rho * params.lambda_k[idx])
        if q_candidate <= 1e-12:
            continue
        cost, price, mode = _try_min_payment(params, idx, q_candidate)
        if not np.isfinite(cost) or cost <= 0:
            continue
        gain = params.V[idx] * (
            phi(params.lambda_k[idx], q_candidate, params.lambda_base)
            - phi(params.lambda_k[idx], 0.0, params.lambda_base)
        )
        candidates.append((gain / cost, gain, idx, q_candidate, cost, price, mode))

    spent = 0.0
    for _, _, idx, q_candidate, cost, price, mode in sorted(candidates, reverse=True):
        if spent + cost > B + 1e-9:
            continue
        q[idx] = q_candidate
        p[idx] = price
        x[idx] = 1.0
        costs[idx] = cost
        modes[idx] = mode
        spent += cost

    result = _make_result(params, q, p, x, costs, modes)
    return BaselineOutput("BinaryAIGC", result, costs, time.perf_counter() - start, {"rho": rho})


def _fixed_price_result(params: MechanismParams, price: float) -> tuple:
    q, p, x, costs, modes = _empty_arrays(params)
    for idx in range(len(params.d)):
        q_candidate = float(
            best_response_q(
                price,
                params.d[idx],
                params.lambda_k[idx],
                params.alpha[idx],
                params.beta[idx],
                params.lambda_base,
            )
        )
        payment = price * params.d[idx] * phi(params.lambda_k[idx], q_candidate, params.lambda_base)
        utility = client_utility(
            q_candidate,
            price,
            params.d[idx],
            params.lambda_k[idx],
            params.S[idx],
            params.alpha[idx],
            params.beta[idx],
            params.lambda_base,
        )
        if utility < -1e-9:
            continue
        q[idx] = q_candidate
        p[idx] = price
        x[idx] = 1.0
        costs[idx] = payment
        modes[idx] = _mode_from_q(q_candidate, params.lambda_k[idx])
    return q, p, x, costs, modes


def baseline_fixed_price(params: MechanismParams, B: float) -> BaselineOutput:
    start = time.perf_counter()

    def total_cost(price):
        return float(np.sum(_fixed_price_result(params, price)[3]))

    high = 1.0
    while total_cost(high) <= B and high < 1e6:
        high *= 2.0
    low = 0.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        if total_cost(mid) <= B:
            low = mid
        else:
            high = mid

    q, p, x, costs, modes = _fixed_price_result(params, low)
    result = _make_result(params, q, p, x, costs, modes)
    return BaselineOutput("FixedPrice", result, costs, time.perf_counter() - start, {"price": low})


def _best_affordable_q(params: MechanismParams, idx: int, budget_share: float):
    best = (0.0, 0.0, 0.0, "exit")
    zero_cost, zero_price = cost_zero(
        params.d[idx],
        params.lambda_k[idx],
        params.S[idx],
        params.alpha[idx],
        params.lambda_base,
    )
    if np.isfinite(zero_cost) and zero_cost <= budget_share + 1e-9:
        best = (0.0, zero_price, zero_cost, "zero")

    upper_cost, upper_price, upper_mode = _try_min_payment(params, idx, params.lambda_k[idx])
    if np.isfinite(upper_cost) and upper_cost <= budget_share + 1e-9:
        return params.lambda_k[idx], upper_price, upper_cost, upper_mode

    low = 0.0
    high = float(params.lambda_k[idx])
    for _ in range(80):
        mid = 0.5 * (low + high)
        cost, price, mode = _try_min_payment(params, idx, mid)
        if np.isfinite(cost) and cost <= budget_share + 1e-9:
            best = (mid, price, cost, mode)
            low = mid
        else:
            high = mid
    return best


def baseline_data_size_proportional(params: MechanismParams, B: float) -> BaselineOutput:
    start = time.perf_counter()
    q, p, x, costs, modes = _empty_arrays(params)
    total_d = float(np.sum(params.d))

    for idx in range(len(params.d)):
        budget_share = B * params.d[idx] / total_d if total_d > 0 else 0.0
        q_i, p_i, cost_i, mode_i = _best_affordable_q(params, idx, budget_share)
        if mode_i == "exit":
            continue
        q[idx] = q_i
        p[idx] = p_i
        x[idx] = 1.0
        costs[idx] = cost_i
        modes[idx] = mode_i

    result = _make_result(params, q, p, x, costs, modes)
    return BaselineOutput("DataSizeProportional", result, costs, time.perf_counter() - start, {})


def baseline_quality_gap_proportional(params: MechanismParams, B: float) -> BaselineOutput:
    """Allocate budget in proportion to each client's quality gap lambda_k."""
    start = time.perf_counter()
    q, p, x, costs, modes = _empty_arrays(params)
    total_gap = float(np.sum(params.lambda_k))

    for idx in range(len(params.d)):
        budget_share = B * params.lambda_k[idx] / total_gap if total_gap > 0 else 0.0
        q_i, p_i, cost_i, mode_i = _best_affordable_q(params, idx, budget_share)
        if mode_i == "exit":
            continue
        q[idx] = q_i
        p[idx] = p_i
        x[idx] = 1.0
        costs[idx] = cost_i
        modes[idx] = mode_i

    result = _make_result(params, q, p, x, costs, modes)
    return BaselineOutput("QualityGapProportional", result, costs, time.perf_counter() - start, {})


def baseline_proposed_active_set(
    params: MechanismParams,
    B: float,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> BaselineOutput:
    """Run the proposed active-set heuristic baseline."""
    start = time.perf_counter()
    active = solve_active_set(params, B, max_iter=max_iter, tol=tol)
    runtime = time.perf_counter() - start
    costs = np.zeros(len(params.d), dtype=np.float64)
    for idx, mode in enumerate(active.result.mode):
        if mode == "exit":
            continue
        cost, _, _ = _try_min_payment(params, idx, active.result.q[idx])
        costs[idx] = cost
    return BaselineOutput(
        "ProposedActiveSet",
        active.result,
        costs,
        runtime,
        {"iterations": active.iterations, "logs": active.logs},
    )


def _client_rows(params: MechanismParams, output: BaselineOutput):
    rows = []
    result = output.result
    for idx in range(len(params.d)):
        utility = (
            client_utility(
                result.q[idx],
                result.p[idx],
                params.d[idx],
                params.lambda_k[idx],
                params.S[idx],
                params.alpha[idx],
                params.beta[idx],
                params.lambda_base,
            )
            if result.mode[idx] != "exit"
            else 0.0
        )
        rows.append(
            {
                "baseline": output.name,
                "client_id": idx,
                "lambda_k": params.lambda_k[idx],
                "q": result.q[idx],
                "p": result.p[idx],
                "x": result.x[idx],
                "mode": result.mode[idx],
                "cost": output.costs[idx],
                "utility": utility,
            }
        )
    return rows


def _summary(params: MechanismParams, output: BaselineOutput, B: float):
    diagnostics = _diagnostics(params, output.result)
    participating = output.result.x > 0
    summary = {
        "baseline": output.name,
        "server_utility": output.result.server_utility,
        "budget_used": output.result.total_cost,
        "budget_utilization": output.result.total_cost / B if B > 0 else 0.0,
        "participation_rate": float(np.mean(participating)),
        "avg_q": float(np.mean(output.result.q[participating])) if np.any(participating) else 0.0,
        "corr_lambda_q": _corr_lambda_q(params.lambda_k, output.result.q),
        "runtime": output.runtime,
    }
    summary.update(diagnostics)
    summary.update(output.extra)
    return summary


def _load_problem(config):
    seed = int(config.get("seed", 0))
    set_seed(seed)
    data_cfg = dataset_config(config)
    dataset_name = _dataset_key(dataset_name_from_config(config))
    dataset = get_dataset(
        dataset_name,
        data_cfg.get("root", "./data"),
        train=True,
        download=bool(data_cfg.get("download", True)),
    )
    labels = _extract_labels(dataset)
    client_indices = dirichlet_partition(
        labels=labels,
        num_clients=int(data_cfg.get("num_clients", 20)),
        alpha=float(data_cfg.get("dirichlet_alpha", 0.5)),
        min_size=int(data_cfg.get("min_size", 10)),
        seed=seed,
    )
    params = build_mechanism_params(labels, client_indices, config)
    budget = compute_budget(params, budget_ratio_from_config(config))
    return params, budget


def run_baselines_experiment(config, output_dir: Path):
    """Run all mechanism baselines and write unified outputs."""
    params, budget = _load_problem(config)
    seed = int(config.get("seed", 0))

    mechanism_cfg = config.get("mechanism", {})
    outputs = [
        baseline_no_aigc(params, budget),
        baseline_random_incentive(params, budget, seed=seed),
        baseline_binary_aigc(params, budget, rho=float(mechanism_cfg.get("binary_rho", 0.5))),
        baseline_fixed_price(params, budget),
        baseline_data_size_proportional(params, budget),
        baseline_quality_gap_proportional(params, budget),
        baseline_proposed_active_set(
            params,
            budget,
            max_iter=int(mechanism_cfg.get("active_set_max_iter", 50)),
            tol=float(mechanism_cfg.get("tol", 1e-8)),
        ),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    client_rows = []
    summaries = {
        "meta": {
            "budget": budget,
            "budget_ratio": budget_ratio_from_config(config),
            "num_clients": len(params.d),
        }
    }
    for output in outputs:
        client_rows.extend(_client_rows(params, output))
        summaries[output.name] = _summary(params, output, budget)

    pd.DataFrame(client_rows).to_csv(output_dir / "baseline_clients.csv", index=False)
    with open(output_dir / "baseline_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    return summaries


def load_config(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main():
    parser = argparse.ArgumentParser(description="Run mechanism baselines")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--budget-ratio", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.budget_ratio is not None:
        config.setdefault("mechanism", {})["budget_ratio"] = args.budget_ratio
    output_dir = args.output_dir or Path(config.get("output", {}).get("dir", "./outputs")) / "baselines"
    summaries = run_baselines_experiment(config, output_dir)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
