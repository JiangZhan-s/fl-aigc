"""Scalable active-set heuristic for the incentive mechanism.

This active-set routine is a heuristic. It returns a feasible mode-stable
solution when successful. The final fixed-mode continuous subproblem is solved
by the KKT-based fixed-mode solver.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from src.mechanism.client_response import best_response_q, client_utility
from src.mechanism.fixed_mode_solver import FixedModeResult, solve_fixed_mode
from src.mechanism.params import MechanismParams
from src.mechanism.payment import cost_upper, cost_zero


@dataclass
class ActiveSetResult:
    """Active-set result with heuristic diagnostics."""

    result: FixedModeResult
    iterations: int
    ir_violations: int
    max_ic_regret: float
    logs: List[str]

    @property
    def feasible(self) -> bool:
        return self.result.feasible


def _partition_from_modes(modes):
    S0 = [idx for idx, mode in enumerate(modes) if mode == "zero"]
    SI = [idx for idx, mode in enumerate(modes) if mode == "internal"]
    SU = [idx for idx, mode in enumerate(modes) if mode == "upper"]
    Sout = [idx for idx, mode in enumerate(modes) if mode == "exit"]
    return S0, SI, SU, Sout


def _initial_modes(params: MechanismParams, B: float, tol: float):
    d = np.asarray(params.d, dtype=np.float64)
    lambda_k = np.asarray(params.lambda_k, dtype=np.float64)
    S = np.asarray(params.S, dtype=np.float64)
    alpha = np.asarray(params.alpha, dtype=np.float64)
    beta = np.asarray(params.beta, dtype=np.float64)
    V = np.asarray(params.V, dtype=np.float64)
    lambda_base = params.lambda_base
    num_clients = len(d)

    modes = ["exit"] * num_clients
    internal_candidates = []
    upper_candidates = []

    for idx in range(num_clients):
        # Prefer enhancement-capable clients for the initial active set. q=0 is
        # cheap under normalized d, so greedily filling S0 can make the heuristic
        # collapse into the NoAIGC baseline before the KKT subproblem ever sees
        # an internal client.
        internal_score = V[idx] * max(lambda_k[idx], tol) / max(alpha[idx] + beta[idx], tol)
        internal_candidates.append((internal_score, idx))

        upper_cost, _ = cost_upper(
            d[idx],
            lambda_k[idx],
            S[idx],
            alpha[idx],
            beta[idx],
            lambda_base,
        )
        upper_gain = V[idx] * lambda_base
        upper_candidates.append((upper_gain / max(upper_cost, tol), upper_gain, idx, upper_cost))

    for _, idx in sorted(internal_candidates, reverse=True):
        if lambda_k[idx] > tol:
            modes[idx] = "internal"

    upper_budget = 0.1 * B
    upper_spent = 0.0
    for _, _, idx, cost in sorted(upper_candidates, reverse=True):
        if modes[idx] == "internal" and upper_spent + cost <= upper_budget + tol:
            modes[idx] = "upper"
            upper_spent += cost

    for idx in range(num_clients):
        if modes[idx] == "exit":
            zero_cost, _ = cost_zero(
                d[idx],
                lambda_k[idx],
                S[idx],
                alpha[idx],
                lambda_base,
                tol,
            )
            if np.isfinite(zero_cost):
                modes[idx] = "zero"

    return modes


def _remove_lowest_score_client(modes, params: MechanismParams):
    V = np.asarray(params.V, dtype=np.float64)
    lambda_k = np.asarray(params.lambda_k, dtype=np.float64)
    alpha = np.asarray(params.alpha, dtype=np.float64)
    beta = np.asarray(params.beta, dtype=np.float64)

    participating = [idx for idx, mode in enumerate(modes) if mode != "exit"]
    if not participating:
        return False

    def score(idx):
        return V[idx] * (1.0 + lambda_k[idx]) / max(alpha[idx] + beta[idx], 1e-12)

    victim = min(participating, key=score)
    modes[victim] = "exit"
    return True


def _diagnostics(params: MechanismParams, result: FixedModeResult, tol: float):
    d = np.asarray(params.d, dtype=np.float64)
    lambda_k = np.asarray(params.lambda_k, dtype=np.float64)
    S = np.asarray(params.S, dtype=np.float64)
    alpha = np.asarray(params.alpha, dtype=np.float64)
    beta = np.asarray(params.beta, dtype=np.float64)
    lambda_base = params.lambda_base

    ir_violations = 0
    max_ic_regret = 0.0
    for idx, mode in enumerate(result.mode):
        if mode == "exit":
            continue

        utility = client_utility(
            result.q[idx],
            result.p[idx],
            d[idx],
            lambda_k[idx],
            S[idx],
            alpha[idx],
            beta[idx],
            lambda_base,
        )
        if utility < -tol:
            ir_violations += 1

        br_q = best_response_q(
            result.p[idx],
            d[idx],
            lambda_k[idx],
            alpha[idx],
            beta[idx],
            lambda_base,
        )
        br_utility = client_utility(
            br_q,
            result.p[idx],
            d[idx],
            lambda_k[idx],
            S[idx],
            alpha[idx],
            beta[idx],
            lambda_base,
        )
        max_ic_regret = max(max_ic_regret, float(br_utility - utility))

    return ir_violations, max(max_ic_regret, 0.0)


def solve_active_set(
    params: MechanismParams,
    B: float,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> ActiveSetResult:
    """Run a scalable active-set heuristic and return a feasible solution."""
    params.validate()
    logs = [
        "active-set is a heuristic that returns a feasible mode-stable solution",
        "the final fixed-mode continuous subproblem is solved by the KKT solver",
    ]

    modes = _initial_modes(params, B, tol)
    num_clients = len(np.asarray(params.d))
    last_result = None

    for iteration in range(1, max_iter + 1):
        S0, SI, SU, Sout = _partition_from_modes(modes)
        result = solve_fixed_mode(params, S0=S0, SI=SI, SU=SU, Sout=Sout, B=B, tol=tol)

        if not result.feasible:
            logs.append(f"iteration {iteration}: infeasible mode, moving lowest-score client to exit")
            if not _remove_lowest_score_client(modes, params):
                last_result = result
                break
            last_result = result
            continue

        last_result = result
        new_modes = list(result.mode)
        for idx, mode in enumerate(result.mode):
            if mode == "internal":
                if result.q[idx] <= tol:
                    new_modes[idx] = "zero"
                elif result.q[idx] >= params.lambda_k[idx] - tol:
                    new_modes[idx] = "upper"

        ir_violations, max_ic_regret = _diagnostics(params, result, tol)
        logs.append(
            f"iteration {iteration}: feasible, cost={result.total_cost:.6f}, "
            f"utility={result.server_utility:.6f}, ir_violations={ir_violations}, "
            f"max_ic_regret={max_ic_regret:.3e}"
        )

        if new_modes == modes:
            return ActiveSetResult(
                result=result,
                iterations=iteration,
                ir_violations=ir_violations,
                max_ic_regret=max_ic_regret,
                logs=logs,
            )
        modes = new_modes

    if last_result is None or not last_result.feasible:
        exit_modes = ["exit"] * num_clients
        S0, SI, SU, Sout = _partition_from_modes(exit_modes)
        last_result = solve_fixed_mode(params, S0=S0, SI=SI, SU=SU, Sout=Sout, B=B, tol=tol)

    ir_violations, max_ic_regret = _diagnostics(params, last_result, tol)
    return ActiveSetResult(
        result=last_result,
        iterations=max_iter,
        ir_violations=ir_violations,
        max_ic_regret=max_ic_regret,
        logs=logs,
    )


def active_set_solver(
    params: MechanismParams,
    B: float,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> ActiveSetResult:
    """Alias for solve_active_set."""
    return solve_active_set(params, B, max_iter, tol)
