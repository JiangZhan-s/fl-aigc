"""Fixed-mode continuous solver for the mechanism subproblem."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.mechanism.client_response import phi
from src.mechanism.params import MechanismParams
from src.mechanism.payment import cost_upper, cost_zero, ir_lower_bound_internal


@dataclass
class FixedModeResult:
    """Result returned by the fixed-mode solver."""

    feasible: bool
    q: np.ndarray
    p: np.ndarray
    x: np.ndarray
    total_cost: float
    server_utility: float
    mode: list


def _empty_result(num_clients: int, mode: list) -> FixedModeResult:
    return FixedModeResult(
        feasible=False,
        q=np.zeros(num_clients, dtype=np.float64),
        p=np.zeros(num_clients, dtype=np.float64),
        x=np.zeros(num_clients, dtype=np.float64),
        total_cost=np.inf,
        server_utility=-np.inf,
        mode=mode,
    )


def _internal_cost(q, alpha, beta, a):
    return (alpha + 2 * beta * q) * (a + q)


def _internal_price(q, d, alpha, beta, lambda_base):
    return lambda_base * (alpha + 2 * beta * q) / d


def solve_fixed_mode(
    params: MechanismParams,
    S0: Sequence[int],
    SI: Sequence[int],
    SU: Sequence[int],
    Sout: Sequence[int],
    B: float,
    tol: float = 1e-9,
    max_iter: int = 100,
) -> FixedModeResult:
    """Solve the continuous subproblem for a fixed client mode assignment."""
    params.validate()

    d = np.asarray(params.d, dtype=np.float64)
    lambda_k = np.asarray(params.lambda_k, dtype=np.float64)
    S = np.asarray(params.S, dtype=np.float64)
    alpha = np.asarray(params.alpha, dtype=np.float64)
    beta = np.asarray(params.beta, dtype=np.float64)
    V = np.asarray(params.V, dtype=np.float64)
    lambda_base = params.lambda_base
    num_clients = len(d)

    mode = ["exit"] * num_clients
    for idx in S0:
        mode[idx] = "zero"
    for idx in SI:
        mode[idx] = "internal"
    for idx in SU:
        mode[idx] = "upper"
    for idx in Sout:
        mode[idx] = "exit"

    q = np.zeros(num_clients, dtype=np.float64)
    p = np.zeros(num_clients, dtype=np.float64)
    x = np.zeros(num_clients, dtype=np.float64)
    total_cost = 0.0

    for idx in S0:
        cost, price = cost_zero(
            d[idx],
            lambda_k[idx],
            S[idx],
            alpha[idx],
            lambda_base,
            tol,
        )
        if not np.isfinite(cost):
            return _empty_result(num_clients, mode)
        p[idx] = price
        x[idx] = 1.0
        total_cost += cost

    for idx in SU:
        cost, price = cost_upper(
            d[idx],
            lambda_k[idx],
            S[idx],
            alpha[idx],
            beta[idx],
            lambda_base,
        )
        if not np.isfinite(cost):
            return _empty_result(num_clients, mode)
        q[idx] = lambda_k[idx]
        p[idx] = price
        x[idx] = 1.0
        total_cost += cost

    remaining_budget = B - total_cost
    if remaining_budget < -tol:
        return _empty_result(num_clients, mode)

    internal_indices = list(SI)
    if internal_indices:
        ell = np.array(
            [
                ir_lower_bound_internal(
                    d[idx],
                    lambda_k[idx],
                    S[idx],
                    alpha[idx],
                    beta[idx],
                    lambda_base,
                )
                for idx in internal_indices
            ],
            dtype=np.float64,
        )
        upper = lambda_k[internal_indices]

        if np.any(ell > upper + tol):
            return _empty_result(num_clients, mode)

        ell = np.minimum(ell, upper)
        a = lambda_base - lambda_k[internal_indices]
        alpha_i = alpha[internal_indices]
        beta_i = beta[internal_indices]
        V_i = V[internal_indices]

        lower_cost = float(np.sum(_internal_cost(ell, alpha_i, beta_i, a)))
        upper_cost = float(np.sum(_internal_cost(upper, alpha_i, beta_i, a)))

        if remaining_budget + tol < lower_cost:
            return _empty_result(num_clients, mode)

        if remaining_budget >= upper_cost - tol:
            q_i = upper.copy()
        else:
            def q_of_mu(mu):
                raw = (
                    V_i / (lambda_base * mu)
                    - alpha_i
                    - 2 * beta_i * a
                ) / (4 * beta_i)
                return np.clip(raw, ell, upper)

            mu_low = tol
            mu_high = 1.0
            while float(np.sum(_internal_cost(q_of_mu(mu_high), alpha_i, beta_i, a))) > remaining_budget:
                mu_high *= 2.0

            for _ in range(max_iter):
                mu_mid = 0.5 * (mu_low + mu_high)
                q_mid = q_of_mu(mu_mid)
                cost_mid = float(np.sum(_internal_cost(q_mid, alpha_i, beta_i, a)))
                if cost_mid > remaining_budget:
                    mu_low = mu_mid
                else:
                    mu_high = mu_mid

            q_i = q_of_mu(mu_high)

        internal_costs = _internal_cost(q_i, alpha_i, beta_i, a)
        q[internal_indices] = q_i
        p[internal_indices] = _internal_price(q_i, d[internal_indices], alpha_i, beta_i, lambda_base)
        x[internal_indices] = 1.0
        total_cost += float(np.sum(internal_costs))

    if total_cost > B + max(tol, 1e-8):
        return _empty_result(num_clients, mode)

    server_utility = float(np.sum(x * V * phi(lambda_k, q, lambda_base)) - total_cost)
    return FixedModeResult(
        feasible=True,
        q=q,
        p=p,
        x=x,
        total_cost=float(total_cost),
        server_utility=server_utility,
        mode=mode,
    )
