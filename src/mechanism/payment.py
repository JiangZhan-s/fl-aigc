"""Minimum implementable payment functions."""

from typing import Tuple

import numpy as np

from src.mechanism.client_response import phi


def ir_lower_bound_internal(
    d,
    lambda_k,
    S,
    alpha,
    beta,
    lambda_base: float = 1.0,
):
    """Return the internal IR lower bound ell_k^I."""
    a = lambda_base - lambda_k
    threshold = d * S - alpha * a

    if threshold <= 0:
        return 0.0

    radicand = a**2 + threshold / beta
    radicand = max(radicand, 0.0)
    return max(-a + np.sqrt(radicand), 0.0)


def cost_zero(
    d,
    lambda_k,
    S,
    alpha,
    lambda_base: float = 1.0,
    tol: float = 1e-9,
) -> Tuple[float, float]:
    """Return minimum cost and price for q=0, or infinity if infeasible."""
    phi_zero = phi(lambda_k, 0.0, lambda_base)

    if abs(phi_zero) <= tol:
        if abs(S) <= tol:
            return 0.0, 0.0
        return np.inf, np.inf

    price = S / phi_zero
    implementable = price <= lambda_base * alpha / d + tol
    if not implementable:
        return np.inf, np.inf

    return d * S, price


def cost_internal(
    q,
    d,
    lambda_k,
    S,
    alpha,
    beta,
    lambda_base: float = 1.0,
    tol: float = 1e-9,
) -> Tuple[float, float]:
    """Return minimum cost and price for an internal q, or infinity."""
    if not (tol < q < lambda_k - tol):
        return np.inf, np.inf

    a = lambda_base - lambda_k
    H = alpha * a + 2 * beta * a * q + beta * q**2 - d * S
    if H < -tol:
        return np.inf, np.inf

    cost = (alpha + 2 * beta * q) * (a + q)
    price = lambda_base * (alpha + 2 * beta * q) / d
    return cost, price


def cost_upper(
    d,
    lambda_k,
    S,
    alpha,
    beta,
    lambda_base: float = 1.0,
) -> Tuple[float, float]:
    """Return minimum cost and price for q=lambda_k."""
    response_price = lambda_base * (alpha + 2 * beta * lambda_k) / d
    ir_price = (d * S + alpha * lambda_k + beta * lambda_k**2) / d
    price = max(response_price, ir_price)
    return price * d, price


def min_payment(
    q,
    d,
    lambda_k,
    S,
    alpha,
    beta,
    lambda_base: float = 1.0,
    tol: float = 1e-9,
) -> Tuple[float, float, str]:
    """Dispatch to the relevant minimum payment function for q."""
    if abs(q) <= tol:
        cost, price = cost_zero(d, lambda_k, S, alpha, lambda_base, tol)
        return cost, price, "zero"

    if abs(q - lambda_k) <= tol:
        cost, price = cost_upper(d, lambda_k, S, alpha, beta, lambda_base)
        return cost, price, "upper"

    cost, price = cost_internal(q, d, lambda_k, S, alpha, beta, lambda_base, tol)
    return cost, price, "internal"
