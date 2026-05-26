"""Client-side response functions for the incentive mechanism."""

import numpy as np


def phi(lambda_k, q, lambda_base: float = 1.0):
    """Compute quality coefficient phi_k(q_k)."""
    return (lambda_base - lambda_k + q) / lambda_base


def client_utility(
    q,
    p,
    d,
    lambda_k,
    S,
    alpha,
    beta,
    lambda_base: float = 1.0,
):
    """Compute client utility for a selected q and payment rate p."""
    return p * d * phi(lambda_k, q, lambda_base) - d * S - alpha * q - beta * q**2


def best_response_q(
    p,
    d,
    lambda_k,
    alpha,
    beta,
    lambda_base: float = 1.0,
):
    """Return the projected best-response q for a client."""
    raw = (p * d / lambda_base - alpha) / (2 * beta)
    return np.clip(raw, 0, lambda_k)


def participates(
    q,
    p,
    d,
    lambda_k,
    S,
    alpha,
    beta,
    lambda_base: float = 1.0,
    tol: float = 1e-9,
) -> bool:
    """Return whether a client accepts under individual rationality."""
    utility = client_utility(q, p, d, lambda_k, S, alpha, beta, lambda_base)
    return bool(utility >= -tol)
