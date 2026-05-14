"""Mechanism parameter containers."""

from dataclasses import dataclass

import numpy as np


@dataclass
class MechanismParams:
    """Parameters shared by incentive mechanism routines."""

    d: np.ndarray
    lambda_k: np.ndarray
    S: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    V: np.ndarray
    lambda_base: float = 1.0

    def validate(self) -> None:
        """Validate shapes and parameter domains."""
        arrays = {
            "d": self.d,
            "lambda_k": self.lambda_k,
            "S": self.S,
            "alpha": self.alpha,
            "beta": self.beta,
            "V": self.V,
        }
        shapes = {name: np.asarray(value).shape for name, value in arrays.items()}
        unique_shapes = set(shapes.values())
        if len(unique_shapes) != 1:
            raise ValueError(f"Parameter shapes must match: {shapes}")

        d = np.asarray(self.d)
        lambda_k = np.asarray(self.lambda_k)
        S = np.asarray(self.S)
        alpha = np.asarray(self.alpha)
        beta = np.asarray(self.beta)
        V = np.asarray(self.V)

        if self.lambda_base <= 0:
            raise ValueError("lambda_base must be positive")
        if not np.all(d > 0):
            raise ValueError("d must be positive")
        if not np.all((0 <= lambda_k) & (lambda_k <= self.lambda_base)):
            raise ValueError("lambda_k must be in [0, lambda_base]")
        if not np.all(S >= 0):
            raise ValueError("S must be non-negative")
        if not np.all(alpha >= 0):
            raise ValueError("alpha must be non-negative")
        if not np.all(beta > 0):
            raise ValueError("beta must be positive")
        if not np.all(V > 0):
            raise ValueError("V must be positive")
