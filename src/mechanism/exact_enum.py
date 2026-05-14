"""Exact mode enumeration for small-scale global benchmarks."""

import itertools
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.mechanism.fixed_mode_solver import FixedModeResult, solve_fixed_mode
from src.mechanism.params import MechanismParams


MODES = ("zero", "internal", "upper", "exit")


@dataclass
class ExactEnumResult:
    """Best fixed-mode result plus enumeration statistics."""

    best_result: Optional[FixedModeResult]
    searched_modes: int
    feasible_modes: int
    best_utility: float
    runtime_sec: float

    @property
    def feasible(self) -> bool:
        return self.best_result is not None and self.best_result.feasible


def _mode_sets(assignment):
    S0 = []
    SI = []
    SU = []
    Sout = []

    for idx, mode in enumerate(assignment):
        if mode == "zero":
            S0.append(idx)
        elif mode == "internal":
            SI.append(idx)
        elif mode == "upper":
            SU.append(idx)
        elif mode == "exit":
            Sout.append(idx)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return S0, SI, SU, Sout


def solve_exact_enum(
    params: MechanismParams,
    B: float,
    max_clients_for_exact: int = 12,
) -> ExactEnumResult:
    """Enumerate all mode assignments and return the best feasible solution."""
    params.validate()
    num_clients = len(np.asarray(params.d))

    if num_clients > max_clients_for_exact:
        raise ValueError(
            f"Exact enumeration is limited to {max_clients_for_exact} clients; "
            f"got {num_clients}"
        )

    start = time.perf_counter()
    searched_modes = 0
    feasible_modes = 0
    best_result = None
    best_utility = -np.inf

    for assignment in itertools.product(MODES, repeat=num_clients):
        searched_modes += 1
        S0, SI, SU, Sout = _mode_sets(assignment)
        result = solve_fixed_mode(params, S0=S0, SI=SI, SU=SU, Sout=Sout, B=B)

        if not result.feasible:
            continue

        feasible_modes += 1
        if result.server_utility > best_utility:
            best_result = result
            best_utility = result.server_utility

    runtime_sec = time.perf_counter() - start
    return ExactEnumResult(
        best_result=best_result,
        searched_modes=searched_modes,
        feasible_modes=feasible_modes,
        best_utility=float(best_utility),
        runtime_sec=runtime_sec,
    )


def exact_enumeration(
    params: MechanismParams,
    B: float,
    max_clients_for_exact: int = 12,
) -> ExactEnumResult:
    """Alias for solve_exact_enum."""
    return solve_exact_enum(params, B, max_clients_for_exact)
