from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple
import numpy as np


@dataclass
class LimitBuffer:
    """
    Project joint angles into an interior buffer and compute a soft barrier cost.

    The buffer keeps solutions away from hard limits to avoid chattering/instability.
    """
    delta_rad: float = 0.12  # default interior distance from hard limits
    barrier_gain: float = 10.0

    def project(self, q: np.ndarray, limits: Iterable[Tuple[float, float]]) -> np.ndarray:
        q = np.asarray(q, dtype=float).copy()
        for i, (lo, hi) in enumerate(limits):
            lo_b = lo + self.delta_rad
            hi_b = hi - self.delta_rad
            if lo_b > hi_b:  # pathological; fall back to midpoint
                mid = 0.5 * (lo + hi)
                q[i] = mid
            else:
                q[i] = min(max(q[i], lo_b), hi_b)
        return q

    def barrier_cost(self, q: np.ndarray, limits: Iterable[Tuple[float, float]]) -> float:
        """Soft cost that grows when q enters the buffered region and blows up near hard limits."""
        q = np.asarray(q, dtype=float)
        cost = 0.0
        eps = 1e-6
        for i, (lo, hi) in enumerate(limits):
            lo_b = lo + self.delta_rad
            hi_b = hi - self.delta_rad
            # penalize when inside the buffer (closer than delta to limit)
            if q[i] < lo_b:
                dist = max(eps, q[i] - lo)
                cost += self.barrier_gain * (1.0 / dist - 1.0 / (lo_b - lo)) ** 2
            elif q[i] > hi_b:
                dist = max(eps, hi - q[i])
                cost += self.barrier_gain * (1.0 / dist - 1.0 / (hi - hi_b)) ** 2
        return float(cost)

