from __future__ import annotations

import numpy as np
from typing import Tuple, Iterable

from .ik_analytic import fk_hip_local
from .limit_buffer import LimitBuffer


def numeric_ik_hi1(
    p_Hi1: np.ndarray,
    a1: float,
    a2: float,
    a3: float,
    q0: Iterable[float] | None = None,
    limits: Iterable[Tuple[float, float]] | None = None,
    buf: LimitBuffer | None = None,
    iters: int = 40,
    tol: float = 1e-4,
    lam: float = 1e-3,
) -> np.ndarray:
    """Damped least-squares numeric IK in {Hi1} for the 3-DoF leg.

    Uses finite-difference Jacobian on (q1,q2,q3) with small step.
    Returns q that minimizes ||f(q) - p|| in Euclidean norm.
    """
    p = np.asarray(p_Hi1, dtype=float)
    q = np.array(q0 if q0 is not None else [0.0, 0.0, 0.0], dtype=float)
    buf = buf or LimitBuffer()
    if limits is None:
        limits = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]

    for _ in range(max(1, iters)):
        f = fk_hip_local(a1, a2, a3, q)
        e = p - f
        if float(np.linalg.norm(e)) < tol:
            break
        # Finite-difference Jacobian
        J = np.zeros((3, 3))
        eps = 1e-5
        for j in range(3):
            qj = q.copy()
            qj[j] += eps
            fj = fk_hip_local(a1, a2, a3, qj)
            J[:, j] = (fj - f) / eps
        # Damped least-squares step
        JT = J.T
        H = JT @ J + (lam ** 2) * np.eye(3)
        step = np.linalg.solve(H, JT @ e)
        # clamp step and update
        step = np.clip(step, -0.25, 0.25)
        q = q + step
        q = buf.project(q, limits)

    return q

