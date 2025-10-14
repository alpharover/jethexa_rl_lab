from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .limit_buffer import LimitBuffer


@dataclass
class LegGeom:
    a1: float  # hip yaw to femur axis offset (coxa link)
    a2: float  # femur length (projected)
    a3: float  # tibia length (to foot contact geom center)
    gamma: float = 0.0  # hip yaw offset about body; subtract from atan2


def fk_hip_local(a1: float, a2: float, a3: float, q: np.ndarray) -> np.ndarray:
    """Minimal 3-DoF FK in hip-local frame {Hi1}: x forward, y left, z up.

    Joint conventions:
      - q1: hip yaw about +z (azimuth in x–y)
      - q2: femur pitch in plane
      - q3: tibia pitch in plane
    """
    q1, q2, q3 = float(q[0]), float(q[1]), float(q[2])
    c1, s1 = math.cos(q1), math.sin(q1)
    # offset from yaw axis to femur pitch axis
    p1 = np.array([a1 * c1, a1 * s1, 0.0])
    c12, s12 = math.cos(q1 + q2), math.sin(q1 + q2)
    p2 = p1 + np.array([a2 * c12, a2 * s12, 0.0])
    c123, s123 = math.cos(q1 + q2 + q3), math.sin(q1 + q2 + q3)
    p3 = p2 + np.array([a3 * c123, a3 * s123, 0.0])
    return p3


def ik3_hi1_closed(p_Hi1: np.ndarray, g: LegGeom, branch: str = "knee_flex") -> Tuple[float, float, float]:
    """Closed-form IK in {Hi1} using standard 2-link planar geometry with lateral offset a1.

    Returns (q1,q2,q3) choosing the knee-flexed branch by default.
    """
    x, y, z = float(p_Hi1[0]), float(p_Hi1[1]), float(p_Hi1[2])
    r_xy = math.hypot(x, y)
    if r_xy < 1e-9:
        # Degenerate; aim along +x with minimal yaw
        q1 = -g.gamma
        x, y = 1e-6, 0.0
        r_xy = 1e-6
    q1 = math.atan2(y, x) - g.gamma

    # Reduce to planar 2R with effective target in the sagittal plane after removing a1
    x_eff = max(1e-9, r_xy - g.a1)
    z_eff = z
    L = math.hypot(x_eff, z_eff)
    L = max(min(L, g.a2 + g.a3 - 1e-9), 1e-9)

    # Law of cosines for knee angle (q3)
    cos_knee = (g.a2 * g.a2 + g.a3 * g.a3 - L * L) / (2.0 * g.a2 * g.a3)
    cos_knee = max(-1.0, min(1.0, cos_knee))
    knee = math.acos(cos_knee)

    if branch == "knee_flex":
        q3 = -knee  # flexed (negative if using our FK convention)
    else:
        q3 = +knee  # extended branch

    phi = math.atan2(z_eff, x_eff)
    cos_sh = (g.a2 * g.a2 + L * L - g.a3 * g.a3) / (2.0 * g.a2 * L)
    cos_sh = max(-1.0, min(1.0, cos_sh))
    theta = math.acos(cos_sh)
    q2 = phi + theta

    return (q1, q2, q3)


def project_into_limits(q: np.ndarray, limits: Tuple[Tuple[float, float], ...], buf: LimitBuffer | None = None) -> np.ndarray:
    buf = buf or LimitBuffer()
    return buf.project(np.asarray(q, dtype=float), limits)


def choose_branch(p_Hi1: np.ndarray, g: LegGeom, limits: Tuple[Tuple[float, float], ...], buf: LimitBuffer | None = None) -> np.ndarray:
    """Compute both branches and choose the one inside limits or with smaller FK error."""
    q_flex = np.array(ik3_hi1_closed(p_Hi1, g, "knee_flex"))
    q_ext = np.array(ik3_hi1_closed(p_Hi1, g, "knee_ext"))
    # Prefer knee-flex if both feasible
    q_flex_p = project_into_limits(q_flex, limits, buf)
    q_ext_p = project_into_limits(q_ext, limits, buf)
    # Use FK distance to choose if both are projected
    p = np.asarray(p_Hi1, dtype=float)
    e_flex = float(np.linalg.norm(fk_hip_local(g.a1, g.a2, g.a3, q_flex_p) - p))
    e_ext = float(np.linalg.norm(fk_hip_local(g.a1, g.a2, g.a3, q_ext_p) - p))
    return q_flex_p if e_flex <= e_ext else q_ext_p

