from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np


@dataclass
class CirclePolicy:
    # Provenance from calibration
    r_paper: float
    r_inscribed_min: float
    r_ctrl: float              # controller workspace radius actually used (m)
    s: float = 1.4             # paper scale factor used to reach r_ctrl
    alpha: float = 0.80        # inscribed radius scale used to reach r_ctrl
    # Geometric circle center offset from hip in {Hi1}
    d_cw: float = 0.0          # center offset along hip-x (m) from hip origin
    # Temporal/gait params
    height: float = 0.0        # nominal z offset in {Hi1}
    lift: float = 0.03         # step height (m)
    duty: float = 0.5          # stance fraction [0,1]
    omega: float = 2.0 * math.pi * 0.5  # rad/s step cycle (phase rate)


def foot_target_hi1(phi: float, pol: CirclePolicy) -> np.ndarray:
    """Parametric foot target over phase phi in {Hi1} using a simple cycloid lift.

    The circle center is at (d_cw, 0, 0); the path is a circular arc in XY
    with vertical cycloid lift during swing and z=height during stance.
    """
    # XY around the circle center; use full circle but lift during swing only
    x = pol.d_cw + pol.r_ctrl * math.cos(phi)
    y = 0.0 + pol.r_ctrl * math.sin(phi)
    # Lift profile: 0 in stance, smooth bump in swing
    if (phi % (2 * math.pi)) < (2 * math.pi) * (1.0 - pol.duty):
        # swing: bump shaped by sin
        t = (phi % (2 * math.pi)) / ((2 * math.pi) * (1.0 - pol.duty))
        z = pol.height + pol.lift * math.sin(math.pi * t)
    else:
        # stance
        z = pol.height
    return np.array([x, y, z], dtype=float)


# ---------------- Motion center and updates (stance/swing) ---------------- #

def motion_center(v_cmd: float, yawrate_cmd: float, psi_heading: float = 0.0) -> Tuple[np.ndarray, float]:
    """Compute common motion center in body frame and yaw increment per second.

    - v_cmd: commanded forward speed along body x (m/s)
    - yawrate_cmd: commanded yaw rate (rad/s)
    - psi_heading: world yaw of the body (rad); if provided, the motion center
      is returned in the body frame assuming heading psi.

    Returns (cm_xy, dpsi_dt). For |yaw| ~ 0, place the center at a large radius
    along left normal to approximate straight motion.
    """
    eps = 1e-6
    if abs(yawrate_cmd) < eps:
        # Approximate very large turning radius: pick rm from speed and a large cap
        rm = 1e6 if abs(v_cmd) > eps else 1e9
    else:
        rm = v_cmd / yawrate_cmd
    # Center lies on the left normal of forward direction (psi - pi/2)
    ang = psi_heading - math.pi/2.0
    cm = np.array([rm * math.cos(ang), rm * math.sin(ang)], dtype=float)
    return cm, float(yawrate_cmd)


def stance_update(ci_xy: np.ndarray, cm_xy: np.ndarray, dpsi: float) -> np.ndarray:
    """Rotate a grounded foot `ci_xy` around common center `cm_xy` by `dpsi`.
    Returns the updated XY.
    """
    vec = np.asarray(ci_xy, dtype=float) - np.asarray(cm_xy, dtype=float)
    c, s = math.cos(dpsi), math.sin(dpsi)
    R = np.array([[c, -s], [s, c]])
    return np.asarray(cm_xy, dtype=float) + R @ vec


def swing_update(ci_xy: np.ndarray, AEP_xy: np.ndarray, steps_left: int, z_mh: float,
                 z_apex: float, step_idx: int, step_count: int) -> Tuple[np.ndarray, float]:
    """Interpolate XY to AEP and shape Z with a smooth half-sine to apex then descend.

    - ci_xy: current foot XY in hip frame
    - AEP_xy: target anterior extreme position in hip frame
    - steps_left: remaining discrete steps in swing
    - z_mh: expected ground height (mean height) in hip frame
    - z_apex: apex height above z_mh during mid-swing
    - step_idx: current swing step index (0..step_count-1)
    - step_count: total steps in swing
    Returns (xy_next, z_next).
    """
    if step_count <= 1:
        return np.asarray(AEP_xy, dtype=float), float(z_mh)
    # Linear XY interpolation
    t = (step_idx + 1) / float(step_count)
    xy_next = (1.0 - t) * np.asarray(ci_xy, dtype=float) + t * np.asarray(AEP_xy, dtype=float)
    # Smooth vertical
    if t <= 0.5:
        # ascend
        tau = t / 0.5
        z = z_mh + z_apex * math.sin(math.pi * tau / 2.0)
    else:
        # descend
        tau = (t - 0.5) / 0.5
        z = z_mh + z_apex * math.sin(math.pi * (1.0 - tau) / 2.0)
    return xy_next, float(z)


# ---------------- Workspace intersections and stride limits ---------------- #

def circle_circle_intersections(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float) -> List[np.ndarray]:
    d = float(np.linalg.norm(c2 - c1))
    if d > r1 + r2 + 1e-9:  # separate
        return []
    if d < abs(r1 - r2) - 1e-9:  # one inside the other
        return []
    if d < 1e-9:  # coincident
        return []
    # Compute points
    a = (r1*r1 - r2*r2 + d*d) / (2*d)
    h2 = r1*r1 - a*a
    h = math.sqrt(max(0.0, h2))
    p2 = c1 + a * (c2 - c1) / d
    rx = -(c2 - c1)[1] * (h / d)
    ry =  (c2 - c1)[0] * (h / d)
    p3a = p2 + np.array([rx, ry])
    p3b = p2 - np.array([rx, ry])
    return [p3a, p3b]


def stride_limits_about_cw(cw_xy: np.ndarray, r_ctrl: float, cm_xy: np.ndarray, r_i: float) -> Optional[Tuple[float,float]]:
    """Return (theta_minus, theta_plus) angles (in radians) around cw where the
    circle about cm intersects the leg workspace circle around cw.
    Angles are measured around cw, in [−pi, pi]. None if no intersections.
    """
    pts = circle_circle_intersections(np.asarray(cw_xy, float), r_ctrl, np.asarray(cm_xy, float), float(r_i))
    if not pts:
        return None
    # Convert to angles around cw
    angs = [math.atan2((p - cw_xy)[1], (p - cw_xy)[0]) for p in pts]
    angs.sort()
    return float(angs[0]), float(angs[1])


class TripodPhaser:
    """Manage tripod grouping A/B with 180-deg phase offset."""

    def __init__(self, group_a: Tuple[str, ...], group_b: Tuple[str, ...]):
        self.group_a = tuple(group_a)
        self.group_b = tuple(group_b)

    def phase_for_leg(self, leg: str, base_phase: float) -> float:
        if leg in self.group_a:
            return base_phase
        if leg in self.group_b:
            return (base_phase + math.pi) % (2 * math.pi)
        return base_phase
