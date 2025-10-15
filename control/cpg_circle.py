from __future__ import annotations

import math
from dataclasses import dataclass, field
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


# ---------------- Adaptive phase helpers (early touchdown, global Δt_G) ---- #

@dataclass
class PhaseState:
    """Minimal per-leg phase state for adaptive swing/stance.

    - phi: current phase angle on the circle about cw (radians, [0, 2π))
    - mode: 'stance' or 'swing'
    - phi_AEP/phi_PEP: latest estimates of the anterior/posterior extreme
      positions for the current stride, used to re-synchronize on events.
    """
    phi: float
    mode: str
    phi_AEP: float
    phi_PEP: float


def wrap2pi(x: float) -> float:
    return float((x + 2.0 * math.pi) % (2.0 * math.pi))


def update_phase_on_touchdown(state: PhaseState, contact_now: bool) -> PhaseState:
    """Early touchdown adaptation: if a leg in 'swing' touches down, switch it to
    'stance' immediately and snap phase to the current AEP estimate. Stance legs
    keep their phase (they continue rotating on the concentric arc).

    This function is deliberately pure and tiny so tests can validate it without
    a running simulator.
    """
    if state.mode == "swing" and contact_now:
        return PhaseState(phi=wrap2pi(state.phi_AEP), mode="stance", phi_AEP=state.phi_AEP, phi_PEP=state.phi_PEP)
    return state


def worst_case_dt_global(legs: Dict[str, PhaseState], dpsi_dt: float, dt_nominal: float) -> float:
    """Global Δt_G selection: choose a time step that does not skip past any
    upcoming phase boundary across legs (simple, conservative rule).

    We look at stance legs (advancing by dpsi_dt) and swing legs (advancing
    their internal clock) and pick the minimum time to the next boundary among
    all legs, then clamp by dt_nominal.
    
    For the baseline we only consider stance arc rotation by dpsi_dt and treat
    swing as bounded by dt_nominal.
    """
    eps = 1e-9
    if abs(dpsi_dt) < eps:
        return float(dt_nominal)
    # time to hit AEP for stance legs (when theta reaches phi_AEP)
    t_hits: List[float] = []
    for st in legs.values():
        if st.mode == "stance":
            # angular distance along direction of rotation
            dphi = (st.phi_AEP - st.phi)
            # wrap to [0, 2π)
            dphi = (dphi + 2.0 * math.pi) % (2.0 * math.pi)
            t_hit = dphi / abs(dpsi_dt)
            t_hits.append(float(t_hit))
    t_min = min(t_hits) if t_hits else float(dt_nominal)
    return float(max(1e-6, min(dt_nominal, t_min)))


# ---------------- Event‑driven gait engine (v2.1 stance anchors) ------------- #

@dataclass
class GaitLegState:
    mode: str = "swing"                 # 'stance' or 'swing'
    anchor_W: Optional[np.ndarray] = None  # 3D world anchor latched at AEP
    pep_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    aep_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    swing_i: int = 0
    swing_N: int = 0


class GaitEngine:
    """Event‑driven gait with anchored stance in world (P2‑FINAL v2.1).

    The engine is kinematic and stateless w.r.t. the simulator: callers pass
    per‑step measurements (hip frames, foot world positions, contact flags) and
    obtain per‑leg foot targets in {Hi1} that implement:
      - Anchored stance: p_foot^W stays fixed at anchor_W (latched on AEP)
      - Swing: straight‑line PEP→AEP in XY with a small, filtered height bump
      - AEP selection: by stride‑limit intersection about the real motion center
      - Early touchdown: snap to stance immediately on contact
    """

    def __init__(self, legs: List[str], pol: CirclePolicy, swing_apex: float = 0.012, phaser: Optional[TripodPhaser] = None):
        self.legs = list(legs)
        self.pol = pol
        self.swing_apex = float(max(0.007, swing_apex))
        self.state: Dict[str, GaitLegState] = {leg: GaitLegState() for leg in legs}
        self.base_phi: float = 0.0
        # Default tripod phaser if none provided
        self.phaser = phaser or TripodPhaser(("LF","RM","LR"), ("RF","LM","RR"))

    def _choose_next_aep(self, hip_pos: np.ndarray, hx: np.ndarray, hy: np.ndarray,
                         torso_pos: np.ndarray, Rb: np.ndarray,
                         yaw_cmd: float, amp_scale: float) -> np.ndarray:
        # Compute stride limits about the real motion center in {Hi1}
        cm_body, _ = motion_center(0.0 if math.isnan(0.0) else 0.0, yaw_cmd, 0.0)
        cm_world = torso_pos + cm_body[0]*Rb[:,0] + cm_body[1]*Rb[:,1]
        cw = hip_pos + self.pol.d_cw * hx
        cm_xy = np.array([np.dot(cm_world-hip_pos, hx), np.dot(cm_world-hip_pos, hy)])
        cw_xy = np.array([np.dot(cw-hip_pos, hx), np.dot(cw-hip_pos, hy)])
        r_i = float(np.linalg.norm(cw_xy - cm_xy))
        lim = stride_limits_about_cw(cw_xy, float(self.pol.r_ctrl*amp_scale), cm_xy, r_i)
        th = None
        if lim is not None:
            th0, th1 = lim
            th = th1 if yaw_cmd >= 0.0 else th0
        if th is None:
            th = 0.0
        r_cmd = float(self.pol.r_ctrl*amp_scale)
        return np.array([self.pol.d_cw + r_cmd*math.cos(th), r_cmd*math.sin(th)], dtype=float)

    def step(self, dt: float,
             torso_pos: np.ndarray, Rb: np.ndarray,
             hip_frames: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
             foot_world: Dict[str, np.ndarray],
             contact: Dict[str, bool],
             yaw_cmd: float, amp_scale: float) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        # advance global phase
        self.base_phi = (self.base_phi + float(self.pol.omega) * float(dt)) % (2.0*math.pi)
        for leg in self.legs:
            hip_pos, hx, hy, hz = hip_frames[leg]
            st = self.state[leg]
            # desired mode by duty fraction
            phi_leg = self.phaser.phase_for_leg(leg, self.base_phi)
            phase_u = (phi_leg % (2.0*math.pi)) / (2.0*math.pi)
            desired_mode = 'stance' if phase_u < float(self.pol.duty) else 'swing'
            # Schedule by desired mode; adapt to contact later (early touchdown)
            if desired_mode == 'stance':
                if st.mode != 'stance':
                    st.mode = 'stance'
                    st.anchor_W = foot_world[leg].copy()
                    st.swing_i = st.swing_N = 0
            else:  # desired swing
                if st.mode != 'swing':
                    st.mode = 'swing'
                    cw = hip_pos + self.pol.d_cw * hx
                    v = foot_world[leg] - cw
                    st.pep_xy = np.array([np.dot(v, hx), np.dot(v, hy)], dtype=float)
                    st.aep_xy = self._choose_next_aep(hip_pos, hx, hy, torso_pos, Rb, yaw_cmd, amp_scale)
                    freq = max(0.2, float(self.pol.omega) / (2.0*math.pi))
                    swing_T = (1.0 - float(self.pol.duty)) / freq
                    st.swing_N = max(2, int(swing_T / max(1e-6, dt)))
                    st.swing_i = 0
                # Early touchdown adaptation: if we are in swing and have (re)gained contact after some progress, snap to stance
                if contact.get(leg, False) and st.swing_i > max(1, st.swing_N//4):
                    st.mode = 'stance'
                    st.anchor_W = foot_world[leg].copy()
                    st.swing_i = st.swing_N = 0
            # If in swing but desired is stance and contact happened early, anchor handled above

            # Targets
            if st.mode == 'stance' and st.anchor_W is not None:
                # anchored world target mapped into {Hi1}
                rel = st.anchor_W - hip_pos
                p_hi1 = np.array([np.dot(rel, hx), np.dot(rel, hy), self.pol.height], dtype=float)
            else:
                # straight-line swing with vertical bump
                if st.swing_N <= 0:
                    st.aep_xy = self._choose_next_aep(hip_pos, hx, hy, torso_pos, Rb, yaw_cmd, amp_scale)
                    st.swing_N = max(2, int(0.25 / max(1e-6, dt)))
                    st.swing_i = 0
                xy_next, z_next = swing_update(st.pep_xy, st.aep_xy, max(1, st.swing_N - st.swing_i), 0.0, self.swing_apex, st.swing_i, st.swing_N)
                st.swing_i = min(st.swing_N, st.swing_i + 1)
                p_hi1 = np.array([xy_next[0], xy_next[1], self.pol.height + z_next], dtype=float)

            out[leg] = p_hi1
        return out
