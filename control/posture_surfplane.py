from __future__ import annotations

import numpy as np


def surf_plane_normal_from_body_xmat(xmat_body: np.ndarray) -> np.ndarray:
    """Estimate support plane normal from torso orientation (use body z-axis)."""
    R = xmat_body.reshape(3, 3)
    n = R[:, 2]
    n = n / (np.linalg.norm(n) + 1e-12)
    return n


def height_compensation(world_point: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray) -> float:
    """Signed height of a world point above the plane defined by (plane_point, normal)."""
    return float(np.dot(world_point - plane_point, plane_normal))


def apply_surfplane_compensation(p_foot_world: np.ndarray, torso_xpos: np.ndarray, torso_xmat: np.ndarray) -> float:
    """Compute z-compensation for a foot relative to a roll/pitch surf plane from torso.

    Returns the signed height above the plane; controller can add/subtract this to the target.
    """
    n = surf_plane_normal_from_body_xmat(torso_xmat)
    return height_compensation(p_foot_world, torso_xpos, n)


def per_leg_ground_height_offsets(leg_xy_world: dict[str, np.ndarray], torso_xpos: np.ndarray, torso_xmat: np.ndarray) -> dict[str, float]:
    """Compute Δh_gr,i = (n_x x_i + n_y y_i)/n_z per the paper, using torso surf-plane.

    Input leg_xy_world maps leg id -> [x,y] world coordinates of hip-projected foot.
    Returns dict leg-> height offset (meters).
    """
    R = torso_xmat.reshape(3, 3)
    n = R[:, 2]
    nx, ny, nz = float(n[0]), float(n[1]), float(n[2] + 1e-12)
    out = {}
    for leg, xy in leg_xy_world.items():
        x, y = float(xy[0] - torso_xpos[0]), float(xy[1] - torso_xpos[1])
        out[leg] = (nx * x + ny * y) / nz
    return out


class SurfPlaneAccumulator:
    """Accumulates expected ground heights with a gain G_i per leg.

    Maintains \hat{h}_{gr,i} by h <- (1-G) h + G * Δh, per tick.
    """

    def __init__(self, legs: list[str], G: float = 0.2):
        self.h = {leg: 0.0 for leg in legs}
        self.G = float(G)

    def update(self, deltas: dict[str, float]) -> dict[str, float]:
        for leg, dh in deltas.items():
            self.h[leg] = (1.0 - self.G) * self.h.get(leg, 0.0) + self.G * float(dh)
        return dict(self.h)

