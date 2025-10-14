from __future__ import annotations

"""Gymnasium-compatible env wrapper for the hexapod MuJoCo model.

Minimal implementation sufficient for viewer/trainer plumbing:
 - reset(seed) -> (obs, info)
 - step(action) -> (obs, reward, terminated, truncated, info)

Observations: concatenated (qpos, qvel).
Actions: joint position targets for position actuators (same length as nu).
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np
import mujoco


@dataclass
class EnvConfig:
    xml_path: str
    timestep: float = 0.002


class HexEnv:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.m = mujoco.MjModel.from_xml_path(cfg.xml_path)
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(0)
        # Ensure timestep matches config if provided
        if cfg.timestep:
            self.m.opt.timestep = float(cfg.timestep)
        mujoco.mj_forward(self.m, self.d)

    @property
    def observation_size(self) -> int:
        return int(self.m.nq + self.m.nv)

    @property
    def action_size(self) -> int:
        return int(self.m.nu)

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.d.qpos.copy(), self.d.qvel.copy()]).astype(np.float32)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        return self._get_obs(), {"time": 0.0}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Action as actuator targets; clip to ctrlrange
        a = np.array(action, dtype=float).reshape(-1)
        if a.size != self.m.nu:
            raise ValueError(f"action size {a.size} != nu {self.m.nu}")
        # Apply per-actuator ctrlrange if available
        lo = np.array(self.m.actuator_ctrlrange[:, 0]) if self.m.actuator_ctrlrange.size else -np.inf
        hi = np.array(self.m.actuator_ctrlrange[:, 1]) if self.m.actuator_ctrlrange.size else +np.inf
        self.d.ctrl[:] = np.clip(a, lo, hi)
        mujoco.mj_step(self.m, self.d)
        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False
        info = {"time": float(self.d.time)}
        return obs, reward, terminated, truncated, info

