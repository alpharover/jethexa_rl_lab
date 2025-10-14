from __future__ import annotations

"""Tiny policy loader for viewer: loads a simple linear/tanh policy from NPZ.

Expected NPZ keys: W (A x O), b (A,). Action = tanh(W @ obs + b).
If keys absent or file missing, returns a zero-action policy.
"""

from pathlib import Path
from typing import Callable

import numpy as np


def load_policy(checkpoint: str, obs_size: int, act_size: int) -> Callable[[np.ndarray], np.ndarray]:
    ckpt = Path(checkpoint)
    if not ckpt.exists():
        def zero(_obs: np.ndarray) -> np.ndarray:
            return np.zeros(act_size, dtype=np.float32)
        return zero
    try:
        data = np.load(ckpt, allow_pickle=False)
        W = data.get('W', None)
        b = data.get('b', None)
        if W is None or b is None:
            raise ValueError("Checkpoint missing W/b")
        W = np.array(W, dtype=np.float32)
        b = np.array(b, dtype=np.float32).reshape(-1)
        if W.shape != (act_size, obs_size) or b.shape != (act_size,):
            raise ValueError(f"Shape mismatch: W {W.shape} b {b.shape} expected {(act_size, obs_size)} and {(act_size,)}")
        def policy(obs: np.ndarray) -> np.ndarray:
            x = np.array(obs, dtype=np.float32).reshape(-1)
            u = W @ x + b
            return np.tanh(u)
        return policy
    except Exception:
        def zero(_obs: np.ndarray) -> np.ndarray:
            return np.zeros(act_size, dtype=np.float32)
        return zero

