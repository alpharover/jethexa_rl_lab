import numpy as np
from control.limit_buffer import LimitBuffer


def test_limit_buffer_projection_and_cost():
    buf = LimitBuffer(delta_rad=0.12, barrier_gain=5.0)
    limits = [(-1.0, +1.0), (-0.5, +1.2), (-2.0, -0.2)]
    # Pick samples near/extreme limits
    qs = np.array([
        [-2.0, -1.0, -3.0],
        [-0.9, 0.0, -0.21],
        [0.95, 1.3, -0.25],
        [0.0, 0.0, -1.1],
    ])
    for q in qs:
        qp = buf.project(q, limits)
        for i, (lo, hi) in enumerate(limits):
            assert lo + buf.delta_rad - 1e-8 <= qp[i] <= hi - buf.delta_rad + 1e-8
        # cost is finite and non-negative
        c = buf.barrier_cost(qp, limits)
        assert c >= 0.0

