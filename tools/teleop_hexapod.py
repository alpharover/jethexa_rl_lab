#!/usr/bin/env python3
# Minimal MuJoCo teleop for the JetHexa model.
# - Keys affect *robot* (not just the viewer) via key_callback
# - Uses viewer.launch_passive (we control stepping)
# - Initializes position-actuator targets to current qpos (stable stand)

import argparse, time, math
from dataclasses import dataclass

import numpy as np
import mujoco as mj
import mujoco.viewer as mjv

# ---------- small helpers ----------

def actuator_joint_qpos_indices(model: mj.MjModel):
    """
    Map each actuator -> the qpos index of its driven joint.
    Assumes joint transmissions (true for our XML).
    """
    # For each actuator, actuator_trnid[i,0] gives attached joint id
    jids = model.actuator_trnid[:, 0].copy()
    qpos_adr = model.jnt_qposadr[jids]
    jtypes = model.jnt_type[jids]
    # We only expect hinge joints (1 DOF) for actuated joints here.
    # Ignore anything that's not a hinge/slide.
    mask = np.logical_or(jtypes == mj.mjtJoint.mjJNT_HINGE,
                         jtypes == mj.mjtJoint.mjJNT_SLIDE)
    return qpos_adr, mask


def sync_ctrl_to_current_qpos(model: mj.MjModel, data: mj.MjData):
    """Set each position actuator's target to the current joint angle."""
    qpos_adr, mask = actuator_joint_qpos_indices(model)
    data.ctrl[:] = 0.0  # default
    data.ctrl[mask] = data.qpos[qpos_adr[mask]]


def id_of(model: mj.MjModel, objtype: int, name: str) -> int:
    try:
        return mj.mj_name2id(model, objtype, name)
    except Exception:
        return -1


@dataclass
class GaitState:
    enabled: bool = False
    t0: float = 0.0
    freq_hz: float = 0.9
    A_coxa: float = 0.00    # rad (disabled to keep gait in place)
    A_femur: float = 0.30   # rad
    A_tibia: float = 0.45   # rad


class Teleop:
    def __init__(self, model: mj.MjModel, data: mj.MjData):
        self.m = model
        self.d = data

        # Build per-leg actuator indices by name for a toy tripod demo.
        # Names from your XML: pos_coxa_joint_XX / pos_femur_joint_XX / pos_tibia_joint_XX
        legs = ["LF", "LM", "LR", "RF", "RM", "RR"]
        self.act = {}
        for leg in legs:
            self.act[leg] = dict(
                coxa=id_of(self.m, mj.mjtObj.mjOBJ_ACTUATOR, f"pos_coxa_joint_{leg}"),
                femur=id_of(self.m, mj.mjtObj.mjOBJ_ACTUATOR, f"pos_femur_joint_{leg}"),
                tibia=id_of(self.m, mj.mjtObj.mjOBJ_ACTUATOR, f"pos_tibia_joint_{leg}"),
            )

        # Tripod groups (classic):
        # Tripod groups (classic diagonals):
        #   A = LF + RM + LR,  B = RF + LM + RR
        self.tripod1 = {"LF", "RM", "LR"}
        self.tripod2 = {"RF", "LM", "RR"}

        # Targets we maintain (start at current qpos)
        qpos_adr, mask = actuator_joint_qpos_indices(self.m)
        self.qpos_adr = qpos_adr
        self.addr_mask = mask
        self.ctrl_target = np.zeros(self.m.nu)
        self.ctrl_target[mask] = self.d.qpos[qpos_adr[mask]]

        self.paused = False
        self.gait = GaitState(enabled=False)

        # Small global offsets you can nudge from the keyboard
        self.coxa_bias = 0.0
        self.femur_bias = 0.0
        self.tibia_bias = 0.0

        # Per-leg signs
        # - Coxa: flip by side so outward/inward swing is consistent visually
        # - Femur/Tibia: compute empirical signs so positive deltas help lift the foot
        self.coxa_side_sign = {leg: (1.0 if leg.startswith('L') else -1.0)
                               for leg in legs}
        self.lift_sign = {leg: {"femur": 1.0, "tibia": 1.0} for leg in legs}
        self._calibrate_lift_signs()

    def reset_targets_to_current(self):
        self.ctrl_target[self.addr_mask] = self.d.qpos[self.qpos_adr[self.addr_mask]]
        print("[teleop] hold targets snapped to current qpos")

    def toggle_pause(self):
        self.paused = not self.paused
        print(f"[teleop] paused={self.paused}")

    def toggle_gait(self):
        self.gait.enabled = not self.gait.enabled
        self.gait.t0 = time.perf_counter()
        print(f"[teleop] tripod_gait={'ON' if self.gait.enabled else 'OFF'}")

    def zero_bias(self):
        self.coxa_bias = self.femur_bias = self.tibia_bias = 0.0
        print("[teleop] biases reset to 0.0")

    def apply_hold(self):
        """Hold ctrl at ctrl_target (with any current bias)."""
        # Clamp to ctrlrange (safety)
        lo = self.m.actuator_ctrlrange[:, 0]
        hi = self.m.actuator_ctrlrange[:, 1]
        u = self.ctrl_target.copy()
        # Add global biases by joint type
        for leg, idxs in self.act.items():
            s_coxa = self.coxa_side_sign.get(leg, 1.0)
            for k, bias in [
                ("coxa", self.coxa_bias),
                ("femur", self.femur_bias),
                ("tibia", self.tibia_bias),
            ]:
                a = idxs[k]
                if a >= 0:
                    if k == "coxa":
                        u[a] += s_coxa * bias
                    else:
                        s_lift = self.lift_sign.get(leg, {}).get(k, 1.0)
                        u[a] += s_lift * bias
        np.clip(u, lo, hi, out=u)
        self.d.ctrl[:] = u

    def apply_tripod_demo(self):
        """Very simple swaying tripod gait (for eyeballing only)."""
        t = time.perf_counter() - self.gait.t0
        w = 2.0 * math.pi * self.gait.freq_hz

        # start from the hold targets:
        u = self.ctrl_target.copy()

        def leg_delta(phase):
            # naive elliptical foot path encoded as joint deltas
            #  - coxa swings with sin
            #  - femur with sin shifted
            #  - tibia lifts on positive half-cycle
            dc = self.gait.A_coxa * math.sin(phase)
            df = self.gait.A_femur * math.sin(phase - math.pi / 2.0)
            z = max(0.0, math.sin(phase))  # swing half
            dt = self.gait.A_tibia * z
            return dc, df, dt

        for leg, idxs in self.act.items():
            group_phase = 0.0 if leg in self.tripod1 else math.pi
            phase = w * t + group_phase
            dc, df, dt = leg_delta(phase)
            s_coxa = self.coxa_side_sign.get(leg, 1.0)
            s_f = self.lift_sign.get(leg, {}).get("femur", 1.0)
            s_t = self.lift_sign.get(leg, {}).get("tibia", 1.0)
            if idxs["coxa"] >= 0:  u[idxs["coxa"]]  += s_coxa * (dc + self.coxa_bias)
            if idxs["femur"] >= 0: u[idxs["femur"]] += s_f * (df + self.femur_bias)
            if idxs["tibia"] >= 0: u[idxs["tibia"]] += s_t * (dt + self.tibia_bias)

        # clamp to ctrlrange
        lo = self.m.actuator_ctrlrange[:, 0]
        hi = self.m.actuator_ctrlrange[:, 1]
        np.clip(u, lo, hi, out=u)
        self.d.ctrl[:] = u

    # --- keyboard callback registered with the viewer ---
    def on_key(self, keycode):
        """Handle ASCII keycodes; ignore others. Never raises."""
        try:
            # Printable ASCII path
            if 0 <= keycode < 256:
                ch = chr(keycode)
                if ch == ' ':
                    self.toggle_pause(); return
                cl = ch.lower()
                if cl == 't': self.toggle_gait(); return
                if cl == 'r': self.reset_targets_to_current(); return
                if cl == 'z': self.zero_bias(); return
                if cl == 'h': self.coxa_bias  -= 0.03; return
                if cl == 'l': self.coxa_bias  += 0.03; return
                if cl == 'j': self.femur_bias -= 0.03; return
                if cl == 'k': self.femur_bias += 0.03; return
                if cl == 'u': self.tibia_bias -= 0.03; return
                if cl == 'i': self.tibia_bias += 0.03; return
        except Exception as e:
            print(f"[teleop] key handler error: {e}")

    # --- internal helpers ---
    def _foot_world_pos(self, geom_name: str):
        gid = id_of(self.m, mj.mjtObj.mjOBJ_GEOM, geom_name)
        if gid < 0:
            return None
        # geom_xpos is (ngeom,3) in Python bindings
        return np.array(self.d.geom_xpos[gid], dtype=float)

    def _calibrate_lift_signs(self):
        """Empirically pick femur/tibia signs so +delta increases foot Z height."""
        eps = 0.01
        # Ensure current state is consistent
        mj.mj_forward(self.m, self.d)
        for leg, idxs in self.act.items():
            gname = f"foot_{leg}"
            p0 = self._foot_world_pos(gname)
            if p0 is None:
                continue
            for k in ("femur", "tibia"):
                a = idxs[k]
                if a < 0:
                    continue
                qj = self.qpos_adr[a]
                qsave = float(self.d.qpos[qj])
                try:
                    self.d.qpos[qj] = qsave + eps
                    mj.mj_forward(self.m, self.d)
                    p1 = self._foot_world_pos(gname)
                    dz = (p1[2] - p0[2]) if p1 is not None else 0.0
                    s = 1.0 if dz >= 0 else -1.0
                    self.lift_sign[leg][k] = s
                finally:
                    self.d.qpos[qj] = qsave
                    mj.mj_forward(self.m, self.d)

    def step_once(self, dt):
        if self.paused:
            return
        # Choose control policy
        if self.gait.enabled:
            self.apply_tripod_demo()
        else:
            self.apply_hold()
        mj.mj_step(self.m, self.d)  # one physics step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, required=True,
                        help="Path to hexapod MJCF XML (your jethexa model).")
    parser.add_argument("--hz", type=float, default=1.0/0.002,  # matches timestep in your XML
                        help="Control loop rate in Hz (default = 1 / timestep).")
    parser.add_argument("--hide-ui", action="store_true",
                        help="Hide left/right viewer UI panels to avoid key overlap.")
    parser.add_argument("--coxa", type=float, default=None,
                        help="Override coxa swing amplitude (rad) for tripod demo (default 0.0).")
    args = parser.parse_args()

    m = mj.MjModel.from_xml_path(args.xml)
    d = mj.MjData(m)

    # Forward once, then hold current pose
    mj.mj_forward(m, d)
    sync_ctrl_to_current_qpos(m, d)

    tele = Teleop(m, d)
    if args.coxa is not None:
        tele.gait.A_coxa = float(args.coxa)

    # Print concise keymap once so it’s obvious how to drive the robot
    print("Teleop keys: [space]=pause | T=toggle gait | R=reset hold | Z=zero bias | h/l=coxa -/+ | j/k=femur -/+ | u/i=tibia -/+")
    try:
        side = " ".join(f"{leg}:{int(tele.coxa_side_sign.get(leg,1)):+d}" for leg in ["LF","LM","LR","RF","RM","RR"])
        lift = " ".join(f"{leg}:(F{int(tele.lift_sign.get(leg,{}).get('femur',1)):+d},T{int(tele.lift_sign.get(leg,{}).get('tibia',1)):+d})" for leg in ["LF","LM","LR","RF","RM","RR"])
        print(f"[teleop] coxa side sign (L=+1,R=-1): {side}")
        print(f"[teleop] lift signs femur/tibia: {lift}")
    except Exception:
        pass

    # passive viewer (we tick physics) + keyboard callback
    with mjv.launch_passive(m, d,
                            key_callback=tele.on_key,
                            show_left_ui=not args.hide_ui,
                            show_right_ui=not args.hide_ui) as v:
        # macOS requires running via `mjpython` for passive viewer
        # (you already are). We control timing and call sync() each loop.
        dt = 1.0 / args.hz
        t_next = time.perf_counter()
        while v.is_running():
            t_now = time.perf_counter()
            if t_now >= t_next:
                with v.lock():  # modify state under viewer lock
                    tele.step_once(dt)
                    v.sync()     # push updated state to GUI
                t_next += dt
            else:
                time.sleep(0.0005)

if __name__ == "__main__":
    main()
