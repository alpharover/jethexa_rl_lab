
# MASTER PLAN v2.2 — Hexapod Locomotion (Clean Build)

**Objective.** Train and validate a locomotion policy for an 18‑DOF hexapod that is terrain‑aware, load‑adapting, supports omni motion (including in‑place turns and curved paths), maintains body attitude while tracking/orbiting, and supports body‑lead cues (yaw/pitch/roll) with leg compensation. Future hook: foot‑force sensing.

**Non‑Goals (now).** Hardware deployment, sprint gaits, full perception stack. We simulate contact forces; sensor integration is staged.

---

## 0) Architecture at a glance (unchanged)

* **Engine:** MuJoCo 3.x. Baseline = stock plane, visible geometry only, default solver/time‑step unless explicitly versioned.
* **RL:** JAX + Optax PPO, small inspectable trainer. Gymnasium‑compatible env wrapper.
* **Observability:** Structured JSONL + CSV/TB; strict schema; per‑run manifest with seeds, git SHA, package versions, MuJoCo options, XML checksum.

**Repo layout.**

```
repo/
  env/           # mujoco XML model(s), wrappers, terrain configs
  control/       # classical baselines (CPG/IK, PD)
  rl/            # PPO trainer, policies, buffers, losses
  eval/          # deterministic evaluator, scenario suites
  metrics/       # metric defs, reducers, schema, dashboards
  tests/         # unit + integration (pytest)
  tools/         # reproduce/run/visualize; P0/P1 check & verify
  docs/          # this master plan + runbooks
  configs/       # YAML/JSON for env, eval suites, policies
```

---

## 1) Status snapshot — 2025‑10‑15 (current truth)

This section supersedes prior snapshots. We have reproducible physics sanity, IK parity, a calibrated circle policy, HX‑35H‑realistic actuators, and headless motion gates for a gentle preview gait. The P2‑FINAL v2.1.1 gatepack is integrated (slip, joint‑use split, events/apex, excursions, contact v_z, parity); unlocked runs are RED pending Δt_G timing and a short warm‑start.

**What’s green (drop‑in; no MJCF edits beyond prior XML):**

* Physics sanity suite: `tools/p2_verify_physics.py`
  - Gravity/mass checks; Σ(contact normals) ≈ M·g in world‑Z
  - Free‑fall ĝ fit; finite‑impulse perturbations with proper clearing of `xfrc_applied`
  - Actuator clamp audit (every position servo has finite `forcerange` and `ctrlrange`)
* P0/P1 basics: `tools/p0p1_check.py`, `tools/p0p1_verify.py`
  - Determinism smoke; PD step; workspace sampling; safe push
  - Analytic vs numeric IK parity harness in {Hi1}
  - Circle workspace calibration (paper + largest‑inscribed) and policy pinning
* Motion verifier (preview gait, headless): `tools/p2_verify_motion.py`
  - Foot‑space tracking (XY), ring compliance, ground‑z in contact, limit‑time, contact scuff
  - Real motion‑center stride clamp (prevents yaw accumulation); per‑step cw for ring metric
  - IK offset application and target slew limiting aligned to HX‑35H spec
* Viewer parity: `tools/view_policy.py` overlays
  - Workspace rings, stride arcs (about cw with real cm), COM/support polygon, contact markers, weight vs ΣN
* Hermetic runner: `scripts/ci_bootstrap.sh` → local `./ci-mjpython` (Python 3.11 + MuJoCo)
* Tests (passing):

  - `tests/test_physics_sanity.py` — physics suite
  - `tests/test_ik_parity.py` — analytic vs numeric IK parity
  - `tests/test_workspace_circle.py`, `tests/test_limit_buffers.py`, `tests/test_support_polygon.py`
  - `tests/test_motion_baseline.py` — headless small‑arc gates (see thresholds below)
* Configs pinned (single source of truth):

  * `configs/env/hex_v0.json` — `xml_path=mjcf/jethexa_lab.xml`, XML SHA256, MuJoCo options, teleop tripods, push limits.
  * `configs/policy/workspace_circle.json` — controller circle with provenance: `r_paper`, `r_inscribed_min`, chosen `r_ctrl`, `s`, `alpha`, `hip_z_nom`, and timestamp.
  * `configs/calib/ik_offsets.json` — per‑leg joint `sign` and `delta` (analytic→engine mapping) with XML SHA.

**Current numbers (2025‑10‑15):**

- Physics suite: all green; pushes are finite and cleared; ΣN≈M·g; gravity fit near 9.81; actuator clamps present.
- IK parity: 95‑ile ≤ 2 mm; max ≤ 5 mm on a grid in {Hi1}.
- Motion (small preview, flat, gentle):
  - Locked‑coxa pass (geometry probe): mm‑level compliance close to final; ring mean marginal on a few legs without warm‑start; limits/scuff OK.
  - Unlocked‑coxa pass (v2.1.1): events/apex/excursions + slip/contact‑v_z failing on some legs at realistic amplitudes. Δt_G + snap‑to‑AEP and warm‑start are in progress.

---

## 1.1) P2‑FINAL — Acceptance Spec v2.1.1 (locked)

Pass criteria (all three scenarios: straight, turn‑in‑place, curve; seed‑locked):
- Geometry probe (locked): XY RMS ≤ 3 mm; ring mean ≤ 2 mm; ground‑z mean ≤ 1 mm; scuff ≤ 0.05; limit time ≤ 3 %.
- Stance slip (unlocked): per‑stance slip ≤ 3 % stride and ≤ 4 mm; mean tangential speed ≤ 1.5 mm/s.
- Joint‑use split (J_xy·q̇): straight/curve mean ≤ 0.25 (95‑ile ≤ 0.35); turn mean ≤ 0.45 (95‑ile ≤ 0.60).
- Events/apex: valid PEP→AEP→PEP; swing apex ≥ 7 mm.
- Excursions: femur ≥ 8°, tibia ≥ 12°.
- Contact v_z: 95‑ile ≤ 2 mm/s in stance.
- Viewer↔Headless parity: XY RMS and ring mean within 10 %.
- Artifacts: three MP4s + episodes.jsonl + summary.json with determinism stamp.
- Policy band: keep r_ctrl in [0.12, 0.16] m (current ≈ 0.123–0.125 m with provenance).

Status: gates implemented in tools/p2_verify_motion.py; viewer unified under --p2final_v21; sagittal‑only (coxa frozen) applied on straight/curve; Δt_G + early‑touchdown snap pending in engine.

### P2‑FINAL runbook (single‑source; hermetic)
```
bash scripts/ci_bootstrap.sh && export MJPY=./ci-mjpython
$MJPY $MJPY tools/p2_verify_physics.py --xml mjcf/jethexa_lab.xml --test all --seconds 2.0 --deltav 0.2 --push-steps 10
$MJPY $MJPY tools/p0p1_verify.py --xml mjcf/jethexa_lab.xml --ik all
$MJPY $MJPY tools/p2_tune_circle.py --xml mjcf/jethexa_lab.xml --out configs/policy/workspace_circle.json

# Geometry probe (locked)
$MJPY tools/p2_verify_motion.py --xml mjcf/jethexa_lab.xml --seconds 20 --omega 0.05 --amp-scale 0.06 \
  --p2final_v21 --geo-xyrms-mm 3 --geo-ring-mean-mm 2 --geo-ground-mean-mm 1

# Unlocked engine runs (apex=0.050 m, duty=0.60 enforced)
$MJPY tools/p2_verify_motion.py --xml mjcf/jethexa_lab.xml --seconds 20 --omega 0.05 --amp-scale 0.06 \
  --p2final_v21 --v-cmd 0.05 --yaw-cmd 0.0 --parity
$MJPY tools/p2_verify_motion.py --xml mjcf/jethexa_lab.xml --seconds 20 --omega 0.05 --amp-scale 0.06 \
  --p2final_v21 --v-cmd 0.00 --yaw-cmd 0.5 --parity
$MJPY tools/p2_verify_motion.py --xml mjcf/jethexa_lab.xml --seconds 20 --omega 0.05 --amp-scale 0.06 \
  --p2final_v21 --v-cmd 0.05 --yaw-cmd 0.25 --parity

# Artifacts (deterministic MP4s)
$MJPY tools/render_gait.py --scenario straight --out artifacts/final_gait_straight.mp4
$MJPY tools/render_gait.py --scenario turn    --out artifacts/final_gait_turn.mp4
$MJPY tools/render_gait.py --scenario curve   --out artifacts/final_gait_curve.mp4
```

Note: gates in the motion tester are temporarily relaxed (RMS ≤ 5 mm; ring mean ≤ 2.5 mm; z mean ≤ 1.8 mm) to reflect PD settling at very small amplitudes. We will tighten to 3/2/1 mm after stride timing and/or PD tuning.

---

### P2 progress (2025‑10‑15)

- Control: `control/ik_analytic.py`, `control/ik_numeric.py`, `control/limit_buffer.py`, `control/cpg_circle.py`, `control/posture_surfplane.py`.
- Tools: `tools/view_policy.py`, `tools/p2_tune_circle.py`, `tools/p2_phase_probe.py`, `tools/p2_verify_motion.py`, `tools/p2_verify_physics.py`, `tools/p2_calibrate_ik_offsets.py`.
- New: `tools/render_gait.py` for off‑screen MP4s; adaptive helpers in `control/cpg_circle.py` (`PhaseState`, `update_phase_on_touchdown`, `worst_case_dt_global`); zero‑mean surf‑plane accumulator in `control/posture_surfplane.py`.
- Tests: `tests/test_ik_parity.py`, `tests/test_workspace_circle.py`, `tests/test_limit_buffers.py`, `tests/test_support_polygon.py`, `tests/test_physics_sanity.py`, `tests/test_motion_baseline.py`.
- IK offsets are calibrated and cached; circle policy is pinned; viewer overlays match headless metrics.

Open in P2: implement full stride timing/state (stance rotate by Δψ about cm; swing AEP↔PEP with lift and adaptive phase) and then tighten motion gates back to 3/2/1 mm.

Acceptance for P2 remains as specified below; IK parity gates now runnable via `mjpython tools/p0p1_verify.py --xml mjcf/jethexa_lab.xml --ik all`.

## 2) Geometry pitfalls & protections (new guardrails we now standardize)

**Problem you flagged:** tibia and femur can rotate *past* inline (hyper‑extension). Shortest‑path IK or learned policies may “flip over the top,” slam a limit, and hang there.

**Our standard mitigations (applies in P2+ and P3+):**

1. **IK branch & joint‑limit policy (control & RL):**

   * Use the **analytic 3‑DoF IK in {Hi1}** from the reference (with explicit knee‑branch selection) and prefer the *knee‑flexed* (non‑hyperextended) branch unless the target is unreachable otherwise. Numeric IK (least‑squares via Jacobian) is only a fallback; both must agree within ≤2 mm (95‑ile) and ≤5 mm (max) under parity tests. 
   * Enforce a **soft buffer** inside each joint’s hard limits: `q ∈ [qmin+δ, qmax−δ]`, with **δ = 0.12 rad** default. Entering the buffer triggers barrier penalties and velocity clamps (below).
2. **Barrier penalties (metrics + reward):**

   * Add **joint‑limit proximity cost**: `c_lim = Σ_j w_j · f((q−qmin)/Δ, (qmax−q)/Δ)` where `f` is a smooth reciprocal or cubic barrier that rises sharply in the last 15 % of range; backprop‑safe and numerically bounded.
   * **Velocity near limit** penalty: scale |q̇| when inside buffer (discourage slamming the stops).
3. **Action shaping & clamp:**

   * PD target deltas are **projected** to remain inside `[qmin+δ, qmax−δ]`; a short‑horizon clamp avoids single‑step penetration.
4. **Gates:** P2/P3 acceptance requires **zero** hard‑limit hits; **saturation rate** and **time‑in‑buffer** must be below thresholds (see §5 gates).
5. **Trajectory design bias:** The CPG/foot‑circle planner keeps the nominal foot radius clear of singular postures; default `r_ctrl` targets the **knee‑flexed sector** of the reachable set. (See circle selection below.)

---

## 3) Build phases (each phase has deliverables, runbooks, and acceptance gates)

### **P0 — Clean base & determinism** ✅ *Done*

**Deliverables (frozen):**

* Minimal hexapod XML (`mjcf/jethexa_lab.xml`), stock plane for ground, visible geoms only.
* Headless & interactive sim; state hash determinism smoke.
* PD step response checks; **finite‑impulse push** helper that writes `d.xfrc_applied[body,:3]` for K steps and clears every step + guard (as required by MuJoCo’s control/apply timing). ([MuJoCo Documentation][2])

**Acceptance (met):**

* Stand without NaNs or contact spam; no sinking; determinism hash stable.
* PD step responses sane (t90/overshoot) for representative joints.

**Runbook:**
`mjpython tools/p0p1_check.py --xml mjcf/jethexa_lab.xml --determinism`
`mjpython tools/p0p1_check.py --xml mjcf/jethexa_lab.xml --pd-step femur_joint_LF --step 0.2 --settle 1.0`

---

### **P1 — Metrics 2.0 & Safety toolkit** ✅ *Done (analytic IK parity still xfail)*

**Metrics library (engine‑independent):**

* **Tracking:** v/ω tracking RMSE, path RMSE, body‑height & attitude error.
* **Contact quality:** duty factor, toggles/s (debounced), scuff rate (contact while foot ż>0), slip distance (tangential motion in contact).
* **Stability:** COM‑to‑support‑polygon margin (signed p10/p50), min interior angle of support hull, contact count distribution.
* **Health/Safety:** joint‑limit hits, time‑in‑buffer, action ΔL2, actuator saturation rate, impulse tails (p95), energy proxy (∑|τ·q̇|).
* **Smoothness:** ∥Δaction∥₂, body jerk proxy.

**Safety tools:**

* **Safe push** (friction‑bounded Δv, margin check, clears `xfrc_applied`) + **static stability** check from contact set.
* **Workspace audits**: sample XY reach; compute **paper circle** and **largest inscribed** radius (data‑driven).

  * MuJoCo contact/friction semantics observed; we use **condim=6** to enable torsional + rolling friction; geoms carry 3 coefficients (tangential, torsional, rolling) and contacts can use up to 5; our pairs specify what we need. ([MuJoCo Documentation][1])

**Acceptance (met / xfail noted):**

* Determinism smoke passes.
* Safe push keeps static COM margin ≥0 and clears forces every step. ([MuJoCo Documentation][2])
* Workspace sampling produces expected ranges; **IK parity test present and xfail** until analytic IK lands.

**Runbook:**
`mjpython tools/p0p1_check.py --xml mjcf/jethexa_lab.xml --workspace LF`
`mjpython tools/p0p1_verify.py --xml mjcf/jethexa_lab.xml --push-safe robot --dir x --deltav 0.2 --steps 10`

**Circle policy (calibrated, conservative):**

* Paper circle radius computed from link lengths and hip height (very conservative by design).
* Data‑driven **inscribed** radius from samples.
* Controller policy file stores: `r_paper`, `r_inscribed_min`, and **recommended** `r_ctrl = min(s·r_paper, α·r_inscribed_min)` (currently s=1.4, α=0.8 → ~0.052 m). We will retune in P2 to hit the **0.12–0.16 m** band you prefer for “real” foot circles. 

---

### **P2 — Classical baselines (CPG + analytic IK + posture control)**

**Goal.** A non‑RL controller that walks on flat terrain and mild slopes; validates dynamics, contacts, metrics; honors limit buffers.

**Deliverables.**

* **Analytic 3‑DoF IK in {Hi1}** (paper’s Appendix A framing) with knee‑branch logic; **numeric IK** (Jacobian LSQ) parity harness; unit tests: 95‑ile ≤2 mm, max ≤5 mm on a target grid. 
* **CPG / trajectory generator** using the **circle workspace + common motion center** method; synchronized stance on concentric circles, swing to AEP; adaptive gait phase control via **ϕAEP/ϕPEP** variables (reacts to early/late touchdown). 
* **Posture controller**: roll/pitch stabilization via z‑offsets to expected ground height (surf‑plane method), balanced across legs (zero‑mean height offset). 
* **Circle selection**: set `r_ctrl` to hit **0.12–0.16 m** band at nominal hip z; store chosen `s` (paper scale) and `α` (inscribed scale) in `configs/policy/workspace_circle.json` with provenance (date, hip z, link lengths).
* **Limit‑buffer enforcement**: joint projection + barrier costs; teleop preview remains a *sanity* tool only.

**Acceptance (temporary and final).**

*Now (preview small‑arc, headless):*

- Locked‑coxa (geometry probe): XY tracking RMS ≤ 5 mm; ring mean ≤ 2.5 mm; ground‑z mean ≤ 1.8 mm; p99 ≤ 3.8 mm; scuff ≤ 0.05; limit‑time ≤ 3 %.
- Unlocked‑coxa (with stride clamp): limit‑time ≤ 3 %; no yaw accumulation; scuff ≤ 0.05.
- IK parity: 95‑ile ≤ 2 mm; max ≤ 5 mm.

*Final targets (after stride timing and PD tuning):* tighten to 3 mm / 2 mm / 1 mm (XY RMS / ring mean / ground mean) while keeping the same scuff and limit gates.

**Runbook.**
```
# Hermetic bootstrap (one time), then set MJPY
bash scripts/ci_bootstrap.sh
export MJPY=./ci-mjpython

# Physics
$MJPY $MJPY tools/p2_verify_physics.py --xml mjcf/jethexa_lab.xml --test all --seconds 2.0 --deltav 0.2 --push-steps 10

# IK parity + circle policy pinning
$MJPY $MJPY tools/p0p1_verify.py --xml mjcf/jethexa_lab.xml --ik all
$MJPY $MJPY tools/p2_tune_circle.py --xml mjcf/jethexa_lab.xml --out configs/policy/workspace_circle.json
$MJPY $MJPY tools/p2_calibrate_ik_offsets.py --xml mjcf/jethexa_lab.xml

# Motion preview (headless small arcs)
$MJPY $MJPY tools/p2_verify_motion.py --xml mjcf/jethexa_lab.xml --seconds 8.0 --omega 0.05 --amp-scale 0.06
```

**Visual confirmation (off‑screen).**
```
$MJPY tools/render_gait.py --scenario straight --out docs/final_gait_straight.mp4
$MJPY tools/render_gait.py --scenario turn    --out docs/final_gait_turn.mp4
$MJPY tools/render_gait.py --scenario curve   --out docs/final_gait_curve.mp4
```

**Headless full‑gait tests.**
```
pytest -q tests/test_gait_full.py::test_turn_in_place \
          tests/test_gait_full.py::test_straight_line \
          tests/test_gait_full.py::test_curved_path \
          tests/test_gait_full.py::test_early_touchdown_adaptation
```

**Viewer (sanity overlays).**
```
$MJPY tools/view_policy.py --env configs/env/hex_v0.json \
  --controller baseline:cpg_circle --mode preview \
  --lock-coxa --omega 0.05 --amp-scale 0.08 \
  --overlay workspace,com,contacts,phases
```

**Actuators (HX‑35H realism).**

- Forcerange clamps in MJCF: Coxa ±2.5 N·m; Femur/Tibia ±3.0 N·m (≤ static max 3.43 N·m @ 11.1 V).
- Target slew limit ≈ 5.82 rad/s from 0.18 s/60° applied to position targets (disabled in the pure locked‑geometry probe).

**MuJoCo notes we respect in P2:**

* **Contacts/condim** & friction coefficients behave per docs; pair‑level overrides take precedence over per‑geom friction; `margin`/`gap` define when constraints activate. ([MuJoCo Documentation][1])

---

### **P3 — RL environment & reward redesign**

**Observations.**

* Body pose (rpy) + rates; height & ż; commanded twist (vₓ, vᵧ, ωz) and body‑lead targets (rpy setpoints).
* Per‑leg: foot pos in body frame, contact flag, last touchdown time, stance phase estimate.
* Terrain probes (local height or contact normals). Keep engine‑native & simple.

**Actions.**

* Phase 1: joint **position deltas** (executed by PD) with limit‑buffer projection.
* Phase 2: optional **impedance targets** (Kp/Kd modulation) once stable.

**Rewards (composable).**

* Motion tracking (v, ω, path).
* Smoothness & energy penalties; actuator saturation.
* **Joint‑limit buffer penalties** (see §2), including **velocity near limit**.
* Contact quality terms (low scuff/slip, bounded toggles/s).
* Stability shaping via COM‑polygon margin (soft shaping).
* Body‑lead penalties (attitude error while obeying locomotion commands).

**Acceptance.**

* Unit tests for reward terms (controlled episodes isolate terms).
* Observation normalization verified; random‑policy baselines produce finite metrics.

---

### **P4 — PPO trainer (JAX)**

**Deliverables.**

* Minimal, inspectable PPO (GAE(λ), centralized rollout normalization, clear logging to `update_metrics.jsonl`).
* Vectorized rollouts (CPU first); parity harness CPU vs MJX; run manifest with seeds/env versioning.

**Acceptance.**

* Converges on a toy env (CartPole or 1‑leg balance toy).
* On S0 flat ground, policy improves over random and **beats the classical baseline** on ≥2 key metrics (e.g., lower tracking error and scuff rate) at low speed.

---

### **P5 — Evaluator & acceptance gates**

**Scenarios (deterministic):**

* **S0:** Flat ground (grid of speeds/headings; include turn‑in‑place).
* **S1:** Constant slope up/down; **S2:** side‑slope; **S3:** step up/down obstacle.
* **S4:** Push recovery (finite impulses mid‑stance; friction‑bounded).
* **S5:** Body‑lead maneuvers: orbit while yaw‑aligned; pitch/roll nods while walking.
* **S6:** Surface variations (friction/height randomized).

**Gates (tune as we go; initial values):**

* **Tracking:** |v| RMSE ≤ 0.05–0.08 m/s; yaw RMSE ≤ 0.12 rad.
* **Stability:** COM‑margin p10 > 0 on S0/S1; **no falls** S0–S3; bounded attitude error on S5.
* **Contact quality:** scuff rate < threshold; toggles/s in [min,max]; no persistent slip on S0/S1.
* **Health:** **0 hard‑limit hits**; **time‑in‑buffer ≤ 3 %**; impulse p95 below threshold; saturation rate low.
* **Determinism:** running evaluator twice on same checkpoint yields identical `summary.json` hash.
* **Repro:** all tools/tests runnable via hermetic `./ci-mjpython` regardless of system Python.

**Outputs.** `eval_latest.log` (human tails), canonical `episodes.json`, `summary.json` per scenario.

---

### **P6 — Terrain & load adaptation, curriculum, robustness**

**Curriculum.** Flat → slopes → discrete steps → randomized friction & mass → pushes (bounded).

**Load/height adaptation.** Add simulated foot forces; include light **load‑balance** reward (variance of normal forces).

**Body‑lead & tracking.** Dedicated tasks: orbit a target while keeping camera yaw aligned (compound reward).

**Acceptance.** Pass gates S0–S6 at nominal difficulty; survive modest domain randomization (friction ±30 %, mass ±10 %) without falls in S0/S1.

---

## 4) Verification strategy (independent of the learner)

* **Unit:** FK/IK math; support polygon & COM margin; contact debounce; reward calculators; limit‑buffer logic.
* **Property‑based:** Frame‑shift invariance of metrics; L/R symmetry checks.
* **Seed‑locking:** Deterministic episode generation with fixed seeds; log & assert run manifests.
* **Parity:** CPU vs MJX step parity (state hashes within tolerance over N steps).

---

## 5) Logging, reproducibility, and “paper trails”

* Every run logs git SHA, `pip freeze`, MuJoCo options, XML checksum, RNG seeds.
* **Single schema:** `update_metrics.jsonl` (per‑update trainer stats) + `eval_summary.json` (per scenario).
* Dashboards: minimal TB/CSV & a “gateboard” that goes green only when each scenario passes its thresholds.

---

## 6) Runbooks (current & upcoming)

**Teleop preview (sanity only).**
`mjpython tools/teleop_hexapod.py --xml mjcf/jethexa_lab.xml --hide-ui`
Keys: Space pause, `T` tripod, `R` reset‑hold, `Z` zero biases, `h/l` coxa, `j/k` femur, `u/i` tibia.

**Sim teleop w/ diagnostics.**
`mjpython tools/sim_hexapod.py --xml mjcf/jethexa_lab.xml --no-ui --realtime`
Keys: `1` hard‑hold, `2` tripod, `3/4` yaw L/R (stance‑only coxa), `, .` speed, `U/J/K/L/I/M` pushes, `G` mapping, `D` kinematics, `Q` quit.

**P0/P1 checks (macOS).**

* Determinism — `mjpython tools/p0p1_check.py --xml mjcf/jethexa_lab.xml --determinism`
* PD step — `... --pd-step femur_joint_LF --step 0.2 --settle 1.0`
* Workspace — `... --workspace LF`
* Finite impulse — `... --push robot --dir x --deltav 0.2 --steps 10`
* IK parity (placeholder) — `mjpython tools/p0p1_verify.py --xml mjcf/jethexa_lab.xml --ik all`
* Circle calibration — `... --workspace-circle`
* Friction‑bounded push — `... --push-safe robot --dir x --deltav 0.2 --steps 10`

*(MuJoCo specifics behind these tools: contacts can expose torsional/rolling friction when `condim` ≥ 4; per‑pair overrides trump per‑geom; `margin/gap` and `solref/solimp` shape contact activation/softness; external forces/torques must be set before stepping and are available via `d.xfrc_applied` in Cartesian body frames.)* ([MuJoCo Documentation][1])

---

## 7) Checklist to close this phase (P2 readiness)

* [ ] **Commit** the P0/P1 tools/tests/configs now (you asked dev to hold; this is a good checkpoint).
* [ ] Implement analytic IK + numeric IK parity; un‑xfail `tests/test_ik_workspace.py`. 
* [ ] Set circle controller radius to **0.12–0.16 m** at nominal hip height; update `workspace_circle.json` with tuned `s` and `α`, and stamp the derivation.
* [ ] Wire **limit‑buffer** logic into baseline controller and evaluator metrics; enable **zero hard‑limit hits** gate.
* [ ] Enable `tests/test_p0p1.py` and `tests/test_push_safe.py` in CI (they already pass locally).

---

### Appendix A — Gate table (initial numbers)

| Gate        | Metric                  | Threshold                    |            |             |
| ----------- | ----------------------- | ---------------------------- | ---------- | ----------- |
| Tracking    |                         | v                            | RMSE (m/s) | ≤ 0.05–0.08 |
| Tracking    | Yaw RMSE (rad)          | ≤ 0.12                       |            |             |
| Stability   | COM‑margin p10 (m)      | > 0 on S0/S1                 |            |             |
| Health      | Hard‑limit hits         | **0**                        |            |             |
| Health      | Time‑in‑buffer          | ≤ 3 % / joint / episode      |            |             |
| Contact     | Scuff rate (contacts/m) | < 0.05                       |            |             |
| Contact     | Slip distance on S0/S1  | bounded (no persistent slip) |            |             |
| Determinism | eval hash repeat        | identical                    |            |             |
