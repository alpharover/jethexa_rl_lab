# AGENTS.md — JetHexa RL Lab (agent working guide)

Scope: This file governs the entire `jethexa_rl` project. Any agent or script that edits files here must follow these rules.

Goals (short):
- Keep the sim physically correct first (P0/P1 physics suite green) and motion rulers honest (headless gates green) before any RL.
- Use the hermetic runner for all CI/headless checks to avoid local Python/MuJoCo drift.
- Never change MJCF joint ranges without explicit human sign‑off; HX‑35H actuator realism (torque clamps + slew) must stay intact.

Golden commands (use these exact one‑liners):
- Bootstrap once (creates `./ci-mjpython`):
  - `bash scripts/ci_bootstrap.sh && export MJPY=./ci-mjpython`
- Physics sanity (must be green):
  - `$MJPY $MJPY tools/p2_verify_physics.py --xml mjcf/jethexa_lab.xml --test all --seconds 2.0 --deltav 0.2 --push-steps 10`
- IK parity + circle pinning + IK-offset calibration:
  - `$MJPY $MJPY tools/p0p1_verify.py --xml mjcf/jethexa_lab.xml --ik all`
  - `$MJPY $MJPY tools/p2_tune_circle.py --xml mjcf/jethexa_lab.xml --out configs/policy/workspace_circle.json`
  - `$MJPY $MJPY tools/p2_calibrate_ik_offsets.py --xml mjcf/jethexa_lab.xml`
- Motion verifier (small preview gait, headless):
  - `$MJPY $MJPY tools/p2_verify_motion.py --xml mjcf/jethexa_lab.xml --seconds 8.0 --omega 0.05 --amp-scale 0.06`
- Viewer sanity (visual rulers only):
  - `$MJPY tools/view_policy.py --env configs/env/hex_v0.json --controller baseline:cpg_circle \
     --mode preview --lock-coxa --omega 0.05 --amp-scale 0.08 --overlay workspace,com,contacts,phases`

Non‑negotiables / guardrails:
- Physics truth:
  - External forces go via `d.xfrc_applied` and are cleared every step + guard (MuJoCo semantics).
  - Contact model: `condim=6`, friction vectors explicit; log any change to `solref/solimp`.
  - Actuator realism: position servos have finite `forcerange`; HX‑35H torque clamps: Coxa ±2.5 N·m; Femur/Tibia ±3.0 N·m. Target slew ≈ 5.82 rad/s.
- Geometry:
  - Do not change MJCF joint ranges unless the human owner approves in writing.
  - IK mapping uses calibrated per‑leg offsets from `configs/calib/ik_offsets.json`; keep this file in sync with XML SHA.
- Tests:
  - Run `pytest -q` before claiming green. P0/P1 physics must be green; motion baseline must pass; IK parity must pass.
- Repro:
  - Always prefer `$MJPY` runner over system Python. Do not add global pip deps.
- Data hygiene:
  - Never commit large artifacts. Keep outputs in `artifacts/` and `runs/`; ensure they remain in `.gitignore`.

Coding & style:
- Python ≥3.11; prefer type hints. Keep functions small; avoid hidden side effects.
- Use ripgrep (`rg`) for search. Keep patches surgical; don’t rename files gratuitously.
- When touching tools or control code, update `docs/MASTER_PLAN.md` (status + runbook) and append a line to `docs/SESSIONS.md`.

What to do if a gate fails:
- Physics failure → fix options/contacts/actuators, not the measuring code.
- Motion failure → check IK offsets, stride clamp (cm), PD gains, and `substeps` before loosening gates.

File ownership hints:
- MJCF: `mjcf/jethexa_lab.xml` is the source of truth for geometry and actuators.
- Circle policy: `configs/policy/workspace_circle.json` is the only source for controller radius.
- IK offsets: `configs/calib/ik_offsets.json` (per‑leg signs+offsets, with XML SHA).

Contact points:
- Architect expectations and gates are mirrored in `docs/MASTER_PLAN.md` (updated to v2.2). Keep it current.

