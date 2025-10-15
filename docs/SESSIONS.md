2025-10-12 — A0 scaffolded
2025-10-12 — A1: fresh base guarded; giant grep pruned; viewer proof (no ckpt); eval-schema smoke OK; MJCF vendored to repo.
2025-10-12 — A2: GPU env verification planned (CUDA+JAX+EGL); artifacts to .proofs/ and envs/a2-freeze.txt
2025-10-12 — A2: GPU env proven (CUDA+JAX+EGL), pip freeze saved, guard enforced (no resume)
2025-10-12 — A5: trainer skeleton (no learning), deterministic, metrics+schema, guard enforced
2025-10-12 — A6: supervised proxy bring-up (loss↓, schema OK), fresh-start enforced; no ckpts
2025-10-13 — P0/P1: added tools/p0p1_check.py (determinism, PD step, workspace, safe push), tests/test_p0p1.py + test_push_safe.py; viewer sanity working; initial circle policy written.
2025-10-13 — Model realism: MJCF position actuators clamped (forcerange) to HX‑35H class; contact pairs set condim=6; ground as infinite plane; servo slew‑limit planned.
2025-10-14 — Physics suite: tools/p2_verify_physics.py added; tests/test_physics_sanity.py green (gravity/mass, ΣN≈Mg, free‑fall ĝ, finite‑impulse hygiene, actuator clamps).
2025-10-14 — IK parity + calibration: analytic 3‑DoF IK ({Hi1}) parity green; tools/p2_calibrate_ik_offsets.py writes configs/calib/ik_offsets.json (per‑leg sign/delta, XML SHA).
2025-10-14 — Circle policy: tools/p2_tune_circle.py pins configs/policy/workspace_circle.json (r_paper, r_inscribed_min, r_ctrl, s, α, hip_z_nom, timestamp).
2025-10-14 — Motion verifier: tools/p2_verify_motion.py (real cm stride clamp, per‑step cw, contact age filter, IK offsets, slew‑limit); tests/test_motion_baseline.py green with temporary small‑arc gates (XY RMS ≤ 5 mm, ring mean ≤ 2.5 mm, z mean ≤ 1.8 mm; to be tightened after stride timing/PD tune).
2025-10-14 — Viewer parity: tools/view_policy.py overlays updated (workspace rings, stride arcs around cw using real cm, COM/support polygon, contact markers, Weight vs ΣN).
2025-10-14 — Hermetic runner: scripts/ci_bootstrap.sh creates ./ci-mjpython (Py 3.11 + MuJoCo) for headless tests.
2025-10-14 — Docs: MASTER_PLAN.md updated to v2.2 with current runbook, gates, HX‑35H specs, and next actions.
2025-10-15 — P2-FINAL: added adaptive phase helpers (early-touchdown snap + Δt_G), zero-mean surfplane accumulator; headless full-gait tests (turn/straight/curve + early-touchdown unit); off-screen renderer tools/render_gait.py; tuned workspace safety scale for calibration gate.
2025-10-15 — P2-FINAL (closeout): circle tuner sweep added; r_ctrl re-pinned to ~0.125 m (in-band 0.12–0.16) with artifacts/circle_sweep.jsonl; headless export written to artifacts/{episodes,summary}.json; full test suite green.
2025-10-15 — P2-FINAL v2.0 gates: integrated stance slip metrics, joint-use split (coxa vs femur+tibia), AEP/PEP event detector with swing apex check, femur/tibia excursion minima, and contact v_z gates in tools/p2_verify_motion.py; added parity harness and tests/test_gait_final.py; artifacts now include episodes.jsonl and determinism stamp.
2025-10-15 — P2-FINAL v2.1.1: viewer unified with engine under --p2final_v21; sagittal-only policy wired for straight/curve (coxa frozen per-leg), yaw allowed for turn. Verifier now logs at sim-dt, small-arc amp scales XY radius (not omega). Locked-geometry probe numerics tightened; unlocked runs currently red on events/apex/excursions/slip; Δt_G + early-touchdown snap + warm-start in progress.
