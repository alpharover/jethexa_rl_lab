# Dev→Architect Reply

**Summary:** A0 preflight scaffolded; map produced; probes and viewer proofs generated; Vast preflight attempted (read‑only).

**Changes:**
- Created scaffold: docs/comms/, configs/, envs/, eval/, scripts/, tools/, .proofs/
- Added docs/comms/ARCHITECT_TO_DEV.md; docs/SESSIONS.md updated
- Discovery outputs: .proofs/ls_tree.txt, grep_eval.txt, grep_trainer.txt, map.md
- Tools: tools/geom_probe.py, tools/gate_schema_check.py
- Viewer: scripts/viewer_cpu.py

**Commands Run:**
- mkdir -p …/jethexa_rl/{docs/comms,configs,envs,eval,scripts,tools,.proofs}
- ls -R old_lab > .proofs/ls_tree.txt; grep eval/train_mjx > .proofs/grep_*.txt
- python tools/geom_probe.py → geom_scan.json, workspace_probe.json
- python tools/gate_schema_check.py → gate_schema.json (ok_ratio reported)
- ssh -p 23626 root@199.68.217.31 ls …/checkpoints | tail -n 20 → vast_preflight.txt (welcome banner only)
- scp latest.npz → .proofs/ckpt/latest.npz (attempted)
- JAX_PLATFORMS=cpu MUJOCO_GL=glfw ~/.local/mjview/bin/python scripts/viewer_cpu.py --seconds 30 → viewer_proof.{json,png}

**Evidence:**
- Map: jethexa_rl/.proofs/map.md
- Geom: .proofs/geom_scan.json; Workspace: .proofs/workspace_probe.json
- Schema: .proofs/gate_schema.json (ok_ratio field; reason if coverage low)
- Vast: .proofs/vast_preflight.txt (2 lines banner)
- Viewer: .proofs/viewer/viewer_proof.png; viewer_proof.json (duration_s ≥ 25)

**Next Steps:**
- Confirm Vast endpoint/SSH key to list checkpoints and fetch latest.npz reliably.
- Decide schema gate interpretation (per‑file vs per‑field presence) and provide sample eval JSON if needed.
- Approve Phase A1 migration targets (rl2 subset, configs, mjcf) based on A0 map Keep/Fix/Drop.
