## A2 — Fresh GPU Env & Reproducibility Smoke (no training)

Checklist
- Allowed Paths:
  - envs/**
  - scripts/**
  - eval/**
  - docs/comms/**
  - mjcf/**
  - .proofs/**
  - Dockerfile.a2

- Acceptance gates (7):
  - ok_cuda_image
  - ok_jax_cuda
  - ok_mj_egl
  - ok_pip_lock
  - ok_fresh_guard
  - ok_docs
  - A2_ACCEPT.json pass_all=true

- Command overview:
  - Use a Vast CUDA 12.8 base image; transfer project (rsync/scp)
  - Remote block: install JAX cuda12 pip wheels and MuJoCo; run CUDA/JAX/EGL probes
  - Freeze environment and lock deps; run fresh-guard pass+block
  - Write artifacts to .proofs/* and envs/a2-freeze.txt

- Note: No training; no resumes; defaults safe/off.

# Architect→Dev Packet

- Title:
- Allowed paths:
- Commands:
- Acceptance:
- Deliverables:
- Rollback & Escalation:
