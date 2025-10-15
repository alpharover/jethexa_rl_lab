Final gait clips (P2‑FINAL)

Canonical clips are written under `artifacts/` (kept out of git):

- artifacts/final_gait_straight.mp4 — straight line preview
- artifacts/final_gait_turn.mp4 — in‑place turn preview
- artifacts/final_gait_curve.mp4 — curved path preview

Render them via the hermetic runner to avoid local MuJoCo drift:

  bash scripts/ci_bootstrap.sh && export MJPY=./ci-mjpython
  $MJPY tools/render_gait.py --scenario straight --out artifacts/final_gait_straight.mp4
  $MJPY tools/render_gait.py --scenario turn    --out artifacts/final_gait_turn.mp4
  $MJPY tools/render_gait.py --scenario curve   --out artifacts/final_gait_curve.mp4

If imageio is not available, the script emits PNG frames and prints the ffmpeg
command to assemble them into an MP4 to the same output path.
