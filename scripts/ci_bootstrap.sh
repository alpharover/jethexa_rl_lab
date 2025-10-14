#!/usr/bin/env bash
set -euo pipefail

# Create a local helper venv with uv so this repo can run MuJoCo headless tests
# on machines where system Python is PEP 668 managed or Python 3.13 (no mujoco wheel).

PY=${PY:-python3}
VENV=.ci-venv

if [[ ! -d "$VENV" ]]; then
  "$PY" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install -U pip >/dev/null
"$VENV/bin/python" -m pip install -U uv >/dev/null

cat > ci-mjpython <<'SH'
#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/.ci-venv/bin/uv" run -p 3.11 python "$@"
SH
chmod +x ci-mjpython

echo "Bootstrapped. Use: MJPY=./ci-mjpython ./ci-mjpython -V" 

