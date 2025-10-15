"""Pytest bootstrap: ensure repository root is importable as a module.

Keeps tests importable regardless of invocation cwd by adding the repo root
to `sys.path` so `import control`, `import tools`, etc. resolve.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

