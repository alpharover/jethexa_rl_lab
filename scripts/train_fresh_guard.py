#!/usr/bin/env python3
import os, sys
from pathlib import Path

ALLOWED_RESUME = os.environ.get('ALLOW_RESUME','0') == '1'
RESUME = os.environ.get('RESUME_CKPT','')
SMOKE = os.environ.get('SMOKE','1') == '1'

if RESUME and not ALLOWED_RESUME:
    sys.stderr.write('[fresh-guard] RESUME BLOCKED: RESUME_CKPT is set\n')
    sys.exit(2)

# Fresh start accepted
proofs = Path('/Users/alpha_dev/robotics_repos/jethexa/jethexa_rl/.proofs')
proofs.mkdir(parents=True, exist_ok=True)
(proofs / 'fresh_start_ok.txt').write_text('fresh-start-ok')
print('[fresh-guard] fresh-start accepted (no resume)')

# In A1 we never invoke the real trainer on Mac; guard only.
sys.exit(0)
