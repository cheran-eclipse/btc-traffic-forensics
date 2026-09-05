"""Runs scripts/offline_selfcheck.py in a subprocess and checks it passes.

Subprocess (not in-process) so the script's socket monkey-patching doesn't
leak into the rest of the test session.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_full_pipeline_runs_offline():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "offline_selfcheck.py")],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ALL 10 STAGES RAN OFFLINE" in proc.stdout
    assert "OFFLINE VIOLATION" not in proc.stdout
