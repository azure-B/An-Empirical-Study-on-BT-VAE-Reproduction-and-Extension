"""Wait for dual_role_train (cls=none), then train cls=emptyfp variant."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
NONE_LOG = ROOT / "results" / "bp_original" / "dual_role_train" / "run.log"
NONE_DONE = ROOT / "results" / "bp_original" / "dual_role_train" / "report.json"
OUT_EFP = ROOT / "results" / "bp_original" / "dual_role_train_emptyfp"
LOG_EFP = OUT_EFP / "run.log"


def main():
    OUT_EFP.mkdir(parents=True, exist_ok=True)
    print("waiting for dual_role_train (cls=none) to finish…", flush=True)
    while not NONE_DONE.exists():
        # also bail if log shows traceback
        if NONE_LOG.exists():
            txt = NONE_LOG.read_text(encoding="utf-8", errors="replace")
            if "Traceback" in txt and "wrote" not in txt[-500:]:
                # still running warm/ft may not have wrote; only fail if process dead — just wait
                pass
        time.sleep(30)
    print("none-cls done; starting emptyfp-cls dual-role train", flush=True)
    with LOG_EFP.open("w", encoding="utf-8") as f:
        r = subprocess.run(
            [str(PY), "-u", str(ROOT / "train_dual_role.py"), "--cls-init", "emptyfp"],
            cwd=str(ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    print(f"emptyfp-cls exit={r.returncode}", flush=True)
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
