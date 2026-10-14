"""Poll until COVID S5+none exist, then run train_covid_gate.py."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import covid_config as C

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def main():
    print(f"waiting for {C.S5_BEST} and {C.NONE_BEST}", flush=True)
    while True:
        if C.S5_BEST.exists() and C.NONE_BEST.exists():
            # also wait until SEG training process likely finished writing
            # (report.json written at end of each run_seg)
            if (C.RESULT_DIR / "s5_concat" / "report.json").exists():
                break
        time.sleep(30)
    print("SEG ready → train_covid_gate.py", flush=True)
    log = C.RESULT_DIR / "run_gate.log"
    with log.open("w", encoding="utf-8") as f:
        p = subprocess.run(
            [str(PY), "-u", str(ROOT / "train_covid_gate.py")],
            cwd=str(ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
