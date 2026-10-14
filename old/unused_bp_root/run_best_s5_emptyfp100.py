"""After lambda_sweep.json exists, fine-tune the winner for 100 epochs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "bp_original" / "s5_emptyfp"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SWEEP = OUT / "lambda_sweep.json"


def pick_lambda() -> float:
    data = json.loads(SWEEP.read_text(encoding="utf-8"))
    ranking = data.get("ranking") or []
    if not ranking:
        raise SystemExit("no ranking in lambda_sweep.json")
    return float(ranking[0]["lambda"])


def main():
    if not SWEEP.exists():
        raise SystemExit(f"missing {SWEEP}")
    lam = pick_lambda()
    tag = f"l{lam:g}_ep100"
    log = OUT / f"train_{tag}.log"
    cmd = [
        str(PY), "-u", str(ROOT / "train_s5_emptyfp.py"),
        "--lambda-fp", str(lam),
        "--max-epoch", "100",
        "--es-patience", "40",
        "--tag", tag,
    ]
    print("BEST λ", lam, "-> 100 epochs", log, flush=True)
    with log.open("w", encoding="utf-8") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
