"""After bt_cascade finishes:

1) presence imgcls (ResNet18 FLAIR)
2) fuse imgcls + cascade SEG (gate eval)
3) bt_cascade_focal (Dice+Focal SEG + focal cls + τ)
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASCADE_LOG = ROOT / "results" / "bp_original" / "bt_cascade" / "run.log"
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def cascade_done() -> bool:
    if not CASCADE_LOG.exists():
        return False
    text = CASCADE_LOG.read_text(encoding="utf-8", errors="ignore")
    return (
        "TEST learned tau final report wrote bt_cascade" in text
        or "TEST learned τ final report wrote bt_cascade" in text
        or (("wrote" in text) and ("bt_cascade" in text) and ("Presence T/F" in text))
    )


def run_step(name: str, script: str, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"==== start {name} → {log_path}", flush=True)
    cmd = [str(PY), "-u", str(ROOT / script)]
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
    print(f"==== done {name} exit={p.returncode}", flush=True)
    return int(p.returncode)


def main():
    print(f"waiting for cascade done… ({CASCADE_LOG})", flush=True)
    while not cascade_done():
        time.sleep(60)

    rc = run_step(
        "1_imgcls",
        "train_presence_img.py",
        ROOT / "results" / "bp_original" / "presence_imgcls" / "run.log",
    )
    if rc != 0:
        sys.exit(rc)

    rc = run_step(
        "2_fuse",
        "eval_presence_fuse.py",
        ROOT / "results" / "bp_original" / "presence_fuse" / "run.log",
    )
    if rc != 0:
        sys.exit(rc)

    rc = run_step(
        "3_cascade_focal",
        "train_bt_cascade_focal.py",
        ROOT / "results" / "bp_original" / "bt_cascade_focal" / "run.log",
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
