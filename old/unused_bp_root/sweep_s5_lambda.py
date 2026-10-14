"""Short λ probes from the same S5 best. 4 epochs each, then 100-ep winner."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "bp_original" / "s5_emptyfp"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
# 0.1 / 0.5 / 6.0 already logged; all collapse nonempty. Probe smaller.
LAMBDAS = (0.02, 0.05)
MAX_EP = 4


def run_probe(lam: float) -> None:
    tag = f"l{lam:g}"
    log = OUT / f"train_{tag}.log"
    cmd = [
        str(PY), "-u", str(ROOT / "train_s5_emptyfp.py"),
        "--lambda-fp", str(lam),
        "--max-epoch", str(MAX_EP),
        "--skip-final-eval",
        "--no-ckpt",
        "--tag", tag,
    ]
    print("RUN", " ".join(cmd), "->", log, flush=True)
    with log.open("w", encoding="utf-8") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"FAILED λ={lam} code={r.returncode}")


def row_from_history(lam: float) -> dict:
    hist = json.loads((OUT / f"history_l{lam:g}.json").read_text(encoding="utf-8"))
    init = hist[0]["init_val"]
    eps = [h for h in hist if "ep" in h]
    last = eps[-1]["val"] if eps else init
    best = max(eps, key=lambda h: float(h["val"]["mean_dice_nonempty"])) if eps else None
    return {
        "lambda": lam,
        "init_nonempty": init["mean_dice_nonempty"],
        "init_empty_fp": init["empty_fp_rate"],
        "last_nonempty": last["mean_dice_nonempty"],
        "last_empty_fp": last["empty_fp_rate"],
        "best_ep": None if best is None else best["ep"],
        "best_nonempty": None if best is None else best["val"]["mean_dice_nonempty"],
        "best_empty_fp": None if best is None else best["val"]["empty_fp_rate"],
        "epochs": [ {"ep": h["ep"], **h["val"]} for h in eps ],
    }


def row_from_lambda6_log() -> dict | None:
    p = OUT / "train_lambda6.log"
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8", errors="replace")
    init_m = re.search(r"nonempty=([0-9.]+) empty_fp=([0-9.]+)", text)
    ep_re = re.compile(
        r"ep (\d+)/\d+ .* nonempty=([0-9.]+) empty_fp=([0-9.]+)"
    )
    eps = [{"ep": int(m.group(1)), "mean_dice_nonempty": float(m.group(2)),
            "empty_fp_rate": float(m.group(3))} for m in ep_re.finditer(text)]
    ep4 = next((e for e in eps if e["ep"] == 4), eps[-1] if eps else None)
    if ep4 is None:
        return None
    return {
        "lambda": 6.0,
        "note": "from train_lambda6.log (ep4 used for ranking)",
        "init_nonempty": float(init_m.group(1)) if init_m else None,
        "init_empty_fp": float(init_m.group(2)) if init_m else None,
        "last_nonempty": ep4["mean_dice_nonempty"],
        "last_empty_fp": ep4["empty_fp_rate"],
        "epochs": eps[:4],
    }


def score_row(r: dict) -> float:
    ne = float(r["last_nonempty"])
    fp = float(r["last_empty_fp"])
    # keep lesion Dice; reward FP drop vs ~0.76 baseline
    return ne - 0.25 * fp - (0.20 if ne < 0.62 else 0.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for lam in LAMBDAS:
        hist = OUT / f"history_l{lam:g}.json"
        if hist.exists():
            print("skip existing", hist.name, flush=True)
            continue
        run_probe(lam)

    rows = []
    for lam in (0.02, 0.05, 0.1, 0.5):
        p = OUT / f"history_l{lam:g}.json"
        if p.exists():
            rows.append(row_from_history(lam))
    l6 = row_from_lambda6_log()
    if l6:
        rows.append(l6)

    ranked = sorted(rows, key=score_row, reverse=True)
    summary = {
        "probes": rows,
        "ranking": [
            {
                "score": score_row(r),
                "lambda": r["lambda"],
                "nonempty": r["last_nonempty"],
                "empty_fp": r["last_empty_fp"],
            }
            for r in ranked
        ],
    }
    (OUT / "lambda_sweep.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("WROTE", OUT / "lambda_sweep.json", flush=True)
    for r in summary["ranking"]:
        print(
            f"  λ={r['lambda']} score={r['score']:.4f} "
            f"nonempty={r['nonempty']:.4f} empty_fp={r['empty_fp']:.3f}",
            flush=True,
        )
    if not ranked:
        raise SystemExit("no probes to rank")
    best = float(ranked[0]["lambda"])
    print("BEST λ", best, "-> 100 epochs", flush=True)

    tag = f"l{best:g}_ep100"
    log = OUT / f"train_{tag}.log"
    cmd = [
        str(PY), "-u", str(ROOT / "train_s5_emptyfp.py"),
        "--lambda-fp", str(best),
        "--max-epoch", "100",
        "--es-patience", "40",
        "--tag", tag,
    ]
    print("RUN", " ".join(cmd), "->", log, flush=True)
    with log.open("w", encoding="utf-8") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
