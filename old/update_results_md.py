"""
Regenerate results/RESULTS.md from current run + archived experiment KPIs.
Safe to call every epoch from train_skiponly.write_status.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "RESULTS.md"
CURRENT = RESULTS / "skiponly_covid_attn_gamma01"
ARCHIVE = RESULTS / "archive"

ORIGINAL_REF = {"mean": 0.76, "median": 0.84}
ADDITIVE_REF = {
    "name": "skiponly_covid_base32 (additive skips)",
    "mean_dice_nonempty": 0.7829,
    "median_dice_nonempty": 0.8391,
    "test_batch_dice": 0.7629,
}
ATTN_REF = {
    "name": "skiponly_covid_attn (pure AG)",
    "mean_dice_nonempty": 0.7787,
    "median_dice_nonempty": 0.8299,
    "test_batch_dice": 0.7729,
}


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _best_val_from_history(hist_path: Path):
    if not hist_path.exists():
        return None
    try:
        with hist_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        best = max(rows, key=lambda r: float(r.get("val_dice", -1.0)))
        last = rows[-1]
        return {
            "n_epochs": len(rows),
            "last_epoch": int(float(last.get("epoch", 0))),
            "last_val_dice": float(last.get("val_dice", float("nan"))),
            "best_val_dice": float(best.get("val_dice", float("nan"))),
            "best_val_epoch": int(float(best.get("epoch", 0))),
        }
    except Exception:
        return None


def _fmt(x, digits=4):
    if x is None:
        return "-"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def build_markdown() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = _load_json(CURRENT / "run_status.json") or {}
    report = _load_json(CURRENT / "report.json") or _load_json(CURRENT / "report_ep400.json")
    test_m = _load_json(CURRENT / "test_metrics.json")
    hist = _best_val_from_history(CURRENT / "history.csv")

    phase = status.get("phase", "n/a")
    epoch = status.get("epoch", "-")
    lines = [
        "# BT-VAE Results Log",
        "",
        f"_Auto-updated: {now}_",
        "",
        "## Active focus",
        "",
        "AG×γ + (1−γ)×ungated skip blend, γ_init=0.01 (COVID).",
        "",
        "## Current run - Attention Gate × gamma blend γ₀=0.01 (COVID)",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Result dir | `results/skiponly_covid_attn_gamma01/` |",
        f"| Model | `attn_skip_gamma`: x+=γ·AG+(1−γ)·ungated, γ_init=0.01 |",
        f"| Protocol | reuse BT encoder → SEG ≤400, early-stop patience=40 |",
        f"| Phase | **{phase}** |",
        f"| Epoch | **{epoch}** |",
        f"| Updated at | {status.get('updated_at', '-')} |",
    ]

    if phase == "bt":
        lines += [
            f"| BT best loss | {_fmt(status.get('best'), 5)} |",
            f"| BT bad epochs | {status.get('bad_epochs', '-')} |",
        ]
        val = status.get("val") or {}
        if val:
            lines += [
                f"| BT val loss | {_fmt(val.get('loss'), 5)} |",
                f"| BT val bt/dist | {_fmt(val.get('bt'), 5)} / {_fmt(val.get('dist'), 5)} |",
            ]
    elif phase in ("seg", "done"):
        lines += [
            f"| Best val Dice | {_fmt(status.get('best_dice', status.get('best')))} |",
            f"| SEG bad epochs | {status.get('bad_epochs', '-')} |",
        ]
        val = status.get("val") or {}
        if val:
            lines += [
                f"| Last val Dice | {_fmt(val.get('dice'))} |",
                f"| Last val IoU | {_fmt(val.get('iou'))} |",
                f"| Last val loss | {_fmt(val.get('loss'))} |",
            ]
        if hist:
            lines += [
                f"| History epochs | {hist['n_epochs']} (last={hist['last_epoch']}) |",
                f"| History best val Dice | {_fmt(hist['best_val_dice'])} @ ep {hist['best_val_epoch']} |",
            ]

    if test_m:
        lines += [
            "",
            "### TEST (batch-avg)",
            "",
            f"- Dice `{_fmt(test_m.get('dice'))}` | IoU `{_fmt(test_m.get('iou'))}` | loss `{_fmt(test_m.get('loss'))}`",
        ]

    per = (report or {}).get("test_per_image") or status.get("test_per_image")
    if per:
        lines += [
            "",
            "### TEST (per-image KPI)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| n / empty GT | {per.get('n', '-')} / {per.get('empty_gt', '-')} |",
            f"| mean Dice (all) | {_fmt(per.get('mean_dice_all'))} |",
            f"| median Dice (all) | {_fmt(per.get('median_dice_all'))} |",
            f"| **mean Dice (nonempty)** | **{_fmt(per.get('mean_dice_nonempty'))}** |",
            f"| **median Dice (nonempty)** | **{_fmt(per.get('median_dice_nonempty'))}** |",
            f"| mean IoU (all) | {_fmt(per.get('mean_iou_all'))} |",
        ]
        if report:
            h = report.get("history") or report.get("history_le400") or {}
            sb = report.get("seg_best_ckpt") or {}
            lines += [
                "",
                "### Report snapshot",
                "",
                f"- best val Dice `{_fmt(h.get('best_val_dice'))}` @ ep `{h.get('best_val_epoch', '-')}`",
                f"- last epoch `{h.get('last_epoch', '-')}` val Dice `{_fmt(h.get('last_val_dice', h.get('epoch400_val_dice')))}`",
                f"- SEG best ckpt ep `{sb.get('epoch', '-')}` dice `{_fmt(sb.get('best_dice'))}`",
            ]
            if sb.get("gammas"):
                lines.append(f"- best gammas (1=AG,0=additive): `{sb['gammas']}`")
            if status.get("gammas"):
                lines.append(f"- status gammas: `{status['gammas']}`")

    lines += [
        "",
        "### References",
        "",
        f"- original/ (no skips): mean~{ORIGINAL_REF['mean']:.2f}, median~{ORIGINAL_REF['median']:.2f}",
        f"- {ADDITIVE_REF['name']}: nonempty mean `{ADDITIVE_REF['mean_dice_nonempty']:.4f}`, "
        f"median `{ADDITIVE_REF['median_dice_nonempty']:.4f}` "
        f"(batch TEST Dice `{ADDITIVE_REF['test_batch_dice']:.4f}`)",
        f"- {ATTN_REF['name']}: nonempty mean `{ATTN_REF['mean_dice_nonempty']:.4f}`, "
        f"median `{ATTN_REF['median_dice_nonempty']:.4f}` "
        f"(batch TEST Dice `{ATTN_REF['test_batch_dice']:.4f}`)",
        "",
        "## Archived experiments",
        "",
        "| Experiment | Dataset / notes | TEST Dice | TEST IoU | Extra KPI |",
        "|---|---|---:|---:|---|",
    ]

    archive_rows = [
        (
            "skiponly_covid_attn_gamma",
            "COVID AG×γ blend γ_init=0.5, SEG ES patience=40",
            "results/skiponly_covid_attn_gamma/test_metrics.json",
            "results/skiponly_covid_attn_gamma/report.json",
        ),
        (
            "skiponly_covid_attn",
            "COVID pure AG skips, base32 lat1024, ep400 no ES",
            "results/skiponly_covid_attn/test_metrics.json",
            "results/skiponly_covid_attn/report_ep400.json",
        ),
        (
            "skiponly_covid_base32",
            "COVID additive skips, base32 lat1024, ~ep400+",
            "results/archive/skiponly_covid_base32/test_metrics.json",
            "results/archive/skiponly_covid_base32/report_ep400.json",
        ),
        (
            "skiponly_lat1024_base32",
            "BP SkipOnly base32 lat1024",
            "results/archive/skiponly_lat1024_base32/test_metrics.json",
            None,
        ),
        (
            "skiponly_lat1024",
            "BP SkipOnly (earlier)",
            "results/archive/skiponly_lat1024/test_metrics.json",
            None,
        ),
        (
            "archive_skip_base16",
            "SkipOnly base16",
            "results/archive/archive_skip_base16_20260920_1854/test_metrics.json",
            None,
        ),
        (
            "archive_lat512",
            "MS-GSF / lat512 archive",
            "results/archive/archive_lat512_20260918_0920/test_metrics.json",
            None,
        ),
        (
            "archive_base16",
            "earlier base16",
            "results/archive/archive_base16_20260917_0726/test_metrics.json",
            None,
        ),
        (
            "ms_gsf_bt_vae",
            "MS-GSF (unstable late)",
            "results/archive/ms_gsf_bt_vae/test_metrics.json",
            None,
        ),
    ]

    for name, note, tm_path, rp_path in archive_rows:
        tm = _load_json(ROOT / tm_path) if tm_path else None
        rp = _load_json(ROOT / rp_path) if rp_path else None
        dice = _fmt(tm.get("dice")) if tm else "-"
        iou = _fmt(tm.get("iou")) if tm else "-"
        extra = "-"
        if rp and rp.get("test_per_image"):
            p = rp["test_per_image"]
            extra = (
                f"nonempty mean/med "
                f"{_fmt(p.get('mean_dice_nonempty'))}/"
                f"{_fmt(p.get('median_dice_nonempty'))}"
            )
        lines.append(f"| `{name}` | {note} | {dice} | {iou} | {extra} |")

    lines += [
        "",
        "## Layout",
        "",
        "```",
        "results/",
        "  RESULTS.md                 # this file (live)",
        "  skiponly_covid_attn_gamma01/ # current AG×gamma γ_init=0.01",
        "  skiponly_covid_attn_gamma/   # prior γ_init=0.5 blend",
        "  skiponly_covid_attn/         # prior pure-AG COVID run",
        "  archive/                   # prior experiments",
        "```",
        "",
        "## Changelog",
        "",
        f"- {now}: refreshed from `run_status.json` / reports",
        "",
    ]
    return "\n".join(lines)


def main() -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    text = build_markdown()
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(OUT)
    return OUT


if __name__ == "__main__":
    p = main()
    print("updated", p)
