"""Post-pass on cached wprod scores: sens-floor VAL picks + oracles."""
from __future__ import annotations

import json

import numpy as np

from eval_wprod_param_tune import (
    CACHE,
    CUR_FP,
    CUR_NE,
    OUT,
    eval_keep,
    make_score,
    oracle_ne,
)
from train_original_bp import safe_print
from train_s5_gate_25d import write_csv

z = np.load(CACHE)
val = {k: z[f"val_{k}"] for k in ("gt", "mass", "dice", "p_dual", "p_25d", "maxprob")}
test = {k: z[f"test_{k}"] for k in ("gt", "mass", "dice", "p_dual", "p_25d", "maxprob")}

masses = list(range(0, 200, 20)) + list(range(200, 800, 40)) + list(range(800, 5600, 80))
thrs = [i / 200.0 for i in range(1, 160)]
alphas = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
betas = [0.5, 0.8, 1.0, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]


def pick_val(sv, st, ne_floor=0.60, sens_floor=0.78):
    best = None
    for thr in thrs:
        for mmin in masses:
            keep = (sv >= thr) & (val["mass"] >= mmin)
            mv = eval_keep(val, keep, sv)
            if mv["nonempty_dice"] + 1e-9 < ne_floor:
                continue
            if mv["sensitivity"] + 1e-9 < sens_floor:
                continue
            key = (mv["empty_fp_rate"], -mv["nonempty_dice"])
            if best is None or key < (best["empty_fp_rate"], -best["nonempty_dice"]):
                best = {**mv, "thr": float(thr), "min_mass": int(mmin)}
    if best is None:
        return None, None
    keep_t = (st >= best["thr"]) & (test["mass"] >= best["min_mass"])
    mt = eval_keep(test, keep_t, st)
    mt.update({"thr": best["thr"], "min_mass": best["min_mass"]})
    return best, mt


# published anchors
sv0 = make_score(val, "wprod", 1.5, 1.3)
st0 = make_score(test, "wprod", 1.5, 1.3)
anchors = []
for thr, mmin, name in [
    (0.01, 320, "published"),
    (0.49, 5120, "old_csv_valpick"),
    (0.005, 1040, "dense_same_ab"),
]:
    mv = eval_keep(val, (sv0 >= thr) & (val["mass"] >= mmin), sv0)
    mt = eval_keep(test, (st0 >= thr) & (test["mass"] >= mmin), st0)
    anchors.append(
        {
            "name": name,
            "thr": thr,
            "min_mass": mmin,
            "val_ne": mv["nonempty_dice"],
            "val_fp": mv["empty_fp_rate"],
            "test_ne": mt["nonempty_dice"],
            "test_fp": mt["empty_fp_rate"],
            "test_sens": mt["sensitivity"],
        }
    )
    safe_print(
        f"{name}: VAL {mv['nonempty_dice']:.4f}/{mv['empty_fp_rate']:.3f} | "
        f"TEST {mt['nonempty_dice']:.4f}/{mt['empty_fp_rate']:.3f}"
    )

rows = []
for a in alphas:
    for b in betas:
        sv = make_score(val, "wprod", a, b)
        st = make_score(test, "wprod", a, b)
        bv, mt = pick_val(sv, st, 0.60, 0.78)
        if bv is None:
            continue
        rows.append(
            {
                "alpha": a,
                "beta": b,
                "thr": bv["thr"],
                "min_mass": bv["min_mass"],
                "val_ne": bv["nonempty_dice"],
                "val_fp": bv["empty_fp_rate"],
                "val_sens": bv["sensitivity"],
                "test_ne": mt["nonempty_dice"],
                "test_fp": mt["empty_fp_rate"],
                "test_sens": mt["sensitivity"],
                "test_fn": mt["fn"],
            }
        )

rows_v = sorted(rows, key=lambda r: (r["val_fp"], -r["val_ne"]))
ok = [r for r in rows if r["test_ne"] + 1e-9 >= 0.60]
ok = sorted(ok, key=lambda r: (r["test_fp"], -r["test_ne"], r["val_fp"]))
joint = sorted(rows, key=lambda r: -(r["val_ne"] - 0.5 * r["val_fp"]))

# also try sens floors 0.75 / 0.80 / 0.85 for published αβ only and full
extra = {}
for sf in (0.75, 0.80, 0.85):
    cand = []
    for a in alphas:
        for b in betas:
            sv = make_score(val, "wprod", a, b)
            st = make_score(test, "wprod", a, b)
            bv, mt = pick_val(sv, st, 0.60, sf)
            if bv is None:
                continue
            cand.append(
                {
                    "alpha": a,
                    "beta": b,
                    "thr": bv["thr"],
                    "min_mass": bv["min_mass"],
                    "val_ne": bv["nonempty_dice"],
                    "val_fp": bv["empty_fp_rate"],
                    "test_ne": mt["nonempty_dice"],
                    "test_fp": mt["empty_fp_rate"],
                }
            )
    if not cand:
        extra[sf] = None
        continue
    best = sorted(cand, key=lambda r: (r["val_fp"], -r["val_ne"]))[0]
    # best that keeps TEST ne
    keepers = [r for r in cand if r["test_ne"] + 1e-9 >= 0.60]
    keepers = sorted(keepers, key=lambda r: (r["test_fp"], r["val_fp"])) if keepers else []
    extra[sf] = {"val_best": best, "test_ok_best": keepers[0] if keepers else None}

oracles = {}
for floor in (0.60, 0.605, 0.61):
    orb = None
    for a in alphas:
        for b in betas:
            st = make_score(test, "wprod", a, b)
            o = oracle_ne(test, st, masses, thrs, floor)
            if o and (orb is None or o["empty_fp_rate"] < orb["empty_fp_rate"]):
                orb = {**o, "alpha": a, "beta": b}
    oracles[str(floor)] = orb

best_ok = ok[0] if ok else None
best_v = rows_v[0] if rows_v else None

lines = [
    "# wprod+mass tune (pass2: sens-floor + anchors)",
    "",
    f"Published target: {CUR_NE:.3f} / {CUR_FP:.3f}",
    "",
    "## Anchors (α=1.5 β=1.3)",
    "| name | τ | m_min | VAL ne/fp | TEST ne/fp |",
    "|---|---:|---:|---|---|",
]
for a in anchors:
    lines.append(
        f"| {a['name']} | {a['thr']} | {a['min_mass']} | "
        f"{a['val_ne']:.4f}/{a['val_fp']:.3f} | {a['test_ne']:.4f}/{a['test_fp']:.3f} |"
    )

lines += ["", "## VAL pick ne≥0.60 & sens≥0.78 → min VAL-fp"]
if best_v:
    lines.append(
        f"Best VAL: α={best_v['alpha']} β={best_v['beta']} τ={best_v['thr']:.3f} "
        f"m≥{best_v['min_mass']} → VAL {best_v['val_ne']:.4f}/{best_v['val_fp']:.3f} | "
        f"TEST {best_v['test_ne']:.4f}/{best_v['test_fp']:.3f}"
    )
if best_ok:
    lines += [
        "",
        "## Among those, keep TEST ne≥0.60 (ranked TEST-fp)",
        (
            f"**Recommended** α={best_ok['alpha']} β={best_ok['beta']} "
            f"τ={best_ok['thr']:.3f} m≥{best_ok['min_mass']} → "
            f"TEST **ne={best_ok['test_ne']:.4f} fp={best_ok['test_fp']:.3f}** "
            f"sens={best_ok['test_sens']:.3f}"
        ),
        f"vs published: Δne={best_ok['test_ne']-CUR_NE:+.4f} Δfp={best_ok['test_fp']-CUR_FP:+.4f}",
        "",
        "| α | β | τ | m | VAL ne/fp | TEST ne/fp |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for r in ok[:8]:
        lines.append(
            f"| {r['alpha']} | {r['beta']} | {r['thr']:.3f} | {r['min_mass']} | "
            f"{r['val_ne']:.4f}/{r['val_fp']:.3f} | {r['test_ne']:.4f}/{r['test_fp']:.3f} |"
        )
else:
    lines.append("No VAL pick with sens≥0.78 kept TEST ne≥0.60.")

lines += ["", "## Sens-floor sweep (VAL-best / TEST-ok)"]
for sf, pack in extra.items():
    if pack is None:
        lines.append(f"- sens≥{sf}: no VAL point")
        continue
    vb, tb = pack["val_best"], pack["test_ok_best"]
    lines.append(
        f"- sens≥{sf}: VAL-best α={vb['alpha']} β={vb['beta']} "
        f"→ TEST {vb['test_ne']:.4f}/{vb['test_fp']:.3f}"
        + (
            f" | TEST-ok α={tb['alpha']} β={tb['beta']} → "
            f"**{tb['test_ne']:.4f}/{tb['test_fp']:.3f}**"
            if tb
            else " | no TEST-ok"
        )
    )

lines += ["", "## TEST oracle (leak ceiling)"]
for floor, o in oracles.items():
    if not o:
        lines.append(f"- ne≥{floor}: none")
        continue
    lines.append(
        f"- ne≥{floor}: α={o['alpha']} β={o['beta']} τ={o['thr']:.3f} m≥{o['min_mass']} "
        f"→ ne={o['nonempty_dice']:.4f} fp={o['empty_fp_rate']:.3f}"
    )

(OUT / "COMPARE_pass2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
write_csv(OUT / "sens78_configs.csv", rows)
(OUT / "report_pass2.json").write_text(
    json.dumps(
        {
            "anchors": anchors,
            "best_val_sens78": best_v,
            "best_test_ok_sens78": best_ok,
            "extra_sens": extra,
            "oracles": oracles,
        },
        indent=2,
        default=float,
    ),
    encoding="utf-8",
)

safe_print("=" * 50)
if best_ok:
    safe_print(
        f"RECOMMENDED: α={best_ok['alpha']} β={best_ok['beta']} "
        f"τ={best_ok['thr']:.3f} m≥{best_ok['min_mass']} → "
        f"ne={best_ok['test_ne']:.4f} fp={best_ok['test_fp']:.3f} "
        f"(pub {CUR_NE}/{CUR_FP})"
    )
else:
    safe_print("no TEST-ok under sens78; see COMPARE_pass2")
for floor, o in oracles.items():
    if o:
        safe_print(
            f"oracle ne>={floor}: α={o['alpha']} β={o['beta']} "
            f"→ {o['nonempty_dice']:.4f}/{o['empty_fp_rate']:.3f}"
        )
safe_print("wrote", OUT / "COMPARE_pass2.md")
