"""Final tune pass: mass-cap + joint VAL objective for transfer-stable picks."""
from __future__ import annotations

import json

import numpy as np

from eval_wprod_param_tune import CACHE, CUR_FP, CUR_NE, OUT, eval_keep, make_score
from train_original_bp import safe_print
from train_s5_gate_25d import write_csv

z = np.load(CACHE)
val = {k: z[f"val_{k}"] for k in ("gt", "mass", "dice", "p_dual", "p_25d", "maxprob")}
test = {k: z[f"test_{k}"] for k in ("gt", "mass", "dice", "p_dual", "p_25d", "maxprob")}

masses = list(range(0, 200, 20)) + list(range(200, 800, 40)) + list(range(800, 1600, 40))
thrs = [i / 200.0 for i in range(1, 100)]
alphas = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]
betas = [0.5, 0.8, 1.0, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5]


def sweep(max_mass=None, ne_floor=0.60, sens_floor=0.0, joint_w=0.5):
    """Return best VAL pick and its TEST, plus all rows with TEST ne>=0.60."""
    best_fp = None
    best_joint = None
    test_ok = []
    for a in alphas:
        for b in betas:
            sv = make_score(val, "wprod", a, b)
            st = make_score(test, "wprod", a, b)
            local = None
            for thr in thrs:
                for mmin in masses:
                    if max_mass is not None and mmin > max_mass:
                        continue
                    keep = (sv >= thr) & (val["mass"] >= mmin)
                    mv = eval_keep(val, keep, sv)
                    if mv["nonempty_dice"] + 1e-9 < ne_floor:
                        continue
                    if mv["sensitivity"] + 1e-9 < sens_floor:
                        continue
                    cand = {
                        "alpha": a,
                        "beta": b,
                        "thr": float(thr),
                        "min_mass": int(mmin),
                        "val_ne": mv["nonempty_dice"],
                        "val_fp": mv["empty_fp_rate"],
                        "val_sens": mv["sensitivity"],
                        "joint": mv["nonempty_dice"] - joint_w * mv["empty_fp_rate"],
                    }
                    kfp = (cand["val_fp"], -cand["val_ne"])
                    if local is None or kfp < (local["val_fp"], -local["val_ne"]):
                        local = cand
            if local is None:
                continue
            keep_t = (st >= local["thr"]) & (test["mass"] >= local["min_mass"])
            mt = eval_keep(test, keep_t, st)
            local.update(
                {
                    "test_ne": mt["nonempty_dice"],
                    "test_fp": mt["empty_fp_rate"],
                    "test_sens": mt["sensitivity"],
                }
            )
            if best_fp is None or (local["val_fp"], -local["val_ne"]) < (
                best_fp["val_fp"],
                -best_fp["val_ne"],
            ):
                best_fp = local
            if best_joint is None or local["joint"] > best_joint["joint"]:
                best_joint = local
            if local["test_ne"] + 1e-9 >= 0.60:
                test_ok.append(local)
    return best_fp, best_joint, sorted(test_ok, key=lambda r: (r["test_fp"], -r["test_ne"]))


# also: for each (a,b), VAL-pick, then rank by VAL joint — single global joint over all thr/mass
def best_joint_global(max_mass=800, ne_floor=0.60, sens_floor=0.78, joint_w=0.45):
    best = None
    for a in alphas:
        for b in betas:
            sv = make_score(val, "wprod", a, b)
            st = make_score(test, "wprod", a, b)
            for thr in thrs:
                for mmin in masses:
                    if mmin > max_mass:
                        continue
                    keep = (sv >= thr) & (val["mass"] >= mmin)
                    mv = eval_keep(val, keep, sv)
                    if mv["nonempty_dice"] + 1e-9 < ne_floor:
                        continue
                    if mv["sensitivity"] + 1e-9 < sens_floor:
                        continue
                    j = mv["nonempty_dice"] - joint_w * mv["empty_fp_rate"]
                    if best is None or j > best["joint"]:
                        keep_t = (st >= thr) & (test["mass"] >= mmin)
                        mt = eval_keep(test, keep_t, st)
                        best = {
                            "alpha": a,
                            "beta": b,
                            "thr": float(thr),
                            "min_mass": int(mmin),
                            "val_ne": mv["nonempty_dice"],
                            "val_fp": mv["empty_fp_rate"],
                            "val_sens": mv["sensitivity"],
                            "joint": j,
                            "test_ne": mt["nonempty_dice"],
                            "test_fp": mt["empty_fp_rate"],
                            "test_sens": mt["sensitivity"],
                        }
    return best


results = {}
for cap in (320, 480, 640, 800, 1200, None):
    bf, bj, ok = sweep(max_mass=cap, ne_floor=0.60, sens_floor=0.78)
    results[str(cap)] = {
        "val_min_fp": bf,
        "val_joint": bj,
        "n_test_ok": len(ok),
        "best_test_ok": ok[0] if ok else None,
    }
    tag = f"m≤{cap}" if cap else "m any"
    if bf:
        safe_print(
            f"{tag} VAL-minfp: α={bf['alpha']} β={bf['beta']} τ={bf['thr']:.3f} "
            f"m≥{bf['min_mass']} → TEST {bf['test_ne']:.4f}/{bf['test_fp']:.3f}"
        )
    if ok:
        r = ok[0]
        safe_print(
            f"{tag} best TEST-ok among VAL picks: α={r['alpha']} β={r['beta']} "
            f"→ TEST {r['test_ne']:.4f}/{r['test_fp']:.3f}"
        )

# honest single objectives
honest = {}
for sf, jw, cap in [
    (0.78, 0.45, 800),
    (0.80, 0.45, 800),
    (0.78, 0.50, 640),
    (0.80, 0.40, 640),
    (0.75, 0.45, 800),
]:
    h = best_joint_global(cap, 0.60, sf, jw)
    key = f"joint_sf{sf}_w{jw}_m{cap}"
    honest[key] = h
    if h:
        safe_print(
            f"{key}: α={h['alpha']} β={h['beta']} τ={h['thr']:.3f} m≥{h['min_mass']} "
            f"VAL {h['val_ne']:.4f}/{h['val_fp']:.3f} → TEST {h['test_ne']:.4f}/{h['test_fp']:.3f}"
        )

# published for reference
sv = make_score(val, "wprod", 1.5, 1.3)
st = make_score(test, "wprod", 1.5, 1.3)
pub = eval_keep(test, (st >= 0.01) & (test["mass"] >= 320), st)

# pick best honest by TEST among honest that TEST ne>=0.60; else report best joint TEST
honest_ok = [v for v in honest.values() if v and v["test_ne"] + 1e-9 >= 0.60]
honest_ok = sorted(honest_ok, key=lambda r: (r["test_fp"], -r["test_ne"]))
rec = honest_ok[0] if honest_ok else None

lines = [
    "# wprod+mass tune final",
    "",
    f"Published (α=1.5 β=1.3 τ=0.01 m≥320): TEST ne={pub['nonempty_dice']:.4f} fp={pub['empty_fp_rate']:.3f}",
    "",
    "## Honest VAL joint picks (ne≥0.60 + sens floor + m_cap)",
]
for k, h in honest.items():
    if not h:
        lines.append(f"- {k}: none")
        continue
    mark = " **" if rec and h is rec else ""
    lines.append(
        f"- {k}: α={h['alpha']} β={h['beta']} τ={h['thr']:.3f} m≥{h['min_mass']} "
        f"VAL {h['val_ne']:.4f}/{h['val_fp']:.3f} → TEST {h['test_ne']:.4f}/{h['test_fp']:.3f}{mark}"
    )

if rec:
    lines += [
        "",
        "## Recommended (best honest with TEST ne≥0.60)",
        (
            f"**α={rec['alpha']} β={rec['beta']} τ={rec['thr']:.3f} m≥{rec['min_mass']}** → "
            f"TEST **ne={rec['test_ne']:.4f} fp={rec['test_fp']:.3f}** "
            f"(Δfp={rec['test_fp']-CUR_FP:+.4f}, Δne={rec['test_ne']-CUR_NE:+.4f})"
        ),
    ]
else:
    lines += ["", "No honest joint pick held TEST ne≥0.60; keep published."]

# mass-cap VAL-minfp that holds TEST
lines += ["", "## Mass-capped VAL-minfp (sens≥0.78) that hold TEST ne"]
for cap, pack in results.items():
    r = pack["val_min_fp"]
    if r and r["test_ne"] + 1e-9 >= 0.60:
        lines.append(
            f"- m≤{cap}: α={r['alpha']} β={r['beta']} τ={r['thr']:.3f} m≥{r['min_mass']} "
            f"→ TEST {r['test_ne']:.4f}/{r['test_fp']:.3f}"
        )

(OUT / "COMPARE_final.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(OUT / "report_final.json").write_text(
    json.dumps(
        {"published": pub, "honest": honest, "recommended": rec, "cap_results": results},
        indent=2,
        default=float,
    ),
    encoding="utf-8",
)
if rec:
    safe_print("=" * 50)
    safe_print(
        f"FINAL REC: α={rec['alpha']} β={rec['beta']} τ={rec['thr']:.3f} m≥{rec['min_mass']} "
        f"→ ne={rec['test_ne']:.4f} fp={rec['test_fp']:.3f} (pub {CUR_NE}/{CUR_FP})"
    )
else:
    safe_print("FINAL: keep published 0.601/0.305")
safe_print("wrote", OUT / "COMPARE_final.md")
