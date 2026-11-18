"""Entry point. Run from the repository root.

  python -m pipeline segment --modes none,s5
  python -m pipeline classify
  python -m pipeline panel
  python -m pipeline figures
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pipeline")
    sub = parser.add_subparsers(dest="stage", required=True)

    seg = sub.add_parser("segment", help="Train BT-VAE and/or BT-VAE+S on COVID.")
    seg.add_argument("--modes", default="BT-VAE,BT-VAE+S", help="BT-VAE, BT-VAE+S (aliases: none,s5)")
    seg.add_argument("--ratios", default="0.1,0.3,0.5,0.7,0.9")
    seg.add_argument("--max-ep", type=int, default=400)

    cls = sub.add_parser("classify", help="Train BT-VAE+C lesion-presence head.")
    cls.add_argument("--ratios", default="", help="COVID ratios. Empty runs the BP panel, then COVID 0.1 and 0.5.")
    cls.add_argument("--panel", action="store_true", help="Redraw the BP panel only.")

    sub.add_parser("panel", help="Redraw results/bp/panel.png.")
    sub.add_parser("figures", help="Write the Dice chart and table under results/.")

    cmp = sub.add_parser(
        "compare",
        help="Bar chart: VAE | BT-VAE | BT-VAE+C | BT-VAE+S | BT-VAE+S&C(Ours)",
    )
    cmp.add_argument("--ratio", type=float, default=0.3)
    cmp.add_argument("--batch-size", type=int, default=4)
    cmp.add_argument("--plot-only", action="store_true")

    args, rest = parser.parse_known_args(argv)
    if args.stage == "segment":
        from pipeline.stage_segment import main as stage_main
        sys.argv = [
            "stage_segment",
            "--modes", args.modes,
            "--ratios", args.ratios,
            "--max-ep", str(args.max_ep),
            *rest,
        ]
        stage_main()
        return
    if args.stage == "classify":
        from pipeline.stage_classify import main as stage_main
        sys.argv = ["stage_classify", *rest]
        if args.panel:
            sys.argv.append("--panel")
        if args.ratios:
            sys.argv.extend(["--ratios", args.ratios])
        stage_main()
        return
    if args.stage == "panel":
        from pipeline.stage_classify import main as stage_main
        sys.argv = ["stage_classify", "--panel"]
        stage_main()
        return
    if args.stage == "figures":
        from pipeline.stage_figures import main as stage_main
        stage_main()
        return
    if args.stage == "compare":
        from pipeline.stage_compare_bar import main as stage_main
        argv = ["--ratio", str(args.ratio), "--batch-size", str(args.batch_size)]
        if args.plot_only:
            argv.append("--plot-only")
        stage_main(argv)
        return
    parser.error(args.stage)


if __name__ == "__main__":
    main()
