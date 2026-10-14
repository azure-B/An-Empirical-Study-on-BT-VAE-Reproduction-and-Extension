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

    seg = sub.add_parser("segment", help="Train none and/or S5 on COVID label ratios.")
    seg.add_argument("--modes", default="none,s5")
    seg.add_argument("--ratios", default="0.1,0.3,0.5,0.7,0.9")
    seg.add_argument("--max-ep", type=int, default=400)

    cls = sub.add_parser("classify", help="Train the lesion-probability head.")
    cls.add_argument("--ratios", default="", help="COVID ratios. Empty runs the BP panel, then COVID 0.1 and 0.5.")
    cls.add_argument("--panel", action="store_true", help="Redraw the BP panel only.")

    sub.add_parser("panel", help="Redraw results/bp/panel.png.")
    sub.add_parser("figures", help="Write the Dice chart and table under results/.")

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
    parser.error(args.stage)


if __name__ == "__main__":
    main()
