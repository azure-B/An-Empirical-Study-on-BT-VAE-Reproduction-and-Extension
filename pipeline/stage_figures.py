"""Save the COVID Dice chart and table as files."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results" / "covid" / "label_ratio" / "figures"
X = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
SERIES = [
    ("none", [0.5046, 0.6399, 0.6864, 0.7903, 0.8097], "#6b7280"),
    ("S5", [0.6627, 0.7449, 0.7925, 0.8058, 0.8185], "#d97706"),
    ("none+classification", [0.6934, 0.7396, 0.7692, 0.7903, 0.8097], "#2563eb"),
    ("S5+classification", [0.7685, 0.8011, 0.8073, 0.8299, 0.8428], "#15803d"),
]
ROWS = [
    ["0.1", "0.505", "0.663", "0.693", "0.768", "+0.075"],
    ["0.3", "0.640", "0.745", "0.740", "0.801", "+0.061"],
    ["0.5", "0.686", "0.792", "0.769", "0.807", "+0.038"],
    ["0.7", "0.790", "0.806", "0.790", "0.830", "+0.040"],
    ["0.9", "0.810", "0.818", "0.810", "0.843", "+0.033"],
]
HEADERS = ["Ratio", "none", "S5", "none+cls", "S5+cls", "S5+cls − none+cls"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    for name, ys, color in SERIES:
        ax.plot(X, ys, marker="o", linewidth=2, label=name, color=color)
    ax.set_xticks(X)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.45, 0.88)
    ax.set_xlabel("Labeled data ratio")
    ax.set_ylabel("Mean Dice on 94 test slices")
    ax.set_title("Dice score with labeled data ratio")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "dice_vs_ratio.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 2.4), dpi=160)
    ax.axis("off")
    table = ax.table(cellText=ROWS, colLabels=HEADERS, loc="center", cellLoc="right")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    fig.tight_layout()
    fig.savefig(OUT / "dice_vs_ratio_table.png", bbox_inches="tight")
    plt.close(fig)

    lines = [",".join(HEADERS)]
    lines += [",".join(row) for row in ROWS]
    (OUT / "dice_vs_ratio.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
