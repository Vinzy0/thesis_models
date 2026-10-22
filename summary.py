"""Collect results/*/metrics.json into summary.csv, summary.md, summary_table.png
and summary_by_angle.csv.

Usage: python summary.py [--out <dir containing results/>]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import ANGLES, MODES, ROOT, SYMPTOMS

COLS = ["Symptom",
        "vs Control: Acc", "vs Control: Bal Acc", "vs Control: F1", "vs Control: AUC",
        "vs Everything: Acc", "vs Everything: Bal Acc", "vs Everything: F1", "vs Everything: AUC",
        "Clip Acc (vs Everything)"]


def _get(m, mode, level, key):
    block = m.get(mode, {})
    if block.get("skipped"):
        return None
    return block.get(level, {}).get(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT))
    args = ap.parse_args()
    res = Path(args.out) / "results"

    rows, angle_rows = [], []
    for s in SYMPTOMS:
        p = res / s / "metrics.json"
        if not p.exists():
            print(f"missing {p}, skipping {s}")
            continue
        m = json.loads(p.read_text())
        rows.append([s.replace("_", " ").title(),
                     _get(m, "vs_control", "window", "accuracy"),
                     _get(m, "vs_control", "window", "balanced_accuracy"),
                     _get(m, "vs_control", "window", "f1"),
                     _get(m, "vs_control", "window", "roc_auc"),
                     _get(m, "vs_everything", "window", "accuracy"),
                     _get(m, "vs_everything", "window", "balanced_accuracy"),
                     _get(m, "vs_everything", "window", "f1"),
                     _get(m, "vs_everything", "window", "roc_auc"),
                     _get(m, "vs_everything", "clip", "accuracy")])
        ar = {"Symptom": s}
        for mode in MODES:
            pa = m.get(mode, {}).get("per_angle", {}) or {}
            for a in ANGLES:
                ar[f"{mode} {a} F1"] = (pa.get(a) or {}).get("f1")
        angle_rows.append(ar)

    df = pd.DataFrame(rows, columns=COLS)
    df.to_csv(res / "summary.csv", index=False, float_format="%.3f")
    pd.DataFrame(angle_rows).to_csv(res / "summary_by_angle.csv", index=False, float_format="%.3f")

    fmt = df.copy()
    for c in COLS[1:]:
        fmt[c] = fmt[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.3f}")
    header = "| " + " | ".join(COLS) + " |"
    sep = "|" + "|".join(["---"] + ["---:"] * (len(COLS) - 1)) + "|"
    lines = [header, sep] + ["| " + " | ".join(r) + " |" for r in fmt.astype(str).values.tolist()]
    md = ("# Per-symptom detection results\n\n"
          "4 s windows (2 s step), 5-fold GroupKFold by participant (test participants unseen), "
          "threshold 0.5. Window-level metrics unless noted; Clip Acc averages window "
          "probabilities per clip.\n\n" + "\n".join(lines) + "\n")
    (res / "summary.md").write_text(md, encoding="utf-8")

    # screenshot-ready table
    header_colors = ["#e8e8f0"] + ["#cfe8fc"] * 4 + ["#fde2cf"] * 4 + ["#d9f2d9"]
    labels = ["Symptom", "Acc", "Bal Acc", "F1", "AUC", "Acc", "Bal Acc", "F1", "AUC", "Clip Acc"]
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(fmt) + 1.0))
    ax.axis("off")
    tbl = ax.table(cellText=fmt.values, colLabels=labels, loc="center", cellLoc="center",
                   colColours=header_colors)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_width(0.19 if c == 0 else 0.09)
        cell.set_edgecolor("#bbbbbb")
        if r == 0:
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#fafafa")
        if c == 0 and r > 0:
            cell.set_text_props(ha="left")
            cell._loc = "left"
    # group headers above the table
    fig.canvas.draw()
    bb = {c: tbl[0, c].get_window_extent().transformed(ax.transAxes.inverted()) for c in range(10)}
    top = bb[0].y1 + 0.01
    for (c0, c1), text, col in [((1, 4), "vs Control", "#cfe8fc"),
                                ((5, 8), "vs Everything", "#fde2cf"),
                                ((9, 9), "Clip level", "#d9f2d9")]:
        x = (bb[c0].x0 + bb[c1].x1) / 2
        ax.text(x, top, text, ha="center", va="bottom", fontsize=11, weight="bold",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.25", fc=col, ec="#bbbbbb"))
    fig.text(0.5, 0.02, "4 s windows, 5-fold CV, tested on unseen participants",
             ha="center", fontsize=9, color="dimgray")
    fig.savefig(res / "summary_table.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(md)
    print(f"wrote {res / 'summary.csv'}, summary.md, summary_table.png, summary_by_angle.csv")


if __name__ == "__main__":
    main()
