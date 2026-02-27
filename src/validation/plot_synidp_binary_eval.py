from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RF on SynIDP all-positive set at threshold=0.5 and "
            "visualize binary results (TP/FN counts + score distribution)."
        )
    )
    parser.add_argument(
        "--preds-file",
        default="ml_dl/validation/Daiyifan_SynIDP_PS/rf_synidp_preds.tsv",
        help="SynIDP preds TSV with columns: syn_id, prob_rf, pred_rf (and optional llps_label).",
    )
    parser.add_argument(
        "--out-dir",
        default="ml_dl/validation/Daiyifan_SynIDP_PS",
        help="Directory to write SynIDP evaluation plots.",
    )
    parser.add_argument(
        "--label-type",
        choices=["pos", "neg"],
        default="pos",
        help="Whether this set is all-positive ('pos') or all-negative ('neg').",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    preds_path = root / args.preds_file
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading SynIDP predictions from: {preds_path}")
    df = pd.read_csv(preds_path, sep="\t")
    if "prob_rf" not in df.columns or "pred_rf" not in df.columns:
        raise ValueError("Preds TSV must contain 'prob_rf' and 'pred_rf' columns.")

    n = len(df)
    if args.label_type == "pos":
        y_true = np.ones(n, dtype=int)
    else:
        y_true = np.zeros(n, dtype=int)
    y_pred = df["pred_rf"].astype(int).values
    y_prob = df["prob_rf"].astype(float).values

    if args.label_type == "pos":
        tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
        fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
        recall = tp / n if n > 0 else 0.0
        print(f"Total SynIDP samples: {n}, TP={tp}, FN={fn}, recall@0.5={recall:.3f}")
        bar_labels = ["TP (pred=1)", "FN (pred=0)"]
        bar_values = [tp, fn]
        bar_colors = ["#2a9d8f", "#e76f51"]
        bar_title = f"SynIDP @ threshold=0.5\nrecall = {recall:.3f} ({tp}/{n})"
    else:
        tn = int(np.logical_and(y_true == 0, y_pred == 0).sum())
        fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
        spec = tn / n if n > 0 else 0.0
        print(f"Total SynIDP samples: {n}, TN={tn}, FP={fp}, specificity@0.5={spec:.3f}")
        bar_labels = ["TN (pred=0)", "FP (pred=1)"]
        bar_values = [tn, fp]
        bar_colors = ["#2a9d8f", "#e76f51"]
        bar_title = f"SynIDP @ threshold=0.5\nspecificity = {spec:.3f} ({tn}/{n})"

    # 1) Bar plot: TP/FN or TN/FP counts
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(bar_labels, bar_values, color=bar_colors)
    ax.set_ylabel("Count")
    ax.set_title(bar_title)
    for i, v in enumerate(bar_values):
        ax.text(i, v + 0.05, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "synidp_tp_fn_bar.png", dpi=220)
    fig.savefig(out_dir / "synidp_tp_fn_bar.svg")
    plt.close(fig)

    # 2) Score histogram, colored by predicted label
    fig, ax = plt.subplots(figsize=(5, 4))
    bins = np.linspace(0.0, 1.0, 21)
    ax.hist(
        y_prob[y_pred == 1],
        bins=bins,
        alpha=0.7,
        label="pred=1",
        color="#2a9d8f",
        edgecolor="black",
    )
    if (y_pred == 0).any():
        ax.hist(
            y_prob[y_pred == 0],
            bins=bins,
            alpha=0.7,
            label="pred=0",
            color="#e76f51",
            edgecolor="black",
        )
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="threshold=0.5")
    ax.set_xlabel("RF score (prob_rf)")
    ax.set_ylabel("Count")
    ax.set_title("SynIDP score distribution @ threshold=0.5")
    ax.set_xlim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "synidp_score_hist.png", dpi=220)
    fig.savefig(out_dir / "synidp_score_hist.svg")
    plt.close(fig)

    print(f"Wrote SynIDP binary evaluation plots to: {out_dir}")


if __name__ == "__main__":
    main()

