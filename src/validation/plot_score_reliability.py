from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve


def plot_reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_path: Path,
    n_bins: int = 10,
) -> None:
    """Plot probability vs. observed positive fraction (calibration / reliability)."""
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax.plot(prob_pred, prob_true, "o-", label="model (binned)")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title("Reliability / Calibration Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=220)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def plot_precision_recall_vs_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_path: Path,
) -> None:
    """Plot precision and recall as functions of the decision threshold."""
    thresholds = np.linspace(0.0, 1.0, 101)
    precisions = []
    recalls = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = np.logical_and(y_pred == 1, y_true == 1).sum()
        fp = np.logical_and(y_pred == 1, y_true == 0).sum()
        fn = np.logical_and(y_pred == 0, y_true == 1).sum()

        if tp + fp == 0:
            prec = np.nan
        else:
            prec = tp / (tp + fp)
        if tp + fn == 0:
            rec = np.nan
        else:
            rec = tp / (tp + fn)
        precisions.append(prec)
        recalls.append(rec)

    precisions = np.array(precisions)
    recalls = np.array(recalls)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, recalls, label="Recall", color="#1b9e77")
    ax.plot(thresholds, precisions, label="Precision", color="#d95f02")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall vs. Threshold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=220)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize relationship between RF scores and binary "
            "classification (calibration + precision/recall vs threshold)."
        )
    )
    parser.add_argument(
        "--preds-file",
        required=True,
        help="TSV with at least columns: llps_label, prob_rf.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write plots into.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    preds_path = root / args.preds_file
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading predictions from: {preds_path}")
    df = pd.read_csv(preds_path, sep="\t")
    if "llps_label" not in df.columns or "prob_rf" not in df.columns:
        raise ValueError("Preds TSV must contain 'llps_label' and 'prob_rf' columns.")

    y_true = df["llps_label"].astype(int).values
    y_prob = df["prob_rf"].astype(float).values

    # Reliability / calibration curve
    rel_path = out_dir / "score_reliability_curve"
    plot_reliability_curve(y_true, y_prob, rel_path)

    # Precision / recall vs threshold
    prt_path = out_dir / "precision_recall_vs_threshold"
    plot_precision_recall_vs_threshold(y_true, y_prob, prt_path)

    print(f"Wrote score reliability plots to: {out_dir}")


if __name__ == "__main__":
    main()

