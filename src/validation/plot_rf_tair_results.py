from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot RF performance on TAIR validation set (confusion, ROC, PR)."
    )
    parser.add_argument(
        "--preds-file",
        default="ml_dl/validation/rf_tair_validation_preds.tsv",
        help="TSV with columns: tair_id, llps_label, prob_rf, pred_rf.",
    )
    parser.add_argument(
        "--metrics-file",
        default="ml_dl/validation/rf_tair_validation_metrics.json",
        help="JSON file with summary metrics (roc_auc, etc.).",
    )
    parser.add_argument(
        "--out-dir",
        default="ml_dl/validation",
        help="Directory to write plots into.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    preds_path = root / args.preds_file
    metrics_path = root / args.metrics_file
    fig_dir = root / args.out_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(preds_path, sep="\t")
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    y_true = preds["llps_label"].values
    y_prob = preds["prob_rf"].values
    y_pred = preds["pred_rf"].values

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["neg", "pos"])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - RF (TAIR validation)")
    fig.tight_layout()
    fig.savefig(fig_dir / "cm_rf_tair.png", dpi=220)
    fig.savefig(fig_dir / "cm_rf_tair.svg")
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"RF (AUROC={metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - RF on TAIR validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_rf_tair.png", dpi=220)
    fig.savefig(fig_dir / "roc_rf_tair.svg")
    plt.close(fig)

    # PR curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, label=f"RF (AP={pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve - RF on TAIR validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "pr_rf_tair.png", dpi=220)
    fig.savefig(fig_dir / "pr_rf_tair.svg")
    plt.close(fig)

    print(f"Wrote RF TAIR validation plots to: {fig_dir}")


if __name__ == "__main__":
    main()

