from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    results_dir = root / "ml_dl" / "results"

    ml_preds = pd.read_csv(results_dir / "ml_baselines_preds.tsv", sep="\t")
    dl_preds = pd.read_csv(results_dir / "deep_baseline_preds.tsv", sep="\t")

    y_true = ml_preds["label"].values
    models = {
        "logistic_regression": ml_preds["prob_logistic_regression"].values,
        "svm_rbf": ml_preds["prob_svm_rbf"].values,
        "random_forest": ml_preds["prob_random_forest"].values,
        "mlp_baseline": dl_preds["prob_mlp"].values,
    }

    # confusion matrices
    for name, probs in models.items():
        y_pred = (probs >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["neg", "pos"])
        fig, ax = plt.subplots(figsize=(4, 4))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion Matrix - {name}")
        fig.tight_layout()
        fig.savefig(results_dir / f"cm_{name}.png", dpi=220)
        fig.savefig(results_dir / f"cm_{name}.svg")
        plt.close(fig)

    # ROC curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, probs in models.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "roc_curves.png", dpi=220)
    fig.savefig(results_dir / "roc_curves.svg")
    plt.close(fig)

    # PR curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, probs in models.items():
        prec, rec, _ = precision_recall_curve(y_true, probs)
        pr_auc = average_precision_score(y_true, probs)
        ax.plot(rec, prec, label=f"{name} (AP={pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "pr_curves.png", dpi=220)
    fig.savefig(results_dir / "pr_curves.svg")
    plt.close(fig)

    print(f"Wrote confusion matrices and ROC/PR curves to: {results_dir}")


if __name__ == "__main__":
    main()
