from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train RF with tuned IDR-only hyperparameters on train set and "
            "evaluate on test set, saving per-sample probabilities."
        )
    )
    parser.add_argument(
        "--train-file",
        default="data/processed/balanced_nardini90_idr_train.tsv",
        help="Balanced IDR-only training TSV.",
    )
    parser.add_argument(
        "--test-file",
        default="data/processed/balanced_nardini90_idr_test.tsv",
        help="Balanced IDR-only test TSV.",
    )
    parser.add_argument(
        "--preds-file",
        default="ml_dl/results/IDR_TRAIN/rf_idr_only_test_preds.tsv",
        help="Where to write test-set predictions (llps_label, prob_rf, pred_rf).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    train_path = root / args.train_file
    test_path = root / args.test_file
    preds_path = root / args.preds_file

    print(f"Reading train data from: {train_path}")
    train_df = pd.read_csv(train_path, sep="\t")
    print(f"Reading test data from: {test_path}")
    test_df = pd.read_csv(test_path, sep="\t")

    feature_cols = [c for c in train_df.columns if c.startswith("comp_") or c.startswith("pat_")]
    x_train = train_df[feature_cols].astype(float).values
    x_test = test_df[feature_cols].astype(float).values
    y_train = (train_df["label"] == "idr_pos").astype(int).values
    y_test = (test_df["label"] == "idr_pos").astype(int).values

    # Use the best hyperparameters found in rf_tuned_idr_only_metrics.json
    rf = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=None,
        random_state=42,
    )
    print("Fitting RF (IDR-only hyperparameters) on train set ...")
    rf.fit(x_train, y_train)
    print("Predicting on test set ...")

    y_prob = rf.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = evaluate(y_test, y_pred, y_prob)
    print("Test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    preds_out = test_df.copy()
    preds_out["llps_label"] = y_test
    preds_out["prob_rf"] = y_prob
    preds_out["pred_rf"] = y_pred
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds_out.to_csv(preds_path, sep="\t", index=False)
    print(f"Wrote test-set predictions to: {preds_path}")


if __name__ == "__main__":
    main()

