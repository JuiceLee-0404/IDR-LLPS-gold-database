from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate tuned RandomForest on TAIR NARDINI90 validation set."
    )
    parser.add_argument(
        "--train-file",
        default="data/processed/balanced_nardini90_train.tsv",
        help="Balanced absolute pos/neg training TSV used for RF.",
    )
    parser.add_argument(
        "--tair-file",
        default="data/validation/tair_nardini90_labeled.tsv",
        help="TAIR validation TSV with llps_label and NARDINI90 features.",
    )
    parser.add_argument(
        "--metrics-file",
        default="ml_dl/validation/rf_tair_validation_metrics.json",
        help="Where to write JSON metrics for TAIR validation (under ml_dl/validation).",
    )
    parser.add_argument(
        "--preds-file",
        default="ml_dl/validation/rf_tair_validation_preds.tsv",
        help="Where to write per-sample predictions on TAIR validation (under ml_dl/validation).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    train_path = root / args.train_file
    tair_path = root / args.tair_file
    metrics_path = root / args.metrics_file
    preds_path = root / args.preds_file
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading RF training data from: {train_path}")
    train_df = pd.read_csv(train_path, sep="\t")
    print(f"Reading TAIR feature table from: {tair_path}")
    tair_df = pd.read_csv(tair_path, sep="\t")

    feature_cols = [c for c in train_df.columns if c.startswith("comp_") or c.startswith("pat_")]
    if not feature_cols:
        raise ValueError("No comp_/pat_ feature columns found in training file.")
    for col in feature_cols:
        if col not in tair_df.columns:
            raise ValueError(f"TAIR validation file is missing feature column '{col}'.")

    print(f"Using {len(feature_cols)} NARDINI90 features for RF.")
    x_train = train_df[feature_cols].astype(float).values
    y_train = (train_df["label"] == "idr_pos").astype(int).values

    x_tair = tair_df[feature_cols].astype(float).values
    if "llps_label" not in tair_df.columns:
        raise ValueError("TAIR validation TSV must contain 'llps_label' column (1=LLPS, 0=non-LLPS).")
    y_tair = tair_df["llps_label"].astype(int).values

    print("Training RandomForest on balanced NARDINI90 training set ...")
    rf = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=None,
        random_state=42,
    )
    rf.fit(x_train, y_train)
    print("Finished RF training; predicting on TAIR validation set ...")

    y_prob = rf.predict_proba(x_tair)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = evaluate(y_tair, y_prob, y_pred)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote TAIR RF validation metrics: {metrics_path}")

    preds_out = pd.DataFrame(
        {
            "tair_id": tair_df["tair_id"],
            "llps_label": y_tair,
            "prob_rf": y_prob,
            "pred_rf": y_pred,
        }
    )
    preds_out.to_csv(preds_path, sep="\t", index=False)
    print(f"Wrote TAIR RF validation predictions: {preds_path}")


if __name__ == "__main__":
    main()

