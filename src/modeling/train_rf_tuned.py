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
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV


def evaluate(y_true, y_pred, y_prob):
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
        description="Tune RandomForest on balanced NARDINI90 features (absolute pos/neg)."
    )
    parser.add_argument("--train-file", default="data/processed/balanced_nardini90_train.tsv")
    parser.add_argument("--test-file", default="data/processed/balanced_nardini90_test.tsv")
    parser.add_argument("--metrics-file", default="reports/rf_tuned_metrics.json")
    parser.add_argument(
        "--preds-file",
        default="",
        help="Optional TSV to write test-set predictions (llps_label, prob_rf, pred_rf).",
    )
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_file, sep="\t")
    test_df = pd.read_csv(args.test_file, sep="\t")

    feature_cols = [c for c in train_df.columns if c.startswith("comp_") or c.startswith("pat_")]
    x_train = train_df[feature_cols].astype(float).values
    x_test = test_df[feature_cols].astype(float).values
    y_train = (train_df["label"] == "idr_pos").astype(int).values
    y_test = (test_df["label"] == "idr_pos").astype(int).values

    base_rf = RandomForestClassifier(random_state=42)
    param_dist = {
        "n_estimators": [200, 400, 800, 1200],
        "max_depth": [None, 4, 6, 8, 10],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
        "class_weight": [None, "balanced"],
    }

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=40,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search.fit(x_train, y_train)
    best_rf = search.best_estimator_

    y_prob = best_rf.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    test_metrics = evaluate(y_test, y_pred, y_prob)

    # Optionally write per-sample test predictions for downstream analysis
    if args.preds_file:
        preds_out = test_df.copy()
        preds_out["llps_label"] = y_test
        preds_out["prob_rf"] = y_prob
        preds_out["pred_rf"] = y_pred
        preds_path = Path(args.preds_file)
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        preds_out.to_csv(preds_path, sep="\t", index=False)
        print(f"Wrote test-set predictions to: {preds_path}")

    out = Path(args.metrics_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cv_best_params": search.best_params_,
        "cv_best_roc_auc": float(search.best_score_),
        "test_metrics": test_metrics,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote metrics: {out}")


if __name__ == "__main__":
    main()

