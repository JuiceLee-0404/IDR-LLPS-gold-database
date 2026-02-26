from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    parser = argparse.ArgumentParser(description="Train MLP baseline on balanced NARDINI90 features.")
    parser.add_argument("--train-file", default="data/processed/balanced_nardini90_train.tsv")
    parser.add_argument("--test-file", default="data/processed/balanced_nardini90_test.tsv")
    parser.add_argument("--metrics-file", default="reports/deep_baseline_metrics.json")
    parser.add_argument("--preds-file", default="reports/deep_baseline_preds.tsv")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_file, sep="\t")
    test_df = pd.read_csv(args.test_file, sep="\t")

    feature_cols = [c for c in train_df.columns if c.startswith("comp_") or c.startswith("pat_")]
    x_train = train_df[feature_cols].astype(float)
    x_test = test_df[feature_cols].astype(float)
    y_train = (train_df["label"] == "idr_pos").astype(int)
    y_test = (test_df["label"] == "idr_pos").astype(int)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=600,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.15,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = evaluate(y_test, y_pred, y_prob)

    out = Path(args.metrics_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote metrics: {out}")

    # save per-sample predictions
    pred_rows = {
        "sample_id": test_df["sample_id"],
        "label": y_test,
        "prob_mlp": y_prob,
        "pred_mlp": y_pred,
    }
    import pandas as _pd  # local import

    _pd.DataFrame(pred_rows).to_csv(args.preds_file, sep="\t", index=False)


if __name__ == "__main__":
    main()
