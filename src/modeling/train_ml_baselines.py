from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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
    parser = argparse.ArgumentParser(description="Train ML baselines on balanced NARDINI90 features.")
    parser.add_argument("--train-file", default="data/processed/balanced_nardini90_train.tsv")
    parser.add_argument("--test-file", default="data/processed/balanced_nardini90_test.tsv")
    parser.add_argument("--metrics-file", default="reports/ml_baselines_metrics.json")
    parser.add_argument("--preds-file", default="reports/ml_baselines_preds.tsv")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_file, sep="\t")
    test_df = pd.read_csv(args.test_file, sep="\t")

    feature_cols = [c for c in train_df.columns if c.startswith("comp_") or c.startswith("pat_")]
    x_train = train_df[feature_cols].astype(float)
    x_test = test_df[feature_cols].astype(float)
    y_train = (train_df["label"] == "idr_pos").astype(int)
    y_test = (test_df["label"] == "idr_pos").astype(int)

    scaler = ColumnTransformer([("num", StandardScaler(), feature_cols)], remainder="drop")

    models = {
        "logistic_regression": Pipeline(
            [
                ("scaler", scaler),
                ("clf", LogisticRegression(max_iter=3000, random_state=42)),
            ]
        ),
        "svm_linear": Pipeline(
            [
                ("scaler", scaler),
                ("clf", SVC(probability=True, kernel="linear", random_state=42)),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("scaler", scaler),
                ("clf", SVC(probability=True, kernel="rbf", random_state=42)),
            ]
        ),
        "knn": Pipeline(
            [
                ("scaler", scaler),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight=None,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    results = {}
    prob_store = {}
    pred_store = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(x_test)[:, 1]
        else:
            # Not expected here, keep fallback.
            y_prob = model.decision_function(x_test)
        y_pred = (y_prob >= 0.5).astype(int)
        results[name] = evaluate(y_test, y_pred, y_prob)
        prob_store[name] = y_prob
        pred_store[name] = y_pred

    out = Path(args.metrics_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote metrics: {out}")

    # save per-sample predictions
    pred_rows = {
        "sample_id": test_df["sample_id"],
        "label": y_test,
    }
    for name in models:
        pred_rows[f"prob_{name}"] = prob_store[name]
        pred_rows[f"pred_{name}"] = pred_store[name]

    import pandas as _pd  # local import to avoid confusion

    _pd.DataFrame(pred_rows).to_csv(args.preds_file, sep="\t", index=False)


if __name__ == "__main__":
    main()
