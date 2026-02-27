from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

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


def evaluate(
    y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
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
        description=(
            "Evaluate tuned RandomForest on TAIR IDR segments, "
            "aggregating to protein-level by max IDR probability."
        )
    )
    parser.add_argument(
        "--train-file",
        default="data/processed/balanced_nardini90_train.tsv",
        help="Balanced absolute pos/neg training TSV used for RF.",
    )
    parser.add_argument(
        "--tair-idr-file",
        default="data/validation/tair_idr_nardini90_labeled.tsv",
        help=(
            "Per-IDR TAIR validation TSV with llps_label and NARDINI90 features "
            "(one row per IDR segment)."
        ),
    )
    parser.add_argument(
        "--metrics-file",
        default="ml_dl/validation/IDR/rf_tair_idr_max_validation_metrics.json",
        help="Where to write JSON metrics for protein-level TAIR validation.",
    )
    parser.add_argument(
        "--segment-preds-file",
        default="ml_dl/validation/IDR/rf_tair_idr_segment_preds.tsv",
        help="Where to write per-IDR segment predictions.",
    )
    parser.add_argument(
        "--protein-preds-file",
        default="ml_dl/validation/IDR/rf_tair_idr_max_validation_preds.tsv",
        help="Where to write protein-level max-IDR predictions.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold on max IDR probability to call a protein positive.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    train_path = root / args.train_file
    tair_idr_path = root / args.tair_idr_file
    metrics_path = root / args.metrics_file
    seg_preds_path = root / args.segment_preds_file
    prot_preds_path = root / args.protein_preds_file
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    seg_preds_path.parent.mkdir(parents=True, exist_ok=True)
    prot_preds_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading RF training data from: {train_path}")
    train_df = pd.read_csv(train_path, sep="\t")
    print(f"Reading TAIR per-IDR feature table from: {tair_idr_path}")
    tair_df = pd.read_csv(tair_idr_path, sep="\t")

    feature_cols = [
        c for c in train_df.columns if c.startswith("comp_") or c.startswith("pat_")
    ]
    if not feature_cols:
        raise ValueError("No comp_/pat_ feature columns found in training file.")
    for col in feature_cols:
        if col not in tair_df.columns:
            raise ValueError(f"TAIR IDR file is missing feature column '{col}'.")

    print(f"Using {len(feature_cols)} NARDINI90 features for RF.")
    x_train = train_df[feature_cols].astype(float).values
    y_train = (train_df["label"] == "idr_pos").astype(int).values

    # Per-IDR segment data
    x_idr = tair_df[feature_cols].astype(float).values
    if "llps_label" not in tair_df.columns:
        raise ValueError(
            "TAIR IDR TSV must contain 'llps_label' column (1=LLPS, 0=non-LLPS)."
        )
    y_idr = tair_df["llps_label"].astype(int).values
    tair_ids = tair_df["tair_id"].astype(str).tolist()

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
    print("Finished RF training; predicting on TAIR IDR segments ...")

    prob_seg = rf.predict_proba(x_idr)[:, 1]
    pred_seg = (prob_seg >= args.threshold).astype(int)

    # Save per-segment predictions (for inspection)
    seg_out = tair_df.copy()
    seg_out["prob_rf"] = prob_seg
    seg_out["pred_rf_segment"] = pred_seg
    seg_out.to_csv(seg_preds_path, sep="\t", index=False)
    print(f"Wrote per-IDR segment predictions to: {seg_preds_path}")

    # Aggregate to protein level: one label per tair_id
    seg_out["tair_id"] = seg_out["tair_id"].astype(str)
    grouped = seg_out.groupby("tair_id", as_index=False)

    prot_rows: List[Dict[str, object]] = []
    for tid, sub in grouped:
        labels = sub["llps_label"].unique()
        if len(labels) > 1:
            # In practice this should not happen; take the majority just in case.
            true_label = int(round(sub["llps_label"].mean()))
        else:
            true_label = int(labels[0])
        max_prob = float(sub["prob_rf"].max())
        # Decision rule: if any IDR has prob >= threshold, protein is positive
        pred_prot = int(max_prob >= args.threshold)
        prot_rows.append(
            {
                "tair_id": tid,
                "llps_label": true_label,
                "prob_rf_max_idr": max_prob,
                "pred_rf": pred_prot,
                "num_idr_segments": int(len(sub)),
            }
        )

    prot_df = pd.DataFrame(prot_rows)
    y_true_prot = prot_df["llps_label"].astype(int).values
    y_prob_prot = prot_df["prob_rf_max_idr"].astype(float).values
    y_pred_prot = prot_df["pred_rf"].astype(int).values

    metrics = evaluate(y_true_prot, y_prob_prot, y_pred_prot)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote TAIR RF (IDR-max) validation metrics: {metrics_path}")

    # For compatibility with plotting script, write a simplified preds file
    preds_plot = prot_df[["tair_id", "llps_label"]].copy()
    preds_plot["prob_rf"] = prot_df["prob_rf_max_idr"].astype(float)
    preds_plot["pred_rf"] = prot_df["pred_rf"].astype(int)
    preds_plot.to_csv(prot_preds_path, sep="\t", index=False)
    print(f"Wrote protein-level IDR-max predictions to: {prot_preds_path}")


if __name__ == "__main__":
    main()

