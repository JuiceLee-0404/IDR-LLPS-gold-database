from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def summarize_scores(y_prob: np.ndarray) -> Dict[str, float]:
    """Summarize score distribution for an all-positive validation set."""
    thresholds = [0.5, 0.7, 0.8, 0.9]
    summary: Dict[str, float] = {
        "mean_prob": float(np.mean(y_prob)),
        "median_prob": float(np.median(y_prob)),
    }
    for t in thresholds:
        frac = float(np.mean(y_prob >= t))
        summary[f"frac_prob_ge_{t}"] = frac
    qs = np.quantile(y_prob, [0.1, 0.25, 0.5, 0.75, 0.9])
    for q, v in zip(["q10", "q25", "q50", "q75", "q90"], qs):
        summary[f"prob_{q}"] = float(v)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate tuned RF (IDR-only or mixed) on Daiyifan SynIDP positive set.\n"
            "Since all labels are positive, we report score distribution and "
            "fraction of sequences above various thresholds."
        )
    )
    parser.add_argument(
        "--train-file",
        default="data/processed/balanced_nardini90_idr_train.tsv",
        help="Balanced training TSV used for RF (recommend IDR-only set).",
    )
    parser.add_argument(
        "--synidp-file",
        default="data/validation/synidp_nardini90.tsv",
        help="SynIDP TSV with syn_id, llps_label=1 and NARDINI90 features.",
    )
    parser.add_argument(
        "--metrics-file",
        default="ml_dl/validation/Daiyifan_SynIDP_PS/rf_synidp_metrics.json",
        help="Where to write JSON summary metrics for SynIDP validation.",
    )
    parser.add_argument(
        "--preds-file",
        default="ml_dl/validation/Daiyifan_SynIDP_PS/rf_synidp_preds.tsv",
        help="Where to write per-sample predictions on SynIDP validation.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    train_path = root / args.train_file
    syn_path = root / args.synidp_file
    metrics_path = root / args.metrics_file
    preds_path = root / args.preds_file
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading RF training data from: {train_path}")
    train_df = pd.read_csv(train_path, sep="\t")
    print(f"Reading SynIDP feature table from: {syn_path}")
    syn_df = pd.read_csv(syn_path, sep="\t")

    feature_cols = [c for c in train_df.columns if c.startswith("comp_") or c.startswith("pat_")]
    if not feature_cols:
        raise ValueError("No comp_/pat_ feature columns found in training file.")
    for col in feature_cols:
        if col not in syn_df.columns:
            raise ValueError(f"SynIDP file is missing feature column '{col}'.")

    print(f"Using {len(feature_cols)} NARDINI90 features for RF.")
    x_train = train_df[feature_cols].astype(float).values
    y_train = (train_df["label"] == "idr_pos").astype(int).values

    x_syn = syn_df[feature_cols].astype(float).values
    if "llps_label" in syn_df.columns:
        y_syn = syn_df["llps_label"].astype(int).values
    else:
        # All positive by construction
        y_syn = np.ones(len(syn_df), dtype=int)

    rf = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=None,
        random_state=42,
    )
    print("Training RandomForest on balanced training set ...")
    rf.fit(x_train, y_train)
    print("Finished RF training; predicting on SynIDP validation set ...")

    y_prob = rf.predict_proba(x_syn)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Save per-sample predictions
    preds_out = syn_df.copy()
    preds_out["prob_rf"] = y_prob
    preds_out["pred_rf"] = y_pred
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds_out.to_csv(preds_path, sep="\t", index=False)
    print(f"Wrote SynIDP RF predictions: {preds_path}")

    # Summarize score distribution (all-pos setting)
    summary = summarize_scores(y_prob)
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote SynIDP RF score summary: {metrics_path}")


if __name__ == "__main__":
    main()

