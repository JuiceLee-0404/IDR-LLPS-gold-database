from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export NARDINI90 feature importances from tuned RandomForest."
    )
    parser.add_argument("--train-file", default="data/processed/balanced_nardini90_train.tsv")
    parser.add_argument(
        "--output-tsv", default="reports/feature_importances_nardini90_rf.tsv"
    )
    parser.add_argument(
        "--output-md", default="reports/feature_importances_nardini90_rf.md"
    )
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_file, sep="\t")
    feature_cols = [c for c in train_df.columns if c.startswith("comp_") or c.startswith("pat_")]
    x_train = train_df[feature_cols].astype(float).values
    y_train = (train_df["label"] == "idr_pos").astype(int).values

    # Use best RF hyperparameters found in rf_tuned_metrics.json
    rf = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=None,
        random_state=42,
    )
    rf.fit(x_train, y_train)
    importances = rf.feature_importances_

    rows = []
    total = float(importances.sum()) if importances.sum() > 0 else 1.0
    for name, imp in zip(feature_cols, importances):
        rows.append(
            {
                "feature": name,
                "rf_importance": float(imp),
                "rf_importance_norm": float(imp / total),
            }
        )
    out_df = pd.DataFrame(rows).sort_values("rf_importance", ascending=False)

    out_tsv = Path(args.output_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"Wrote {out_tsv} ({len(out_df)} features)")

    top_k = min(30, len(out_df))
    out_md = Path(args.output_md)
    out_md.write_text(
        "# NARDINI90 feature importances (tuned RandomForest)\n\n"
        "Importances are Gini-based feature_importances_ from the tuned RandomForest\n"
        "trained on the balanced absolute pos/neg dataset.\n\n"
        "| Rank | Feature | Importance | Normalized |\n"
        "|------|---------|------------|-----------|\n"
        + "\n".join(
            f"| {rank} | {r.feature} | {r.rf_importance:.5f} | {r.rf_importance_norm:.5f} |"
            for rank, r in enumerate(out_df.itertuples(), 1)
            if rank <= top_k
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()

