"""
Export 90-dim NARDINI feature scores from Logistic Regression weights.
Reads the same balanced train TSV used by train_ml_baselines, fits LR with StandardScaler,
and writes feature name, coefficient, and |coefficient| to reports.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export NARDINI90 feature weights from Logistic Regression."
    )
    parser.add_argument("--train-file", default="data/processed/balanced_nardini90_train.tsv")
    parser.add_argument("--output-tsv", default="reports/feature_weights_nardini90.tsv")
    parser.add_argument("--output-md", default="reports/feature_scores_nardini90.md")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_file, sep="\t")
    feature_cols = [c for c in train_df.columns if c.startswith("comp_") or c.startswith("pat_")]
    x_train = train_df[feature_cols].astype(float)
    y_train = (train_df["label"] == "idr_pos").astype(int)

    scaler = ColumnTransformer([("num", StandardScaler(), feature_cols)], remainder="drop")
    pipe = Pipeline(
        [
            ("scaler", scaler),
            ("clf", LogisticRegression(max_iter=3000, random_state=42)),
        ]
    )
    pipe.fit(x_train, y_train)
    lr = pipe.named_steps["clf"]
    coef = lr.coef_.ravel()

    rows = []
    for name, c in zip(feature_cols, coef):
        rows.append({"feature": name, "coefficient": float(c), "abs_coefficient": float(abs(c))})
    out_df = pd.DataFrame(rows).sort_values("abs_coefficient", ascending=False)

    out_tsv = Path(args.output_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"Wrote {out_tsv} ({len(out_df)} features)")

    out_md = Path(args.output_md)
    out_md.write_text(
        "# NARDINI90 feature scores (Logistic Regression weights)\n\n"
        "Weights are in **scaled** feature space (StandardScaler applied before LR).\n"
        "Positive coefficient => higher value favors IDR-positive; negative => favors negative.\n\n"
        "| Rank | Feature | Coefficient | |Coefficient|\n"
        "|------|---------|-------------|-------------|\n"
        + "\n".join(
            f"| {rank} | {r.feature} | {r.coefficient:.4f} | {r.abs_coefficient:.4f} |"
            for rank, r in enumerate(out_df.itertuples(), 1)
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
