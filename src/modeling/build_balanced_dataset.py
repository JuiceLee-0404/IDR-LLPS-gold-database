from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced pos/neg dataset from nardini90 features.")
    parser.add_argument("--sqlite-file", default="data/processed/idr_llps.sqlite")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.sqlite_file)
    try:
        # Only keep absolute positives (idr_pos, experimental)
        # and absolute negatives (neg, exp_neg); drop pseudo negatives.
        query = """
        SELECT
          s.sample_id,
          s.label,
          s.label_source,
          n.*
        FROM samples s
        JOIN nardini90_features n
          ON s.sample_id = n.sample_id
        WHERE (s.label = 'idr_pos' AND s.label_source = 'experimental')
           OR (s.label = 'neg' AND s.label_source = 'exp_neg')
        """
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    df = df.loc[:, ~df.columns.duplicated()]

    pos = df[df["label"] == "idr_pos"].copy()
    neg = df[df["label"] == "neg"].copy()
    n = min(len(pos), len(neg))
    if n == 0:
        raise RuntimeError("No positive or negative samples available.")

    pos_bal = pos.sample(n=n, random_state=args.seed)
    neg_bal = neg.sample(n=n, random_state=args.seed)
    bal = pd.concat([pos_bal, neg_bal], axis=0).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    train_df, test_df = train_test_split(
        bal,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=bal["label"],
    )

    bal.to_csv(out_dir / "balanced_nardini90_full.tsv", sep="\t", index=False)
    train_df.to_csv(out_dir / "balanced_nardini90_train.tsv", sep="\t", index=False)
    test_df.to_csv(out_dir / "balanced_nardini90_test.tsv", sep="\t", index=False)

    print(f"Balanced dataset saved: {len(bal)} samples ({n} pos + {n} neg)")
    print(f"Train/Test: {len(train_df)} / {len(test_df)}")


if __name__ == "__main__":
    main()
