from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build balanced idr_pos/neg dataset from IDR-only NARDINI90 features."
    )
    parser.add_argument(
        "--features-tsv",
        default="data/processed/nardini90_features_idr_only.tsv",
        help="TSV with IDR-only NARDINI90 features (from samples_idr_only).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to write balanced IDR-only train/test TSVs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_path = Path(args.features_tsv)
    print(f"Reading IDR-only features from: {feat_path}")
    df = pd.read_csv(feat_path, sep="\t")

    if "label" not in df.columns:
        raise ValueError("IDR-only features TSV must contain 'label' column.")

    pos = df[df["label"] == "idr_pos"].copy()
    neg = df[df["label"] == "neg"].copy()
    n = min(len(pos), len(neg))
    if n == 0:
        raise RuntimeError("No positive or negative samples available for IDR-only set.")

    print(f"IDR-only pos: {len(pos)}, neg: {len(neg)}; using n={n} for each class.")
    pos_bal = pos.sample(n=n, random_state=args.seed)
    neg_bal = neg.sample(n=n, random_state=args.seed)
    bal = (
        pd.concat([pos_bal, neg_bal], axis=0)
        .sample(frac=1.0, random_state=args.seed)
        .reset_index(drop=True)
    )

    train_df, test_df = train_test_split(
        bal,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=bal["label"],
    )

    full_out = out_dir / "balanced_nardini90_idr_full.tsv"
    train_out = out_dir / "balanced_nardini90_idr_train.tsv"
    test_out = out_dir / "balanced_nardini90_idr_test.tsv"

    bal.to_csv(full_out, sep="\t", index=False)
    train_df.to_csv(train_out, sep="\t", index=False)
    test_df.to_csv(test_out, sep="\t", index=False)

    print(
        f"Balanced IDR-only dataset saved: {len(bal)} samples ({n} idr_pos + {n} neg)"
    )
    print(f"Train/Test: {len(train_df)} / {len(test_df)}")
    print(f"Wrote: {full_out}, {train_out}, {test_out}")


if __name__ == "__main__":
    main()

