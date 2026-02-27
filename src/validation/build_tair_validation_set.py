from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join TAIR LLPS labels with NARDINI90 features into a validation set."
    )
    parser.add_argument(
        "--labels-tsv",
        default="data/validation/tair_llps_labels.tsv",
        help="Input TSV with columns including tair_id and llps_label.",
    )
    parser.add_argument(
        "--features-tsv",
        default="data/validation/tair_nardini90.tsv",
        help="Input TSV with tair_id and NARDINI90 features.",
    )
    parser.add_argument(
        "--output-tsv",
        default="data/validation/tair_nardini90_labeled.tsv",
        help="Output TSV with merged labels and features.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    labels_path = root / args.labels_tsv
    feats_path = root / args.features_tsv
    out_path = root / args.output_tsv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(labels_path, sep="\t")
    feats_df = pd.read_csv(feats_path, sep="\t")

    if "tair_id" not in labels_df.columns or "llps_label" not in labels_df.columns:
        raise ValueError("labels TSV must contain 'tair_id' and 'llps_label' columns.")
    if "tair_id" not in feats_df.columns:
        raise ValueError("features TSV must contain 'tair_id' column.")

    merged = labels_df.merge(feats_df, on="tair_id", how="inner")
    merged.to_csv(out_path, sep="\t", index=False)
    print(
        f"Merged {len(labels_df)} labels with {len(feats_df)} features "
        f"into {len(merged)} rows at {out_path}"
    )


if __name__ == "__main__":
    main()

