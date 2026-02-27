from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_idr_overlap_frac(row: pd.Series) -> float:
    """
    Compute fraction of the annotated IDR that is covered by the tested region.

    All coordinates are 1-based inclusive. Returns value in [0, 1].
    """
    try:
        rs = int(row["region_start"])
        re = int(row["region_end"])
        is_ = int(row["idr_start"])
        ie = int(row["idr_end"])
    except Exception:
        return 0.0

    if rs <= 0 or re <= 0 or is_ <= 0 or ie <= 0 or re < rs or ie < is_:
        return 0.0

    idr_len = ie - is_ + 1
    if idr_len <= 0:
        return 0.0

    # Overlap between [rs, re] and [is_, ie]
    left = max(rs, is_)
    right = min(re, ie)
    if right < left:
        return 0.0

    overlap = right - left + 1
    return float(overlap / idr_len)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an IDR-only samples table where both positives and negatives "
            "are IDR constructs (region overlaps annotated IDR sufficiently)."
        )
    )
    parser.add_argument(
        "--samples-file",
        default="data/processed/samples.tsv",
        help="Input samples TSV (from main IDR-LLPS pipeline).",
    )
    parser.add_argument(
        "--output-file",
        default="data/processed/samples_idr_only.tsv",
        help="Output TSV with IDR-only samples.",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.7,
        help="Minimum fraction of IDR length that must be covered by the region.",
    )
    parser.add_argument(
        "--min-idr-len",
        type=int,
        default=15,
        help="Minimum IDR length to be considered valid.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    in_path = root / args.samples_file
    out_path = root / args.output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading samples from: {in_path}")
    df = pd.read_csv(in_path, sep="\t")

    required_cols = {
        "sample_id",
        "label",
        "label_source",
        "region_start",
        "region_end",
        "idr_start",
        "idr_end",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"samples TSV missing required columns: {missing}")

    # Coerce coordinates to numeric
    for col in ["region_start", "region_end", "idr_start", "idr_end"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute IDR length
    df["idr_length"] = (df["idr_end"] - df["idr_start"] + 1).clip(lower=0)

    # Compute overlap fraction (how much of IDR is covered by tested region)
    print("Computing region/IDR overlap fractions ...")
    df["idr_overlap_frac"] = df.apply(compute_idr_overlap_frac, axis=1)

    # Filter to entries with valid IDR and sufficient overlap
    mask_valid_idr = df["idr_length"] >= args.min_idr_len
    mask_overlap = df["idr_overlap_frac"] >= args.overlap_threshold
    idr_df = df[mask_valid_idr & mask_overlap].copy()

    # Keep only strict positives and strict experimental negatives
    # (we still filter labels after the IDR criteria, but both sides are IDR constructs).
    mask_pos = (idr_df["label"] == "idr_pos") & (
        idr_df["label_source"] == "experimental"
    )
    mask_neg = (idr_df["label"] == "neg") & (idr_df["label_source"] == "exp_neg")

    idr_df = idr_df[mask_pos | mask_neg].reset_index(drop=True)

    print(
        f"Total samples: {len(df)}, IDR-overlapping: {mask_valid_idr.sum()} "
        f"({mask_overlap.sum()} with overlap >= {args.overlap_threshold}), "
        f"IDR-only strict pos/neg: {len(idr_df)}"
    )
    print("Label counts in IDR-only set:")
    print(idr_df["label"].value_counts())

    idr_df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote IDR-only samples to: {out_path}")


if __name__ == "__main__":
    main()

