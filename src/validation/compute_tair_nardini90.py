from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.export.compute_nardini90_features import (
    compositional_feature_names,
    compositional_features,
    patterning_36_features,
    patterning_feature_names,
    sanitize_sequence,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute NARDINI90 features for TAIR validation sequences."
    )
    parser.add_argument(
        "--input-tsv",
        default="data/validation/tair_sequences.tsv",
        help="TSV with columns: tair_id, sequence",
    )
    parser.add_argument(
        "--output-tsv",
        default="data/validation/tair_nardini90.tsv",
        help="Output TSV with tair_id and 90 NARDINI features.",
    )
    parser.add_argument(
        "--num-scrambles",
        type=int,
        default=30,
        help="Number of scrambles for patterning features (like training).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base random seed for patterning scrambles.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    in_path = root / args.input_tsv
    out_path = root / args.output_tsv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) 读取序列表和标签表，只保留有 llps_label 的 TAIR
    print(f"Reading sequences from: {in_path}")
    seq_df = pd.read_csv(in_path, sep="\t")
    labels_path = root / "data/validation/tair_llps_labels.tsv"
    print(f"Reading TAIR LLPS labels from: {labels_path}")
    labels_df = pd.read_csv(labels_path, sep="\t")
    labeled_ids = set(labels_df["tair_id"].astype(str))
    before_filter = len(seq_df)
    seq_df = seq_df[seq_df["tair_id"].astype(str).isin(labeled_ids)].reset_index(drop=True)
    print(
        f"Total sequences: {before_filter}, with LLPS labels: {len(seq_df)}"
    )

    if "tair_id" not in seq_df.columns or "sequence" not in seq_df.columns:
        raise ValueError("Expected columns 'tair_id' and 'sequence' in input TSV.")

    # 2) sanitize 序列
    print("Sanitizing sequences ...")
    seq_df["sequence_clean"] = seq_df["sequence"].fillna("").map(sanitize_sequence)
    seq_df = seq_df[seq_df["sequence_clean"].str.len() > 0].reset_index(drop=True)
    if seq_df.empty:
        raise RuntimeError("No valid sequences after sanitization.")

    # 3) 读取训练集 compositional 特征的 z-score 统计量（mean/std）
    stats_path = root / "data/processed/nardini90_comp_zstats.tsv"
    print(f"Loading compositional z-score stats from: {stats_path}")
    stats_df = pd.read_csv(stats_path, sep="\t")
    comp_names = compositional_feature_names()
    pat_names = patterning_feature_names()
    all_feature_names = comp_names + pat_names

    stats_df = stats_df.set_index("feature")
    means = stats_df["mean"].to_dict()
    stds = stats_df["std"].to_dict()

    # 4) 计算 raw compositional 特征，并使用训练分布的 mean/std 做 z-score
    print(f"Computing compositional features for {len(seq_df)} sequences ...")
    raw_comp: List[List[float]] = [
        compositional_features(seq) for seq in seq_df["sequence_clean"]
    ]

    z_comp_rows: List[List[float]] = []
    for row_vals in raw_comp:
        z_row: List[float] = []
        for feat, val in zip(comp_names, row_vals):
            mu = means.get(feat, 0.0)
            sd = stds.get(feat, 1.0)
            if sd <= 0:
                z_val = 0.0
            else:
                z_val = (val - mu) / sd
            z_row.append(float(z_val))
        z_comp_rows.append(z_row)

    # 5) 计算 patterning 特征，并拼接成最终 90 维向量
    rows_out: List[Dict[str, object]] = []
    print("Computing patterning features and assembling NARDINI90 matrix ...")
    for idx, (tair_id, seq, z_comp) in enumerate(
        zip(seq_df["tair_id"], seq_df["sequence_clean"], z_comp_rows), start=0
        ):
        pat = patterning_36_features(
            seq,
            num_scrambles=args.num_scrambles,
            seed=args.seed + idx,
        )
        vals = z_comp + pat
        row: Dict[str, object] = {"tair_id": tair_id}
        for name, val in zip(all_feature_names, vals):
            row[name] = float(val)
        rows_out.append(row)
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1} / {len(seq_df)} sequences")

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote NARDINI90 features for {len(out_df)} TAIR sequences to {out_path}")


if __name__ == "__main__":
    main()

