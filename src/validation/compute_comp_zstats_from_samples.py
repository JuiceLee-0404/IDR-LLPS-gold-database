from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.export.compute_nardini90_features import (
    compositional_feature_names,
    compositional_features,
    sanitize_sequence,
)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    samples_path = root / "data/processed/samples.tsv"
    out_path = root / "data/processed/nardini90_comp_zstats.tsv"

    df = pd.read_csv(samples_path, sep="\t")
    df = df[df["label"].isin(["idr_pos", "neg"])].copy()
    df["region_sequence"] = df["region_sequence"].fillna("").map(sanitize_sequence)
    df = df[df["region_sequence"].str.len() > 0].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No valid IDR sequences found in samples.tsv")

    comp_names = compositional_feature_names()
    rows = []
    for seq in df["region_sequence"]:
        rows.append(compositional_features(seq))

    comp_df = pd.DataFrame(rows, columns=comp_names)
    means = comp_df.mean(axis=0)
    stds = comp_df.std(axis=0, ddof=0)  # population std to match zscore_matrix

    stats_df = pd.DataFrame(
        {
            "feature": comp_names,
            "mean": [float(means[f]) for f in comp_names],
            "std": [float(stds[f]) for f in comp_names],
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote compositional z-score stats for {len(comp_names)} features to {out_path}")


if __name__ == "__main__":
    main()

