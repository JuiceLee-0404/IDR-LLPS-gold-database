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


CODON_TABLE: Dict[str, str] = {
    # Standard genetic code (IUPAC), DNA codons
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


def translate_dna_to_protein(dna: str) -> str:
    """Translate a DNA coding sequence to a protein sequence (single frame, stop-terminated)."""
    dna = "".join([b for b in dna.upper() if b in {"A", "C", "G", "T"}])
    aas: List[str] = []
    for i in range(0, len(dna) - 2, 3):
        codon = dna[i : i + 3]
        aa = CODON_TABLE.get(codon, "X")
        if aa == "*":
            break
        aas.append(aa)
    return "".join(aas)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute NARDINI90 features for Daiyifan SynIDP positive dataset "
            "by translating DNA sequences to protein."
        )
    )
    parser.add_argument(
        "--input-csv",
        default="ml_dl/validation/Daiyifan_SynIDP_PS/all_pos_SynIDP.csv",
        help="CSV with columns: syn_id, dna_sequence (no header).",
    )
    parser.add_argument(
        "--output-tsv",
        default="data/validation/synidp_nardini90.tsv",
        help="Output TSV with syn_id, llps_label and 90 NARDINI features.",
    )
    parser.add_argument(
        "--num-scrambles",
        type=int,
        default=30,
        help="Number of scrambles for patterning features.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base random seed for patterning scrambles.",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=1,
        help="Label value to assign (1=positive, 0=negative).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    in_path = root / args.input_csv
    out_path = root / args.output_tsv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading SynIDP DNA CSV from: {in_path}")
    df = pd.read_csv(
        in_path,
        header=None,
        names=["syn_id", "dna"],
    )
    df = df.dropna(subset=["syn_id", "dna"])
    df["syn_id"] = df["syn_id"].astype(str).str.strip()
    df["dna"] = df["dna"].astype(str).str.replace(r"\s+", "", regex=True)
    df = df[df["syn_id"] != ""].reset_index(drop=True)
    print(f"Total SynIDP entries: {len(df)}")

    # Translate DNA to protein
    print("Translating DNA to protein sequences ...")
    df["protein_seq"] = df["dna"].map(translate_dna_to_protein)
    df["protein_seq_clean"] = df["protein_seq"].fillna("").map(sanitize_sequence)
    df = df[df["protein_seq_clean"].str.len() > 0].reset_index(drop=True)
    print(f"Valid protein sequences after translation/sanitization: {len(df)}")

    # Load compositional z-score stats from training set
    stats_path = root / "data" / "processed" / "nardini90_comp_zstats.tsv"
    print(f"Loading compositional z-score stats from: {stats_path}")
    stats_df = pd.read_csv(stats_path, sep="\t")

    comp_names = compositional_feature_names()
    pat_names = patterning_feature_names()
    all_feature_names = comp_names + pat_names

    stats_df = stats_df.set_index("feature")
    means = stats_df["mean"].to_dict()
    stds = stats_df["std"].to_dict()

    # Compute compositional features (raw -> z-score using training stats)
    print("Computing compositional features for SynIDP proteins ...")
    raw_comp: List[List[float]] = [
        compositional_features(seq) for seq in df["protein_seq_clean"]
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

    # Compute patterning features and assemble final 90D vectors
    print("Computing patterning features for SynIDP proteins ...")
    rows_out: List[Dict[str, object]] = []
    for idx, (row, z_comp) in enumerate(
        zip(df.to_dict(orient="records"), z_comp_rows), start=0
    ):
        seq = row["protein_seq_clean"]
        pat = patterning_36_features(
            seq,
            num_scrambles=args.num_scrambles,
            seed=args.seed + idx,
        )
        vals = z_comp + pat
        out_row: Dict[str, object] = {
            "syn_id": row["syn_id"],
            "llps_label": int(args.label),
        }
        for name, val in zip(all_feature_names, vals):
            out_row[name] = float(val)
        rows_out.append(out_row)

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(out_path, sep="\t", index=False)
    print(
        f"Wrote NARDINI90 features for {len(out_df)} SynIDP sequences to {out_path}"
    )


if __name__ == "__main__":
    main()

