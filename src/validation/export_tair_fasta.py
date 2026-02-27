from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    """
    Export TAIR validation sequences to a FASTA file for IDR prediction tools.

    Input:  data/validation/tair_sequences.tsv  (tair_id, sequence[, ...])
    Output: data/validation/tair_sequences.fasta  (FASTA with >tair_id headers)
    """
    root = Path(__file__).resolve().parents[2]
    tsv_path = root / "data" / "validation" / "tair_sequences.tsv"
    fasta_path = root / "data" / "validation" / "tair_sequences.fasta"

    print(f"Reading TAIR sequences from: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")
    if "tair_id" not in df.columns or "sequence" not in df.columns:
        raise ValueError("Expected columns 'tair_id' and 'sequence' in tair_sequences.tsv")

    n = 0
    with fasta_path.open("w", encoding="utf-8") as out:
        for _, row in df.iterrows():
            tid = str(row["tair_id"]).strip()
            seq = str(row["sequence"] or "").strip().upper()
            if not tid or not seq:
                continue
            n += 1
            out.write(f">{tid}\n")
            # wrap sequence to 60 chars per line
            for i in range(0, len(seq), 60):
                out.write(seq[i : i + 60] + "\n")

    print(f"Wrote {n} TAIR sequences to FASTA: {fasta_path}")


if __name__ == "__main__":
    main()

