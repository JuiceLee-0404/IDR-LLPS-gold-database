from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FASTA = ROOT / "data/validation/uniprotkb_proteome_UP000006548_2026_02_26.fasta"
OUT = ROOT / "data/validation/tair_sequences.tsv"


def iter_fasta(path: Path):
    header = None
    seq = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq)
                header = line
                seq = []
            else:
                seq.append(line)
        if header is not None:
            yield header, "".join(seq)


def main() -> None:
    rows = []
    for header, seq in iter_fasta(FASTA):
        h = header[1:]  # drop '>'
        tair_id = None

        # 1) Try to find ATxGxxxxx(.x) locus ID
        m = re.search(r"(AT[1-5MC]G\d{5}(?:\.\d+)?)", h, re.IGNORECASE)
        if m:
            tair_id = m.group(1).upper()
            # strip version: AT1G01030.1 -> AT1G01030
            tair_id = re.sub(r"\.\d+$", "", tair_id)

        # 2) Fallback: from UniProt GN= field if it looks like a TAIR ID
        if tair_id is None:
            m = re.search(r"GN=([A-Za-z0-9_\.]+)", h)
            if m:
                gene = m.group(1)
                if gene.upper().startswith("AT") and "G" in gene.upper():
                    tair_id = re.sub(r"\.\d+$", "", gene.upper())

        if tair_id is None:
            continue

        rows.append({"tair_id": tair_id, "sequence": seq})

    df = pd.DataFrame(rows).drop_duplicates(subset=["tair_id"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, sep="\t", index=False, encoding="utf-8")
    print(f"Wrote {len(df)} TAIR sequences to {OUT}")


if __name__ == "__main__":
    main()

