from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = ROOT / "data/validation/tair_llps_labels.tsv"
OUT_PATH = ROOT / "data/validation/tair_sequences.tsv"

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"


def fetch_sequence_for_tair(tair_id: str) -> tuple[str, str] | None:
    """
    Fetch sequence for a TAIR locus using UniProt REST.

    We try several strategies:
    1) direct TAIR / Araport cross-reference
    2) exact gene name match
    All restricted to Arabidopsis thaliana (taxon 3702).
    """
    gene = tair_id.upper()

    # Try TAIR / Araport cross-references first
    queries = [
        f"(xref:TAIR:{gene} OR xref:Araport:{gene}) AND organism_id:3702",
        f"gene_exact:{gene} AND organism_id:3702",
    ]

    for q in queries:
        params = {
            "query": q,
            "format": "tsv",
            "fields": "accession,sequence",
        }
        print(f"  UniProt query: {q}")
        try:
            resp = requests.get(UNIPROT_URL, params=params, timeout=20)
        except Exception as e:  # noqa: BLE001
            print(f"  request error for {gene}: {e}")
            continue

        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} for {gene}")
            continue

        lines = resp.text.strip().splitlines()
        if len(lines) < 2:
            print(f"  no hits for {gene} with query: {q}")
            continue

        # Take the first hit
        acc, seq = lines[1].split("\t")
        print(f"  hit {acc} for {gene}")
        return acc, seq

    return None


def main() -> None:
    labels = pd.read_csv(LABELS_PATH, sep="\t")
    tair_ids = sorted(set(labels["tair_id"].astype(str)))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, tid in enumerate(tair_ids, start=1):
        print(f"[{i}/{len(tair_ids)}] fetching {tid} ...")
        try:
            res = fetch_sequence_for_tair(tid)
        except Exception as e:  # noqa: BLE001
            print(f"  error fetching {tid}: {e}")
            continue
        if res is None:
            print(f"  no UniProt match for {tid}")
            continue
        acc, seq = res
        rows.append({"tair_id": tid, "protein_accession": acc, "sequence": seq})
        time.sleep(0.5)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, sep="\t", index=False, encoding="utf-8")
    print(f"Wrote {len(df)} sequences to {OUT_PATH}")


if __name__ == "__main__":
    main()

