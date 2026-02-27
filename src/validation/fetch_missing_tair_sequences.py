from __future__ import annotations

import argparse
import concurrent.futures
import re
from pathlib import Path

import pandas as pd

from src.validation.fetch_tair_sequences import fetch_sequence_for_tair


ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = ROOT / "data/validation/tair_llps_labels.tsv"
EXISTING_PATH = ROOT / "data/validation/tair_sequences.tsv"


def _normalize_tair(t: str) -> str:
    """Uppercase and strip version suffix, e.g. AT1G01030.1 -> AT1G01030."""
    s = str(t).upper()
    return re.sub(r"\.\d+$", "", s)


def _fetch_one(idx: int, total: int, tid: str):
    """Worker for parallel UniProt fetching (IO-bound, use threads)."""
    print(f"[{idx}/{total}] fetching {tid} ...")
    try:
        res = fetch_sequence_for_tair(tid)
    except Exception as e:  # noqa: BLE001
        print(f"  error fetching {tid}: {e}")
        return None
    if res is None:
        print(f"  no UniProt match for {tid}")
        return None
    acc, seq = res
    return {
        "tair_id": tid,
        "protein_accession": acc,
        "sequence": seq,
        "source": "uniprot_api",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch missing TAIR sequences from UniProt in parallel."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel threads for UniProt requests.",
    )
    args = parser.parse_args()

    print(f"Loading labels from: {LABELS_PATH}")
    labels = pd.read_csv(LABELS_PATH, sep="\t")
    print(f"Loading existing sequences from: {EXISTING_PATH}")
    existing = pd.read_csv(EXISTING_PATH, sep="\t")

    label_ids_norm = {_normalize_tair(t) for t in labels["tair_id"].astype(str)}
    existing_ids_norm = {_normalize_tair(t) for t in existing["tair_id"].astype(str)}

    missing_ids = sorted(label_ids_norm - existing_ids_norm)
    total_missing = len(missing_ids)
    print(f"Total TAIR IDs in labels: {len(label_ids_norm)}")
    print(f"Existing TAIR sequences: {len(existing_ids_norm)}")
    print(f"Missing sequences to fetch from UniProt: {total_missing}")

    if not missing_ids:
        print("No missing TAIR IDs; nothing to fetch from UniProt.")
        return

    rows_new = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(_fetch_one, i, total_missing, tid)
            for i, tid in enumerate(missing_ids, start=1)
        ]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            if row is not None:
                rows_new.append(row)

    if not rows_new:
        print("No new sequences fetched; leaving existing file unchanged.")
        return

    new_df = pd.DataFrame(rows_new)

    # Ensure existing has the same columns (fill missing with NA)
    if "protein_accession" not in existing.columns:
        existing["protein_accession"] = pd.NA
    if "source" not in existing.columns:
        existing["source"] = "proteome_fasta"

    merged = pd.concat([existing, new_df], ignore_index=True)
    # Drop any accidental exact-duplicate rows
    merged = merged.drop_duplicates(subset=["tair_id", "sequence"])

    merged.to_csv(EXISTING_PATH, sep="\t", index=False, encoding="utf-8")
    print(
        f"Wrote merged TAIR sequences with {len(merged)} rows ("
        f"{len(existing)} existing + {len(new_df)} new) to {EXISTING_PATH}"
    )


if __name__ == "__main__":
    main()

