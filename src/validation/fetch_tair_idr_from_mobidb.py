from __future__ import annotations

import argparse
import concurrent.futures
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
TAIR_SEQ_PATH = ROOT / "data" / "validation" / "tair_sequences.tsv"
OUT_PATH = ROOT / "data" / "validation" / "tair_idr_regions.tsv"

MOBIDB_URL = "https://mobidb.org/api/download"


def load_tair_sequences() -> pd.DataFrame:
    df = pd.read_csv(TAIR_SEQ_PATH, sep="\t")
    if "tair_id" not in df.columns or "sequence" not in df.columns:
        raise ValueError("tair_sequences.tsv must contain 'tair_id' and 'sequence' columns.")
    # protein_accession is recommended but not strictly required for all rows
    if "protein_accession" not in df.columns:
        df["protein_accession"] = pd.NA
    # normalize
    df["tair_id"] = df["tair_id"].astype(str).str.strip()
    df["protein_accession"] = df["protein_accession"].astype(str).str.strip()
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    return df


def fetch_mobidb_record(accession: str) -> dict | None:
    params = {"acc": accession, "format": "json"}
    try:
        resp = requests.get(MOBIDB_URL, params=params, timeout=20)
    except Exception as e:  # noqa: BLE001
        print(f"  error contacting MobiDB for {accession}: {e}")
        return None
    if resp.status_code != 200:
        print(f"  MobiDB HTTP {resp.status_code} for {accession}")
        return None
    try:
        return resp.json()
    except Exception as e:  # noqa: BLE001
        print(f"  error parsing JSON for {accession}: {e}")
        return None


def extract_idr_regions(record: dict) -> List[Tuple[int, int, str]]:
    """
    Extract IDR regions from a MobiDB JSON record.

    Strategy:
      1) Prefer curated DisProt / IDEAL if present.
      2) Otherwise fall back to prediction-disorder-mobidb_lite.
    Returns list of (start, end, source_name), 1-based inclusive coordinates.
    """
    regions: List[Tuple[int, int, str]] = []

    # Curated first
    for key in ["curated-disorder-disprot", "curated-disorder-ideal"]:
        ann = record.get(key)
        if not ann:
            continue
        for start, end in ann.get("regions", []):
            try:
                s = int(start)
                e = int(end)
            except Exception:
                continue
            regions.append((s, e, key))

    # Predicted mobidb_lite as fallback / supplement
    pred = record.get("prediction-disorder-mobidb_lite")
    if pred:
        for start, end in pred.get("regions", []):
            try:
                s = int(start)
                e = int(end)
            except Exception:
                continue
            regions.append((s, e, "prediction-disorder-mobidb_lite"))

    return regions


def _process_one_accession(
    idx: int,
    total: int,
    acc: str,
    tid_seq_list: List[Tuple[str, str]],
) -> List[Dict[str, object]]:
    """
    Worker to fetch and project IDR regions for a single UniProt accession.
    Designed to run in parallel (IO-bound, use threads).
    """
    print(f"[{idx}/{total}] fetching IDR annotations for {acc} from MobiDB ...")
    record = fetch_mobidb_record(acc)
    if not record:
        return []
    idr_regions = extract_idr_regions(record)
    if not idr_regions:
        print(f"  no IDR regions found for {acc}")
        return []

    rows: List[Dict[str, object]] = []
    for tid, seq in tid_seq_list:
        seq_len = len(seq)
        for start, end, source_name in idr_regions:
            if start <= 0 or end <= 0 or end < start:
                continue
            # Clamp to sequence length
            if start > seq_len:
                continue
            end_clamped = min(end, seq_len)
            idr_seq = seq[start - 1 : end_clamped]
            if not idr_seq:
                continue
            rows.append(
                {
                    "tair_id": tid,
                    "protein_accession": acc,
                    "idr_start": start,
                    "idr_end": end_clamped,
                    "idr_length": len(idr_seq),
                    "idr_source": source_name,
                    "idr_seq": idr_seq,
                }
            )
    # be a bit gentle per accession
    time.sleep(0.1)
    print(f"  {acc}: collected {len(rows)} IDR segments across {len(tid_seq_list)} TAIR proteins")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch IDR annotations for TAIR proteins from MobiDB (DisProt/IDEAL + mobidb_lite)."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel threads for MobiDB requests (IO-bound).",
    )
    args = parser.parse_args()

    print(f"Loading TAIR sequences from: {TAIR_SEQ_PATH}")
    df = load_tair_sequences()

    # Build mapping accession -> list of (tair_id, sequence)
    acc_map: Dict[str, List[Tuple[str, str]]] = {}
    for _, row in df.iterrows():
        tid = row["tair_id"]
        acc = row["protein_accession"]
        seq = row["sequence"]
        if not acc or acc == "nan":
            continue
        acc_map.setdefault(acc, []).append((tid, seq))

    n_acc = len(acc_map)
    print(f"Total TAIR rows: {len(df)}, with UniProt accession: {n_acc} unique accessions.")
    if n_acc == 0:
        print("No UniProt accessions available; cannot query MobiDB.")
        return

    all_rows: List[Dict[str, object]] = []
    # Parallelize over accessions (IO-bound)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for i, (acc, tid_seq_list) in enumerate(acc_map.items(), start=1):
            futures.append(
                ex.submit(_process_one_accession, i, n_acc, acc, tid_seq_list)
            )
        for fut in concurrent.futures.as_completed(futures):
            rows = fut.result()
            if rows:
                all_rows.extend(rows)

    if not all_rows:
        print("No IDR regions collected from MobiDB; nothing to write.")
        return

    out_df = pd.DataFrame(all_rows)
    # Order segments and assign idr_index within each TAIR protein
    out_df = out_df.sort_values(["tair_id", "idr_start", "idr_end"]).reset_index(drop=True)
    out_df["idr_index"] = out_df.groupby("tair_id").cumcount() + 1
    # Reorder columns for readability
    cols = [
        "tair_id",
        "protein_accession",
        "idr_index",
        "idr_start",
        "idr_end",
        "idr_length",
        "idr_source",
        "idr_seq",
    ]
    cols = [c for c in cols if c in out_df.columns] + [
        c for c in out_df.columns if c not in cols
    ]
    out_df = out_df.loc[:, cols]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, sep="\t", index=False, encoding="utf-8")
    print(
        f"Wrote {len(out_df)} IDR segments for {out_df['tair_id'].nunique()} TAIR proteins "
        f"to: {OUT_PATH}"
    )


if __name__ == "__main__":
    main()

