from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class ResidueScore:
    pos: int
    score: float


def load_tair_sequences(path: Path) -> Dict[str, str]:
    df = pd.read_csv(path, sep="\t")
    if "tair_id" not in df.columns or "sequence" not in df.columns:
        raise ValueError("Expected columns 'tair_id' and 'sequence' in tair_sequences.tsv")
    seqs: Dict[str, str] = {}
    for _, row in df.iterrows():
        tid = str(row["tair_id"]).strip()
        seq = str(row["sequence"] or "").strip().upper()
        if not tid or not seq:
            continue
        seqs[tid] = seq
    return seqs


def parse_iupred_long(path: Path) -> Dict[str, List[ResidueScore]]:
    """
    Parse IUPred2A 'long' mode output.

    Assumes the input was generated e.g. by:
      iupred2a.py tair_sequences.fasta long > tair_iupred_long.txt

    Format per sequence (typical):
      >sequence_id
      # comment ...
      1   M   0.45
      2   E   0.67
      ...
    """
    scores: Dict[str, List[ResidueScore]] = {}
    current_id: str | None = None
    current_list: List[ResidueScore] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            if line.startswith(">"):
                # flush previous
                if current_id is not None and current_list:
                    scores[current_id] = current_list
                current_id = line[1:].strip()
                current_list = []
                continue

            if current_id is None:
                # unexpected content before any header; skip
                continue

            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                pos = int(parts[0])
                sc = float(parts[-1])
            except ValueError:
                continue
            current_list.append(ResidueScore(pos=pos, score=sc))

    if current_id is not None and current_list:
        scores[current_id] = current_list

    return scores


def scores_to_idrs(
    residues: List[ResidueScore],
    threshold: float = 0.5,
    min_len: int = 30,
) -> List[Tuple[int, int, List[float]]]:
    """
    Convert per-residue disorder scores into IDR segments.

    Returns a list of (start, end, scores_in_segment).
    start/end are 1-based inclusive positions in the original sequence.
    """
    segments: List[Tuple[int, int, List[float]]] = []
    in_run = False
    run_start = 0
    run_scores: List[float] = []
    prev_pos = 0

    for r in residues:
        if r.score >= threshold:
            if not in_run:
                in_run = True
                run_start = r.pos
                run_scores = [r.score]
            else:
                # if positions are not strictly consecutive, we still treat as contiguous
                run_scores.append(r.score)
        else:
            if in_run:
                run_end = prev_pos
                length = run_end - run_start + 1
                if length >= min_len:
                    segments.append((run_start, run_end, run_scores.copy()))
                in_run = False
                run_scores = []
        prev_pos = r.pos

    if in_run and residues:
        run_end = residues[-1].pos
        length = run_end - run_start + 1
        if length >= min_len:
            segments.append((run_start, run_end, run_scores.copy()))

    return segments


def main() -> None:
    """
    Parse IUPred2A 'long' scores for TAIR proteins and extract IDR segments.

    Inputs:
      - data/validation/tair_sequences.tsv
      - data/validation/tair_iupred_long.txt

    Output:
      - data/validation/tair_idr_regions.tsv with columns:
          tair_id, idr_index, idr_start, idr_end,
          idr_length, mean_score, max_score, idr_seq
    """
    root = Path(__file__).resolve().parents[2]
    seq_tsv = root / "data" / "validation" / "tair_sequences.tsv"
    iupred_path = root / "data" / "validation" / "tair_iupred_long.txt"
    out_path = root / "data" / "validation" / "tair_idr_regions.tsv"

    print(f"Loading TAIR sequences from: {seq_tsv}")
    seqs = load_tair_sequences(seq_tsv)
    print(f"Loaded {len(seqs)} sequences.")

    print(f"Parsing IUPred2A scores from: {iupred_path}")
    scores = parse_iupred_long(iupred_path)
    print(f"Parsed scores for {len(scores)} proteins.")

    rows: List[Dict[str, object]] = []
    total_idrs = 0
    for tid, res_list in scores.items():
        if tid not in seqs:
            # Skip sequences not used in TAIR validation set
            continue
        segs = scores_to_idrs(res_list, threshold=0.5, min_len=30)
        if not segs:
            continue
        seq = seqs[tid]
        for idx, (start, end, seg_scores) in enumerate(segs, start=1):
            # 1-based inclusive; Python slice is [start-1:end)
            if start <= 0 or end <= 0 or end < start or start > len(seq):
                continue
            end_clamped = min(end, len(seq))
            idr_seq = seq[start - 1 : end_clamped]
            if not idr_seq:
                continue
            length = len(idr_seq)
            mean_score = float(sum(seg_scores) / len(seg_scores))
            max_score = float(max(seg_scores))
            rows.append(
                {
                    "tair_id": tid,
                    "idr_index": idx,
                    "idr_start": start,
                    "idr_end": end_clamped,
                    "idr_length": length,
                    "mean_score": mean_score,
                    "max_score": max_score,
                    "idr_seq": idr_seq,
                }
            )
            total_idrs += 1

    if not rows:
        print("No IDR segments detected above threshold; nothing to write.")
        return

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, sep="\t", index=False, encoding="utf-8")
    n_proteins = out_df["tair_id"].nunique()
    print(
        f"Wrote {total_idrs} IDR segments for {n_proteins} TAIR proteins "
        f"to: {out_path}"
    )


if __name__ == "__main__":
    main()

