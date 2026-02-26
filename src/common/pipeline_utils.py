from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def load_yaml(path: Path) -> Dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required. Install with: pip install pyyaml"
        ) from exc

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_table(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return [dict(r) for r in reader]


def write_tsv(path: Path, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def bool_from_any(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_int(value: str | int | None, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def sample_id_from_key(values: Sequence[str]) -> str:
    key = "|".join(values)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return f"SMP_{digest[:16]}"


def kmer_set(seq: str, k: int = 3) -> set[str]:
    seq = (seq or "").upper().strip()
    if len(seq) < k:
        return {seq} if seq else set()
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def safe_slice(seq: str, start: int, end: int) -> str:
    if not seq:
        return ""
    left = max(start - 1, 0)
    right = max(end, left)
    return seq[left:right]


def overlap_ratio(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    left = max(a_start, b_start)
    right = min(a_end, b_end)
    if right < left:
        return 0.0
    overlap = right - left + 1
    a_len = max(a_end - a_start + 1, 1)
    b_len = max(b_end - b_start + 1, 1)
    return overlap / min(a_len, b_len)
