from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from src.common.pipeline_utils import (
    bool_from_any,
    load_yaml,
    parse_int,
    read_table,
    sample_id_from_key,
    utc_date,
    write_tsv,
)


def normalize_accession(raw_acc: str) -> tuple[str, str]:
    raw_acc = (raw_acc or "").strip()
    if "-" in raw_acc:
        acc, iso = raw_acc.split("-", 1)
        return acc, f"iso{iso}"
    return raw_acc, "canonical"


def normalize_row(row: Dict[str, str], source_db: str, db_version: str) -> Dict[str, str]:
    acc_input = row.get("protein_accession") or row.get("accession") or row.get("uniprot_id", "")
    protein_accession, isoform = normalize_accession(acc_input)
    region_start = parse_int(row.get("region_start"), default=parse_int(row.get("start"), 0))
    region_end = parse_int(row.get("region_end"), default=parse_int(row.get("end"), 0))
    idr_start = parse_int(row.get("idr_start"))
    idr_end = parse_int(row.get("idr_end"))

    evidence_id = row.get("evidence_id") or sample_id_from_key(
        [
            source_db,
            protein_accession,
            isoform,
            str(region_start),
            str(region_end),
            row.get("pmid", ""),
        ]
    )
    return {
        "evidence_id": evidence_id,
        "source_db": source_db,
        "source_entry_id": row.get("source_entry_id", row.get("entry_id", "")),
        "protein_accession": protein_accession,
        "isoform": row.get("isoform", isoform),
        "gene_name": row.get("gene_name", ""),
        "organism": row.get("organism", row.get("species", "")),
        "taxon_id": row.get("taxon_id", ""),
        "sequence": row.get("sequence", ""),
        "construct_context": row.get("construct_context", "full_length"),
        "region_start": str(region_start),
        "region_end": str(region_end),
        "idr_start": str(idr_start),
        "idr_end": str(idr_end),
        "idr_source": row.get("idr_source", ""),
        "llps_observed": str(bool_from_any(row.get("llps_observed"))),
        "idr_driver_supported": str(bool_from_any(row.get("idr_driver_supported"))),
        "evidence_grade": row.get("evidence_grade", "C"),
        "condition_text": row.get("condition_text", ""),
        "pmid": row.get("pmid", ""),
        "db_version": row.get("db_version", db_version),
        "download_date": row.get("download_date", utc_date()),
    }


def collect_normalized_rows(raw_dir: Path) -> List[Dict[str, str]]:
    all_rows: List[Dict[str, str]] = []
    for source_dir in sorted([p for p in raw_dir.iterdir() if p.is_dir()]):
        source = source_dir.name
        normalized_file = source_dir / "normalized.tsv"
        if not normalized_file.exists():
            continue
        rows = read_table(normalized_file)
        for row in rows:
            all_rows.append(
                normalize_row(row=row, source_db=source, db_version=row.get("db_version", "unknown"))
            )
    return all_rows


def attach_idr_to_non_idr_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    idr_index: Dict[tuple[str, str], List[Dict[str, str]]] = {}
    for r in rows:
        idr_start = parse_int(r.get("idr_start"))
        idr_end = parse_int(r.get("idr_end"))
        if idr_start <= 0 or idr_end <= 0:
            continue
        key = (r.get("protein_accession", ""), r.get("isoform", "canonical"))
        idr_index.setdefault(key, []).append(r)

    expanded = []
    for r in rows:
        idr_start = parse_int(r.get("idr_start"))
        idr_end = parse_int(r.get("idr_end"))
        if idr_start > 0 and idr_end > 0:
            expanded.append(r)
            continue

        key = (r.get("protein_accession", ""), r.get("isoform", "canonical"))
        candidates = idr_index.get(key, [])
        if not candidates:
            expanded.append(r)
            continue

        for idr in candidates:
            merged = dict(r)
            merged["idr_start"] = idr.get("idr_start", "")
            merged["idr_end"] = idr.get("idr_end", "")
            merged["idr_source"] = idr.get("idr_source", "")
            expanded.append(merged)
    return expanded


def make_idr_region_rows(evidence_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for r in evidence_rows:
        if not r["protein_accession"]:
            continue
        idr_start = parse_int(r.get("idr_start"))
        idr_end = parse_int(r.get("idr_end"))
        if idr_start <= 0 or idr_end <= 0:
            continue
        key = (
            r["protein_accession"],
            r["isoform"],
            str(idr_start),
            str(idr_end),
            r.get("idr_source", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "protein_accession": r["protein_accession"],
                "isoform": r["isoform"],
                "gene_name": r.get("gene_name", ""),
                "organism": r.get("organism", ""),
                "taxon_id": r.get("taxon_id", ""),
                "sequence": r.get("sequence", ""),
                "idr_start": str(idr_start),
                "idr_end": str(idr_end),
                "idr_source": r.get("idr_source", ""),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cross-DB evidence layer for IDR-LLPS.")
    parser.add_argument("--config", default="configs/dataset.yaml")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    raw_dir = Path(config["paths"]["raw_dir"])
    evidence_file = Path(config["paths"]["evidence_file"])
    idr_regions_file = Path(config["paths"]["idr_regions_file"])

    rows = collect_normalized_rows(raw_dir)
    rows = attach_idr_to_non_idr_rows(rows)
    write_tsv(evidence_file, rows, config["output_fields"]["evidence"])

    idr_rows = make_idr_region_rows(rows)
    write_tsv(
        idr_regions_file,
        idr_rows,
        [
            "protein_accession",
            "isoform",
            "gene_name",
            "organism",
            "taxon_id",
            "sequence",
            "idr_start",
            "idr_end",
            "idr_source",
        ],
    )
    print(f"Wrote evidence rows: {len(rows)}")
    print(f"Wrote IDR regions: {len(idr_rows)}")


if __name__ == "__main__":
    main()
