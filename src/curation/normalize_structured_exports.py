from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List

from src.common.pipeline_utils import load_yaml, utc_date, write_tsv


def read_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f, delimiter="\t")]


def read_json_dict(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_pmid(ref: str) -> str:
    ref = (ref or "").strip()
    if not ref:
        return ""
    if "pmid:" in ref.lower():
        return ref.split(":", 1)[-1].split(",")[0].strip()
    return ref.split(",")[0].strip()


def parse_boundaries(boundaries: str, seq_len: int) -> List[tuple[int, int]]:
    text = (boundaries or "").strip()
    if not text:
        return []
    out: List[tuple[int, int]] = []
    for start_s, end_s in re.findall(r"(\d+)\s*-\s*(\d+)", text):
        start = int(start_s)
        end = int(end_s)
        if start <= 0 or end <= 0 or end < start:
            continue
        if seq_len > 0 and end > seq_len:
            end = seq_len
        out.append((start, end))
    return out


def normalize_drllps(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for r in rows:
        llps_type = (r.get("LLPS Type") or "").strip().lower()
        if llps_type != "scaffold":
            # Keep dataset conservative: scaffold as driver positive candidates.
            continue
        seq = (r.get("Protein Sequence") or "").strip()
        if not seq:
            continue
        out.append(
            {
                "evidence_id": r.get("DrLLPS ID", ""),
                "source_entry_id": r.get("DrLLPS ID", ""),
                "protein_accession": (r.get("UniProt ID") or "").strip(),
                "isoform": "canonical",
                "gene_name": (r.get("Gene name") or "").strip(),
                "organism": (r.get("Species") or "").strip(),
                "taxon_id": "",
                "sequence": seq,
                "construct_context": "full_length",
                "region_start": "1",
                "region_end": str(len(seq)),
                "idr_start": "",
                "idr_end": "",
                "idr_source": "",
                "llps_observed": "True",
                "idr_driver_supported": "True",
                "evidence_grade": "B",
                "condition_text": f"drllps_type={r.get('LLPS Type', '')}",
                "pmid": first_pmid(r.get("References", "")),
                "db_version": "unknown",
                "download_date": utc_date(),
            }
        )
    return out


def normalize_disprot(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    seen = set()
    for r in rows:
        acc = (r.get("acc") or "").strip()
        start = (r.get("start") or "").strip()
        end = (r.get("end") or "").strip()
        if not acc or not start or not end:
            continue
        key = (acc, start, end, r.get("region_id", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "evidence_id": r.get("region_id", ""),
                "source_entry_id": r.get("disprot_id", ""),
                "protein_accession": acc,
                "isoform": "canonical",
                "gene_name": (r.get("name") or "").strip(),
                "organism": (r.get("organism") or "").strip(),
                "taxon_id": (r.get("ncbi_taxon_id") or "").strip(),
                "sequence": (r.get("region_sequence") or "").strip(),
                "construct_context": "idr_region",
                "region_start": start,
                "region_end": end,
                "idr_start": start,
                "idr_end": end,
                "idr_source": "disprot",
                "llps_observed": "",
                "idr_driver_supported": "",
                "evidence_grade": "C",
                "condition_text": (r.get("term_name") or "").strip(),
                "pmid": first_pmid(r.get("reference", "")),
                "db_version": "2025_06",
                "download_date": utc_date(),
            }
        )
    return out


def normalize_phasepro(entries: Dict) -> List[Dict[str, str]]:
    out = []
    for _, e in entries.items():
        acc = (e.get("accession") or "").strip()
        seq = (e.get("sequence") or "").strip()
        if not acc or not seq:
            continue
        bounds = parse_boundaries(str(e.get("boundaries", "")), seq_len=len(seq))
        if not bounds:
            bounds = [(1, len(seq))]

        for i, (start, end) in enumerate(bounds, start=1):
            out.append(
                {
                    "evidence_id": f"phasepro_{acc}_{i}",
                    "source_entry_id": str(e.get("id", "")),
                    "protein_accession": acc,
                    "isoform": "canonical",
                    "gene_name": (e.get("gene") or e.get("common_name") or "").strip(),
                    "organism": (e.get("organism") or "").strip(),
                    "taxon_id": str(e.get("taxon") or "").strip(),
                    "sequence": seq,
                    "construct_context": "driver_region",
                    "region_start": str(start),
                    "region_end": str(end),
                    "idr_start": str(start),
                    "idr_end": str(end),
                    "idr_source": "phasepro",
                    "llps_observed": "True",
                    "idr_driver_supported": "True",
                    "evidence_grade": "A",
                    "condition_text": (e.get("experiment_llps") or "").strip(),
                    "pmid": first_pmid(str(e.get("pmids", ""))),
                    "db_version": "1.1.0",
                    "download_date": utc_date(),
                }
            )
    return out


def _is_negative_row(r: Dict[str, str], config: Dict) -> bool:
    """Identify negative (non-LLPS) proteins in PPMC llpsdatasets format."""
    # Explicit category column
    for col in ("Category", "Classification", "Type", "category", "classification", "type"):
        if (r.get(col) or "").strip().lower() == "negative":
            return True
    # Column indicating presence in negative set (non-empty = negative)
    for col in ("Negative_databases", "Databases_negatives", "Negative", "Negative.Databases"):
        if (r.get(col) or "").strip():
            return True
    # Datasets column: "NP" = negative (PDB?), "ND" = negative (disordered?), or token "N"
    datasets = (r.get("Datasets") or r.get("datasets") or "").strip()
    if datasets:
        if datasets in ("NP", "ND") or datasets.startswith("N;"):
            return True
        for part in datasets.replace(";", " ").split():
            if part.strip() == "N":
                return True
    # Config: treat all rows as negative (e.g. when file is negative-only)
    if config.get("llpsdatasets", {}).get("treat_all_as_negative"):
        return True
    return False


def normalize_llpsdatasets(rows: List[Dict[str, str]], config: Dict) -> List[Dict[str, str]]:
    """Normalize PPMC Confident LLPS Datasets: output both negatives (NP/ND) and positives (driver/client)."""
    out = []
    for i, r in enumerate(rows):
        acc = (
            r.get("UniProt.Acc")
            or r.get("UniProt ID")
            or r.get("protein_accession")
            or r.get("accession")
            or ""
        ).strip()
        seq = (
            r.get("Full.seq")
            or r.get("Full_seq")
            or r.get("sequence")
            or r.get("Protein Sequence")
            or ""
        ).strip()
        if not acc or not seq:
            continue
        gene = (
            r.get("Gene.Name")
            or r.get("Gene name")
            or r.get("gene_name")
            or ""
        ).strip()
        organism = (r.get("organism") or r.get("Species") or "").strip()
        n = len(seq)
        taxon = (r.get("taxon_id") or r.get("taxon") or "").strip()
        db_version = config.get("llpsdatasets", {}).get("db_version", "unknown")

        if _is_negative_row(r, config):
            out.append(
                {
                    "evidence_id": f"llpsdatasets_neg_{acc}_{i}",
                    "source_entry_id": r.get("source_entry_id", acc),
                    "protein_accession": acc,
                    "isoform": "canonical",
                    "gene_name": gene,
                    "organism": organism,
                    "taxon_id": taxon,
                    "sequence": seq,
                    "construct_context": "full_length",
                    "region_start": "1",
                    "region_end": str(n),
                    "idr_start": "",
                    "idr_end": "",
                    "idr_source": "llpsdatasets",
                    "llps_observed": "False",
                    "idr_driver_supported": "False",
                    "evidence_grade": "C",
                    "condition_text": "PPMC curated negative (non-LLPS)",
                    "pmid": "",
                    "db_version": db_version,
                    "download_date": utc_date(),
                }
            )
        else:
            # Driver or client (CE, DE, C_D, etc.) -> positive evidence
            datasets = (r.get("Datasets") or r.get("datasets") or "").strip()
            out.append(
                {
                    "evidence_id": f"llpsdatasets_pos_{acc}_{i}",
                    "source_entry_id": r.get("source_entry_id", acc),
                    "protein_accession": acc,
                    "isoform": "canonical",
                    "gene_name": gene,
                    "organism": organism,
                    "taxon_id": taxon,
                    "sequence": seq,
                    "construct_context": "full_length",
                    "region_start": "1",
                    "region_end": str(n),
                    "idr_start": "",
                    "idr_end": "",
                    "idr_source": "llpsdatasets",
                    "llps_observed": "True",
                    "idr_driver_supported": "True",
                    "evidence_grade": "B",
                    "condition_text": f"PPMC curated driver/client (Datasets={datasets})",
                    "pmid": "",
                    "db_version": db_version,
                    "download_date": utc_date(),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize structured exports into per-source normalized.tsv")
    parser.add_argument("--config", default="configs/dataset.yaml")
    args = parser.parse_args()
    config = load_yaml(Path(args.config))
    raw_dir = Path(config["paths"]["raw_dir"])

    dr_rows = read_tsv(raw_dir / "drllps" / "LLPS.txt")
    dr_norm = normalize_drllps(dr_rows)
    write_tsv(
        raw_dir / "drllps" / "normalized.tsv",
        dr_norm,
        [
            "evidence_id",
            "source_entry_id",
            "protein_accession",
            "isoform",
            "gene_name",
            "organism",
            "taxon_id",
            "sequence",
            "construct_context",
            "region_start",
            "region_end",
            "idr_start",
            "idr_end",
            "idr_source",
            "llps_observed",
            "idr_driver_supported",
            "evidence_grade",
            "condition_text",
            "pmid",
            "db_version",
            "download_date",
        ],
    )

    dp_rows = read_tsv(raw_dir / "disprot" / "disprot_regions.tsv")
    dp_norm = normalize_disprot(dp_rows)
    write_tsv(
        raw_dir / "disprot" / "normalized.tsv",
        dp_norm,
        [
            "evidence_id",
            "source_entry_id",
            "protein_accession",
            "isoform",
            "gene_name",
            "organism",
            "taxon_id",
            "sequence",
            "construct_context",
            "region_start",
            "region_end",
            "idr_start",
            "idr_end",
            "idr_source",
            "llps_observed",
            "idr_driver_supported",
            "evidence_grade",
            "condition_text",
            "pmid",
            "db_version",
            "download_date",
        ],
    )
    pp_entries = read_json_dict(raw_dir / "phasepro" / "download_full.json")
    pp_norm = normalize_phasepro(pp_entries)
    write_tsv(
        raw_dir / "phasepro" / "normalized.tsv",
        pp_norm,
        [
            "evidence_id",
            "source_entry_id",
            "protein_accession",
            "isoform",
            "gene_name",
            "organism",
            "taxon_id",
            "sequence",
            "construct_context",
            "region_start",
            "region_end",
            "idr_start",
            "idr_end",
            "idr_source",
            "llps_observed",
            "idr_driver_supported",
            "evidence_grade",
            "condition_text",
            "pmid",
            "db_version",
            "download_date",
        ],
    )
    print(f"drllps normalized rows: {len(dr_norm)}")
    print(f"disprot normalized rows: {len(dp_norm)}")
    print(f"phasepro normalized rows: {len(pp_norm)}")

    llpsds_path = raw_dir / "llpsdatasets" / "datasets.tsv"
    if llpsds_path.exists():
        llpsds_rows = read_tsv(llpsds_path)
        llpsds_norm = normalize_llpsdatasets(llpsds_rows, config)
        write_tsv(
            raw_dir / "llpsdatasets" / "normalized.tsv",
            llpsds_norm,
            [
                "evidence_id",
                "source_entry_id",
                "protein_accession",
                "isoform",
                "gene_name",
                "organism",
                "taxon_id",
                "sequence",
                "construct_context",
                "region_start",
                "region_end",
                "idr_start",
                "idr_end",
                "idr_source",
                "llps_observed",
                "idr_driver_supported",
                "evidence_grade",
                "condition_text",
                "pmid",
                "db_version",
                "download_date",
            ],
        )
        print(f"llpsdatasets normalized rows (positives + negatives): {len(llpsds_norm)}")
    else:
        print("llpsdatasets/datasets.tsv not found, skipping llpsdatasets normalization")


if __name__ == "__main__":
    main()
