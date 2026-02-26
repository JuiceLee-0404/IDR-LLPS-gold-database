from __future__ import annotations

import argparse
from pathlib import Path

from src.common.pipeline_utils import (
    bool_from_any,
    load_yaml,
    overlap_ratio,
    parse_int,
    read_table,
    write_tsv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter evidence to IDR-driven LLPS positives.")
    parser.add_argument("--config", default="configs/dataset.yaml")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    evidence_file = Path(config["paths"]["evidence_file"])
    positives_file = Path(config["paths"]["positives_file"])
    conflicts_file = Path(config["paths"]["conflicts_file"])

    overlap_th = float(config["idr_rules"]["idr_overlap_threshold"])
    accepted_grades = set(config["positive_rules"]["include_evidence_grades"])

    evidence_rows = read_table(evidence_file)
    positives = []
    conflict = []
    seen_positive_keys = set()
    negative_keys = set()

    for r in evidence_rows:
        llps = bool_from_any(r.get("llps_observed"))
        driver = bool_from_any(r.get("idr_driver_supported"))
        grade = (r.get("evidence_grade") or "C").upper()
        region_start = parse_int(r.get("region_start"))
        region_end = parse_int(r.get("region_end"))
        idr_start = parse_int(r.get("idr_start"))
        idr_end = parse_int(r.get("idr_end"))
        key = (
            r.get("protein_accession", ""),
            r.get("isoform", ""),
            str(region_start),
            str(region_end),
            r.get("construct_context", ""),
        )

        idr_present = idr_start > 0 and idr_end > 0
        overlap = 0.0
        if idr_present and region_start > 0 and region_end > 0:
            overlap = overlap_ratio(region_start, region_end, idr_start, idr_end)

        is_positive = (
            llps
            and grade in accepted_grades
            and idr_present
            and (driver or overlap >= overlap_th)
        )
        if is_positive:
            if key in negative_keys:
                conflict.append(
                    {
                        "conflict_type": "positive_negative_conflict",
                        "protein_accession": key[0],
                        "isoform": key[1],
                        "region_start": key[2],
                        "region_end": key[3],
                        "construct_context": key[4],
                        "evidence_id": r.get("evidence_id", ""),
                    }
                )
                continue
            if key not in seen_positive_keys:
                seen_positive_keys.add(key)
                out = dict(r)
                out["label"] = "idr_pos"
                out["label_source"] = "experimental"
                positives.append(out)
        elif not llps:
            negative_keys.add(key)
            if key in seen_positive_keys:
                conflict.append(
                    {
                        "conflict_type": "negative_positive_conflict",
                        "protein_accession": key[0],
                        "isoform": key[1],
                        "region_start": key[2],
                        "region_end": key[3],
                        "construct_context": key[4],
                        "evidence_id": r.get("evidence_id", ""),
                    }
                )

    positive_fields = list(config["output_fields"]["evidence"]) + ["label", "label_source"]
    write_tsv(positives_file, positives, positive_fields)
    write_tsv(
        conflicts_file,
        conflict,
        [
            "conflict_type",
            "protein_accession",
            "isoform",
            "region_start",
            "region_end",
            "construct_context",
            "evidence_id",
        ],
    )
    print(f"Wrote IDR-positive rows: {len(positives)}")
    print(f"Wrote conflict rows: {len(conflict)}")


if __name__ == "__main__":
    main()
