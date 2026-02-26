from __future__ import annotations

import argparse
from pathlib import Path

from src.common.pipeline_utils import (
    bool_from_any,
    jaccard,
    kmer_set,
    load_yaml,
    overlap_ratio,
    parse_int,
    read_table,
    safe_slice,
    write_tsv,
)


def has_high_risk_keyword(text: str, keywords: list[str]) -> bool:
    t = (text or "").lower()
    return any(k in t for k in keywords)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build experimental and pseudo negatives.")
    parser.add_argument("--config", default="configs/dataset.yaml")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    evidence_rows = read_table(Path(config["paths"]["evidence_file"]))
    idr_rows = read_table(Path(config["paths"]["idr_regions_file"]))
    positives = read_table(Path(config["paths"]["positives_file"]))
    negatives_file = Path(config["paths"]["negatives_file"])

    overlap_th = float(config["idr_rules"]["idr_overlap_threshold"])
    min_idr_length = int(config["idr_rules"]["min_idr_length"])
    keyword_blocklist = [k.lower() for k in config["negative_rules"]["high_risk_keyword_blocklist"]]
    exp_neg_sources = set(config["negative_rules"].get("experimental_negative_source_whitelist", []))
    homology_th = float(config["homology_control"]["identity_threshold"])
    ksize = int(config["homology_control"]["kmer_size"])

    positive_acc = {r.get("protein_accession", "") for r in positives}
    positive_seq_sets = [kmer_set(r.get("sequence", ""), k=ksize) for r in positives if r.get("sequence")]

    negatives = []
    seen = set()

    # A) Experimental negatives from evidence table.
    for r in evidence_rows:
        if exp_neg_sources and r.get("source_db", "") not in exp_neg_sources:
            continue
        if bool_from_any(r.get("llps_observed")):
            continue
        region_start = parse_int(r.get("region_start"))
        region_end = parse_int(r.get("region_end"))
        idr_start = parse_int(r.get("idr_start"))
        idr_end = parse_int(r.get("idr_end"))
        if not (region_start > 0 and region_end > 0 and idr_start > 0 and idr_end > 0):
            continue
        if overlap_ratio(region_start, region_end, idr_start, idr_end) < overlap_th:
            continue

        key = (
            r.get("protein_accession", ""),
            r.get("isoform", ""),
            str(region_start),
            str(region_end),
            r.get("construct_context", ""),
            "exp_neg",
        )
        if key in seen:
            continue
        seen.add(key)
        out = dict(r)
        out["label"] = "neg"
        out["label_source"] = "exp_neg"
        negatives.append(out)

    # B) High-confidence pseudo negatives from IDR pool.
    for r in idr_rows:
        acc = r.get("protein_accession", "")
        if not acc or acc in positive_acc:
            continue
        idr_start = parse_int(r.get("idr_start"))
        idr_end = parse_int(r.get("idr_end"))
        if idr_start <= 0 or idr_end <= 0 or (idr_end - idr_start + 1) < min_idr_length:
            continue

        text = " ".join([r.get("gene_name", ""), r.get("organism", ""), r.get("idr_source", "")]).strip()
        if has_high_risk_keyword(text, keyword_blocklist):
            continue

        seq = r.get("sequence", "")
        region_seq = safe_slice(seq, idr_start, idr_end)
        if not region_seq:
            continue

        region_set = kmer_set(region_seq, k=ksize)
        too_close = any(jaccard(region_set, pos_set) >= homology_th for pos_set in positive_seq_sets)
        if too_close:
            continue

        key = (acc, r.get("isoform", ""), str(idr_start), str(idr_end), "idr_region", "pseudo_neg")
        if key in seen:
            continue
        seen.add(key)
        negatives.append(
            {
                "evidence_id": f"pseudo_{acc}_{idr_start}_{idr_end}",
                "source_db": "pseudo_background",
                "source_entry_id": "",
                "protein_accession": acc,
                "isoform": r.get("isoform", "canonical"),
                "gene_name": r.get("gene_name", ""),
                "organism": r.get("organism", ""),
                "taxon_id": r.get("taxon_id", ""),
                "sequence": seq,
                "construct_context": "idr_region",
                "region_start": str(idr_start),
                "region_end": str(idr_end),
                "idr_start": str(idr_start),
                "idr_end": str(idr_end),
                "idr_source": r.get("idr_source", ""),
                "llps_observed": "False",
                "idr_driver_supported": "False",
                "evidence_grade": "C",
                "condition_text": "",
                "pmid": "",
                "db_version": "n/a",
                "download_date": "",
                "label": "neg",
                "label_source": "pseudo_neg",
            }
        )

    fields = list(config["output_fields"]["evidence"]) + ["label", "label_source"]
    write_tsv(negatives_file, negatives, fields)
    print(f"Wrote negative rows: {len(negatives)}")


if __name__ == "__main__":
    main()
