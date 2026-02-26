from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from src.common.pipeline_utils import (
    jaccard,
    kmer_set,
    load_yaml,
    parse_int,
    read_table,
    safe_slice,
    sample_id_from_key,
    utc_timestamp,
    write_json,
    write_tsv,
)


def idr_length_bin(length: int) -> str:
    if length < 30:
        return "short"
    if length < 100:
        return "medium"
    return "long"


def taxon_group(taxon_id: str) -> str:
    if not taxon_id:
        return "unknown"
    if taxon_id == "9606":
        return "human"
    return "non_human"


def make_sample_rows(config: Dict, rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped = {}
    for r in rows:
        key_values = [r.get(k, "") for k in config["sample_primary_key"]]
        key = tuple(key_values)
        grouped.setdefault(key, []).append(r)

    samples = []
    for key, g in grouped.items():
        first = g[0]
        region_start = parse_int(first.get("region_start"))
        region_end = parse_int(first.get("region_end"))
        seq = first.get("sequence", "")
        region_sequence = safe_slice(seq, region_start, region_end)
        evidence_grades = sorted({x.get("evidence_grade", "C") for x in g})
        source_dbs = sorted({x.get("source_db", "") for x in g if x.get("source_db")})
        pmids = sorted({x.get("pmid", "") for x in g if x.get("pmid")})

        sample_id = sample_id_from_key([str(v) for v in key])
        samples.append(
            {
                "sample_id": sample_id,
                "protein_accession": first.get("protein_accession", ""),
                "isoform": first.get("isoform", ""),
                "gene_name": first.get("gene_name", ""),
                "organism": first.get("organism", ""),
                "taxon_id": first.get("taxon_id", ""),
                "region_start": str(region_start),
                "region_end": str(region_end),
                "region_sequence": region_sequence,
                "idr_start": first.get("idr_start", ""),
                "idr_end": first.get("idr_end", ""),
                "idr_source": first.get("idr_source", ""),
                "label": first.get("label", ""),
                "label_source": first.get("label_source", ""),
                "evidence_grade_max": evidence_grades[0] if evidence_grades else "",
                "evidence_count": str(len(g)),
                "source_dbs": ";".join(source_dbs),
                "pmids": ";".join(pmids),
                "construct_context": first.get("construct_context", ""),
                "idr_length_bin": idr_length_bin(max(region_end - region_start + 1, 0)),
                "taxon_group": taxon_group(first.get("taxon_id", "")),
            }
        )
    return samples


def cluster_samples(samples: List[Dict[str, str]], identity_th: float, ksize: int) -> Dict[str, int]:
    clusters: Dict[str, int] = {}
    cluster_representatives: Dict[int, set[str]] = {}
    next_cluster = 0
    for s in samples:
        sid = s["sample_id"]
        seq_set = kmer_set(s.get("region_sequence", ""), k=ksize)
        assigned = None
        for cid, rep in cluster_representatives.items():
            if jaccard(seq_set, rep) >= identity_th:
                assigned = cid
                break
        if assigned is None:
            assigned = next_cluster
            cluster_representatives[assigned] = seq_set
            next_cluster += 1
        clusters[sid] = assigned
    return clusters


def assign_split_by_cluster(
    samples: List[Dict[str, str]],
    clusters: Dict[str, int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, str]:
    random.seed(seed)
    cluster_to_samples = defaultdict(list)
    for s in samples:
        cluster_to_samples[clusters[s["sample_id"]]].append(s["sample_id"])

    cluster_ids = list(cluster_to_samples.keys())
    random.shuffle(cluster_ids)
    total = len(cluster_ids)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    train_set = set(cluster_ids[:n_train])
    val_set = set(cluster_ids[n_train : n_train + n_val])

    split_map = {}
    for cid, sids in cluster_to_samples.items():
        if cid in train_set:
            split = "train"
        elif cid in val_set:
            split = "val"
        else:
            split = "test"
        for sid in sids:
            split_map[sid] = split
    return split_map


def write_versioned_tables(config: Dict, samples: List[Dict[str, str]], split_map: Dict[str, str]) -> None:
    processed = Path(config["paths"]["processed_dir"])
    strict_rows = [r for r in samples if r.get("label_source") in {"experimental", "exp_neg"}]
    hybrid_rows = list(samples)

    # The base samples table is the hybrid view.
    write_tsv(Path(config["paths"]["samples_file"]), hybrid_rows, config["output_fields"]["samples"])
    write_tsv(processed / "samples_strict.tsv", strict_rows, config["output_fields"]["samples"])
    write_tsv(processed / "samples_hybrid.tsv", hybrid_rows, config["output_fields"]["samples"])

    splits = []
    for r in hybrid_rows:
        splits.append(
            {
                "sample_id": r["sample_id"],
                "split": split_map[r["sample_id"]],
                "dataset_version": config["project"]["version"],
            }
        )
    write_tsv(Path(config["paths"]["splits_file"]), splits, config["output_fields"]["splits"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ML-ready samples and data splits.")
    parser.add_argument("--config", default="configs/dataset.yaml")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    positives = read_table(Path(config["paths"]["positives_file"]))
    negatives = read_table(Path(config["paths"]["negatives_file"]))
    merged_rows = positives + negatives

    samples = make_sample_rows(config, merged_rows)
    identity_th = float(config["homology_control"]["identity_threshold"])
    ksize = int(config["homology_control"]["kmer_size"])
    clusters = cluster_samples(samples, identity_th=identity_th, ksize=ksize)
    split_map = assign_split_by_cluster(
        samples=samples,
        clusters=clusters,
        train_ratio=float(config["split"]["train"]),
        val_ratio=float(config["split"]["val"]),
        seed=int(config["split"]["random_seed"]),
    )
    write_versioned_tables(config, samples, split_map)

    metadata = {
        "dataset_name": config["project"]["name"],
        "dataset_version": config["project"]["version"],
        "created_at": utc_timestamp(),
        "sample_count_hybrid": len(samples),
        "sample_count_strict": len(
            [r for r in samples if r.get("label_source") in {"experimental", "exp_neg"}]
        ),
    }
    write_json(Path(config["paths"]["run_metadata_file"]), metadata)
    print(f"Wrote samples: {len(samples)}")


if __name__ == "__main__":
    main()
