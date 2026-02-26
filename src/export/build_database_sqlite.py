from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List

from src.common.pipeline_utils import load_yaml, read_table, utc_timestamp


def infer_sql_type(col: str) -> str:
    int_cols = {
        "region_start",
        "region_end",
        "idr_start",
        "idr_end",
        "taxon_id",
        "evidence_count",
    }
    if col in int_cols:
        return "INTEGER"
    return "TEXT"


def create_table(conn: sqlite3.Connection, table: str, columns: List[str]) -> None:
    cols = ", ".join([f'"{c}" {infer_sql_type(c)}' for c in columns])
    conn.execute(f'DROP TABLE IF EXISTS "{table}"')
    conn.execute(f'CREATE TABLE "{table}" ({cols})')


def insert_rows(conn: sqlite3.Connection, table: str, columns: List[str], rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    placeholders = ", ".join(["?"] * len(columns))
    col_sql = ", ".join([f'"{c}"' for c in columns])
    sql = f'INSERT INTO "{table}" ({col_sql}) VALUES ({placeholders})'
    payload = [tuple(r.get(c, "") for c in columns) for r in rows]
    conn.executemany(sql, payload)


def create_indexes(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_evidence_acc_iso ON evidence (protein_accession, isoform)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_llps ON evidence (llps_observed)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_samples_label ON samples (label)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_samples_acc_region ON samples (protein_accession, region_start, region_end)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_splits_sample ON splits (sample_id)")


def build_meta_table(conn: sqlite3.Connection, stats: Dict[str, str]) -> None:
    conn.execute("DROP TABLE IF EXISTS metadata")
    conn.execute('CREATE TABLE metadata ("key" TEXT PRIMARY KEY, "value" TEXT)')
    conn.executemany(
        'INSERT INTO metadata ("key","value") VALUES (?,?)',
        [(k, str(v)) for k, v in stats.items()],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SQLite database from processed TSV tables.")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument(
        "--output",
        default="data/processed/idr_llps.sqlite",
        help="Output SQLite path.",
    )
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    evidence_rows = read_table(Path(config["paths"]["evidence_file"]))
    idr_rows = read_table(Path(config["paths"]["idr_regions_file"]))
    sample_rows = read_table(Path(config["paths"]["samples_file"]))
    split_rows = read_table(Path(config["paths"]["splits_file"]))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(out)
    try:
        create_table(conn, "evidence", config["output_fields"]["evidence"])
        insert_rows(conn, "evidence", config["output_fields"]["evidence"], evidence_rows)

        idr_cols = [
            "protein_accession",
            "isoform",
            "gene_name",
            "organism",
            "taxon_id",
            "sequence",
            "idr_start",
            "idr_end",
            "idr_source",
        ]
        create_table(conn, "idr_regions", idr_cols)
        insert_rows(conn, "idr_regions", idr_cols, idr_rows)

        create_table(conn, "samples", config["output_fields"]["samples"])
        insert_rows(conn, "samples", config["output_fields"]["samples"], sample_rows)

        create_table(conn, "splits", config["output_fields"]["splits"])
        insert_rows(conn, "splits", config["output_fields"]["splits"], split_rows)

        create_indexes(conn)

        stats = {
            "dataset_name": config["project"]["name"],
            "dataset_version": config["project"]["version"],
            "created_at": utc_timestamp(),
            "evidence_rows": str(len(evidence_rows)),
            "idr_regions_rows": str(len(idr_rows)),
            "samples_rows": str(len(sample_rows)),
            "splits_rows": str(len(split_rows)),
        }
        build_meta_table(conn, stats)
        conn.commit()
    finally:
        conn.close()

    print(f"Wrote SQLite DB: {out}")


if __name__ == "__main__":
    main()
