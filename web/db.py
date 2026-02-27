from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
SQLITE_PATH = ROOT_DIR / "data/processed/idr_llps.sqlite"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def list_samples(
    label: Optional[str] = None,
    label_source: Optional[str] = None,
    taxon_group: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    where_clauses: List[str] = []
    params: List[Any] = []
    if label:
        where_clauses.append("label = ?")
        params.append(label)
    if label_source:
        where_clauses.append("label_source = ?")
        params.append(label_source)
    if taxon_group:
        where_clauses.append("taxon_group = ?")
        params.append(taxon_group)
    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    with _connect() as conn:
        cur = conn.cursor()
        total = cur.execute(f"SELECT COUNT(*) FROM samples {where_sql}", params).fetchone()[0]
        rows = cur.execute(
            f"""
            SELECT
              sample_id,
              protein_accession,
              gene_name,
              organism,
              label,
              label_source,
              source_dbs,
              taxon_group,
              region_start,
              region_end
            FROM samples
            {where_sql}
            ORDER BY sample_id
            LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        ).fetchall()
    return [dict(r) for r in rows], int(total)


def get_sample(sample_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM samples
            WHERE sample_id = ?
            """,
            (sample_id,),
        ).fetchone()
    return dict(row) if row else None


def get_sample_features(sample_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM nardini90_features
            WHERE sample_id = ?
            """,
            (sample_id,),
        ).fetchone()
    return dict(row) if row else None


def get_stats() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    with _connect() as conn:
        cur = conn.cursor()
        out["n_samples"] = cur.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        out["label_counts"] = {
            r["label"]: r["sample_count"]
            for r in cur.execute("SELECT label, sample_count FROM v_label_summary").fetchall()
        }
        out["label_source_counts"] = {
            r["label_source"]: r["sample_count"]
            for r in cur.execute("SELECT label_source, sample_count FROM v_label_source_summary").fetchall()
        }
        out["group_counts"] = {
            "pos": int(
                cur.execute(
                    "SELECT COUNT(*) FROM samples WHERE label = 'idr_pos' AND label_source = 'experimental'"
                ).fetchone()[0]
            ),
            "neg": int(
                cur.execute(
                    "SELECT COUNT(*) FROM samples WHERE label = 'neg' AND label_source = 'exp_neg'"
                ).fetchone()[0]
            ),
            "neg_pseudo": int(
                cur.execute(
                    "SELECT COUNT(*) FROM samples WHERE label = 'neg' AND label_source = 'pseudo_neg'"
                ).fetchone()[0]
            ),
        }
    return out


