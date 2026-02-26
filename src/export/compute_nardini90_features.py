from __future__ import annotations

import argparse
import math
import random
import sqlite3
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import scipy.stats as stats

from src.common.pipeline_utils import load_yaml, read_table, utc_timestamp, write_tsv


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)

GROUPS_8 = {
    "POLAR": set("STNQCH"),
    "HYDROPHOBIC": set("ILMV"),
    "POSITIVE": set("RK"),
    "NEGATIVE": set("ED"),
    "AROMATIC": set("FWY"),
    "ALANINE": set("A"),
    "PROLINE": set("P"),
    "GLYCINE": set("G"),
}
TYPEALL_8x8 = (
    ["S", "T", "N", "Q", "C", "H"],
    ["I", "L", "M", "V"],
    ["R", "K"],
    ["E", "D"],
    ["F", "W", "Y"],
    ["A"],
    ["P"],
    ["G"],
)

HYDROPATHY_KD = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

PPII_SCALE = {
    "A": 0.37,
    "C": 0.25,
    "D": 0.54,
    "E": 0.37,
    "F": 0.30,
    "G": 0.13,
    "H": 0.36,
    "I": 0.39,
    "K": 0.56,
    "L": 0.24,
    "M": 0.36,
    "N": 0.27,
    "P": 1.00,
    "Q": 0.53,
    "R": 0.38,
    "S": 0.24,
    "T": 0.32,
    "V": 0.39,
    "W": 0.25,
    "Y": 0.25,
}


def sanitize_sequence(seq: str) -> str:
    return "".join([aa for aa in (seq or "").upper() if aa in AA_SET])


def aa_counts(seq: str) -> Dict[str, int]:
    return {aa: seq.count(aa) for aa in AA_ORDER}


def fraction(seq_len: int, count: int) -> float:
    return (count / seq_len) if seq_len > 0 else 0.0


def cumulative_patch_length(seq: str, aa: str, min_len: int = 2) -> int:
    total = 0
    run = 0
    for ch in seq:
        if ch == aa:
            run += 1
        else:
            if run >= min_len:
                total += run
            run = 0
    if run >= min_len:
        total += run
    return total


def estimate_isoelectric_point(seq: str) -> float:
    counts = aa_counts(seq)
    if not seq:
        return 0.0

    pka = {
        "Cterm": 3.1,
        "Nterm": 8.0,
        "C": 8.5,
        "D": 3.9,
        "E": 4.1,
        "H": 6.5,
        "K": 10.8,
        "R": 12.0,
        "Y": 10.1,
    }

    def net_charge(ph: float) -> float:
        pos = (10 ** pka["Nterm"]) / (10 ** pka["Nterm"] + 10**ph)
        pos += counts["K"] * ((10 ** pka["K"]) / (10 ** pka["K"] + 10**ph))
        pos += counts["R"] * ((10 ** pka["R"]) / (10 ** pka["R"] + 10**ph))
        pos += counts["H"] * ((10 ** pka["H"]) / (10 ** pka["H"] + 10**ph))

        neg = (10**ph) / (10 ** pka["Cterm"] + 10**ph)
        neg += counts["D"] * ((10**ph) / (10 ** pka["D"] + 10**ph))
        neg += counts["E"] * ((10**ph) / (10 ** pka["E"] + 10**ph))
        neg += counts["C"] * ((10**ph) / (10 ** pka["C"] + 10**ph))
        neg += counts["Y"] * ((10**ph) / (10 ** pka["Y"] + 10**ph))
        return pos - neg

    low, high = 0.0, 14.0
    for _ in range(50):
        mid = (low + high) / 2
        if net_charge(mid) > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def compositional_feature_names() -> List[str]:
    names = []
    names.extend([f"comp_frac_{aa}" for aa in AA_ORDER])  # 20
    names.extend(
        [
            "comp_frac_polar",
            "comp_frac_hydrophobic",
            "comp_frac_aromatic",
            "comp_frac_KR",
            "comp_frac_DE",
            "comp_FCR",
            "comp_FCE",
            "comp_frac_disorder_promoting",
        ]
    )  # +8 = 28
    names.extend([f"comp_cum_patch_{aa}" for aa in AA_ORDER])  # +20 = 48
    names.extend(["comp_ratio_RK", "comp_ratio_ED"])  # +2 = 50
    names.extend(["comp_NCPR", "comp_pI", "comp_hydropathy_KD", "comp_PPII"])  # +4 = 54
    return names


def compositional_features(seq: str) -> List[float]:
    seq = sanitize_sequence(seq)
    n = len(seq)
    counts = aa_counts(seq)

    vals: List[float] = []
    vals.extend([fraction(n, counts[aa]) for aa in AA_ORDER])

    frac_polar = fraction(n, sum(counts[a] for a in GROUPS_8["POLAR"]))
    frac_hyd = fraction(n, sum(counts[a] for a in GROUPS_8["HYDROPHOBIC"]))
    frac_aro = fraction(n, sum(counts[a] for a in GROUPS_8["AROMATIC"]))
    frac_pos = fraction(n, counts["K"] + counts["R"])
    frac_neg = fraction(n, counts["D"] + counts["E"])
    fcr = fraction(n, counts["K"] + counts["R"] + counts["D"] + counts["E"])
    # Chain-expanding residues (simplified practical set).
    fce = fraction(n, counts["D"] + counts["E"] + counts["K"] + counts["R"] + counts["P"])
    frac_disorder = fraction(
        n,
        counts["A"]
        + counts["R"]
        + counts["G"]
        + counts["Q"]
        + counts["S"]
        + counts["E"]
        + counts["K"]
        + counts["P"],
    )
    vals.extend([frac_polar, frac_hyd, frac_aro, frac_pos, frac_neg, fcr, fce, frac_disorder])

    vals.extend([float(cumulative_patch_length(seq, aa=aa, min_len=2)) for aa in AA_ORDER])

    eps = 1e-8
    vals.extend(
        [
            (counts["R"] + eps) / (counts["K"] + eps),
            (counts["E"] + eps) / (counts["D"] + eps),
        ]
    )

    ncpr = fraction(n, (counts["K"] + counts["R"] - counts["D"] - counts["E"]))
    p_i = estimate_isoelectric_point(seq)
    hydropathy = sum(HYDROPATHY_KD[a] for a in seq) / n if n else 0.0
    ppii = sum(PPII_SCALE[a] for a in seq) / n if n else 0.0
    vals.extend([ncpr, p_i, hydropathy, ppii])
    return vals


def zscore_matrix(values: List[List[float]]) -> List[List[float]]:
    if not values:
        return []
    n_rows = len(values)
    n_cols = len(values[0])
    means = []
    stds = []
    for j in range(n_cols):
        col = [values[i][j] for i in range(n_rows)]
        m = sum(col) / n_rows
        v = sum((x - m) ** 2 for x in col) / n_rows
        s = math.sqrt(v)
        means.append(m)
        stds.append(s)
    out = []
    for row in values:
        out.append([(row[j] - means[j]) / stds[j] if stds[j] > 0 else 0.0 for j in range(n_cols)])
    return out


def patterning_feature_names() -> List[str]:
    labels = ["pol", "hyd", "pos", "neg", "aro", "ala", "pro", "gly"]
    names = []
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if j < i:
                continue
            names.append(f"pat_{li}_{lj}")
    return names


def count_residues_in_sequence(sequence: str, residue_types: Sequence[str]) -> int:
    total = 0
    for residue in residue_types:
        total += sequence.count(residue)
    return total


def get_kappa(seq: str, type1: Sequence[str], type2: Sequence[str]) -> float:
    blob_size = 5
    count1 = count_residues_in_sequence(seq, type1)
    count2 = count_residues_in_sequence(seq, type2)
    count1_frac = count1 / len(seq)
    count2_frac = count2 / len(seq)
    sig_all = (count1_frac - count2_frac) ** 2 / max((count1_frac + count2_frac), 1e-12)

    sig_x = []
    for x in range(0, len(seq) - blob_size + 2):
        subseq = seq[x : x + blob_size]
        ss_count1 = count_residues_in_sequence(subseq, type1)
        ss_count2 = count_residues_in_sequence(subseq, type2)
        ss_count1_frac = ss_count1 / blob_size
        ss_count2_frac = ss_count2 / blob_size
        if ss_count1 + ss_count2 == 0:
            sig_x.append(0.0)
        else:
            sig = (ss_count1_frac - ss_count2_frac) ** 2 / (ss_count1_frac + ss_count2_frac)
            sig_x.append(sig)

    asym = [(val - sig_all) ** 2 for val in sig_x]
    return float(np.mean(asym)) if asym else 0.0


def get_omega(seq: str, type1: Sequence[str]) -> float:
    blob_size = 5
    count = count_residues_in_sequence(seq, type1)
    count_frac = count / len(seq)
    sig_all = (count_frac - (1 - count_frac)) ** 2

    sig_x = []
    for x in range(0, len(seq) - blob_size + 2):
        subseq = seq[x : x + blob_size]
        ss_count = count_residues_in_sequence(subseq, type1)
        ss_count_frac = ss_count / blob_size
        sig = (ss_count_frac - (1 - ss_count_frac)) ** 2
        sig_x.append(sig)

    asym = [(val - sig_all) ** 2 for val in sig_x]
    return float(np.mean(asym)) if asym else 0.0


def get_org_seq_vals(myseq: str, typeall: Sequence[Sequence[str]], fracsall: Sequence[float]) -> np.ndarray:
    tlen = len(typeall)
    org = np.zeros((tlen, tlen))
    for i in range(tlen):
        type1 = typeall[i]
        for j in range(i, tlen):
            type2 = typeall[j]
            if type1 == type2 and fracsall[i] > 0.10:
                org[i, j] = get_omega(myseq, type1)
            if type1 != type2 and fracsall[i] > 0.10 and fracsall[j] > 0.10:
                org[i, j] = get_kappa(myseq, type1, type2)
    return org.reshape([1, tlen**2])


def get_scramble_seqs_vals(
    myseq: str,
    num_seqs: int,
    typeall: Sequence[Sequence[str]],
    fracsall: Sequence[float],
    random_seed: int,
) -> tuple[List[float], List[float]]:
    random.seed(random_seed)
    np.random.seed(random_seed)
    tlen = len(typeall)
    scr_vals = np.zeros((num_seqs, tlen**2))
    for i in range(num_seqs):
        curr = "".join(random.sample(myseq, len(myseq)))
        arr = np.zeros((tlen, tlen))
        for c1 in range(tlen):
            type1 = typeall[c1]
            for c2 in range(c1, tlen):
                type2 = typeall[c2]
                if type1 == type2 and fracsall[c1] > 0.10:
                    arr[c1, c2] = get_omega(curr, type1)
                if type1 != type2 and fracsall[c1] > 0.10 and fracsall[c2] > 0.10:
                    arr[c1, c2] = get_kappa(curr, type1, type2)
        scr_vals[i, : tlen**2] = arr.reshape([1, tlen**2])

    mean_vals = []
    var_vals = []
    for row in scr_vals.transpose():
        try:
            fit_alpha, fit_loc, fit_beta = stats.gamma.fit(row)
            mean_vals.append(float(stats.gamma.mean(fit_alpha, fit_loc, fit_beta)))
            var_vals.append(float(stats.gamma.var(fit_alpha, fit_loc, fit_beta)))
        except Exception:  # noqa: BLE001
            mean_vals.append(float(np.mean(row)))
            var_vals.append(float(np.var(row)))
    return mean_vals, var_vals


def patterning_36_features(seq: str, num_scrambles: int, seed: int) -> List[float]:
    seq = sanitize_sequence(seq)
    if len(seq) < 5:
        return [0.0] * 36

    fracsall = []
    for type1 in TYPEALL_8x8:
        c = count_residues_in_sequence(seq, type1)
        fracsall.append(c / len(seq))

    obs = get_org_seq_vals(seq, TYPEALL_8x8, fracsall)[0]
    mean_vals, var_vals = get_scramble_seqs_vals(seq, num_scrambles, TYPEALL_8x8, fracsall, seed)
    zvec = []
    for idx in range(len(obs)):
        if obs[idx] == 0 or var_vals[idx] <= 0:
            zvec.append(0.0)
        else:
            zvec.append((obs[idx] - mean_vals[idx]) / math.sqrt(var_vals[idx]))

    tlen = 8
    mat = [zvec[i * tlen : (i + 1) * tlen] for i in range(tlen)]
    out = []
    for i in range(tlen):
        for j in range(tlen):
            if j < i:
                continue
            out.append(float(mat[i][j]))
    return out


def ensure_table(conn: sqlite3.Connection, feature_cols: Sequence[str]) -> None:
    conn.execute("DROP TABLE IF EXISTS nardini90_features")
    cols = [
        '"sample_id" TEXT PRIMARY KEY',
        '"label" TEXT',
        '"feature_version" TEXT',
        '"random_seed" INTEGER',
        '"num_scrambles" INTEGER',
        '"computed_at" TEXT',
    ] + [f'"{c}" REAL' for c in feature_cols]
    conn.execute(f"CREATE TABLE nardini90_features ({', '.join(cols)})")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nardini90_label ON nardini90_features (label)")


def insert_rows(conn: sqlite3.Connection, rows: List[Dict[str, object]], cols: Sequence[str]) -> None:
    if not rows:
        return
    sql_cols = ", ".join([f'"{c}"' for c in cols])
    placeholders = ", ".join(["?"] * len(cols))
    sql = f"INSERT INTO nardini90_features ({sql_cols}) VALUES ({placeholders})"
    payload = [tuple(r.get(c, None) for c in cols) for r in rows]
    conn.executemany(sql, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute NARDINI+ 90D features for sample regions.")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--samples-file", default="")
    parser.add_argument("--sqlite-file", default="data/processed/idr_llps.sqlite")
    parser.add_argument("--output-tsv", default="data/processed/nardini90_features.tsv")
    parser.add_argument("--labels", default="idr_pos,neg")
    parser.add_argument("--num-scrambles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--feature-version", default="nardini90_v1")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    samples_file = Path(args.samples_file) if args.samples_file else Path(config["paths"]["samples_file"])
    samples = read_table(samples_file)
    wanted_labels = {x.strip() for x in args.labels.split(",") if x.strip()}
    samples = [s for s in samples if s.get("label", "") in wanted_labels]
    samples = [s for s in samples if sanitize_sequence(s.get("region_sequence", ""))]

    comp_names = compositional_feature_names()
    pat_names = patterning_feature_names()
    all_feature_names = comp_names + pat_names

    raw_comp = [compositional_features(s.get("region_sequence", "")) for s in samples]
    z_comp = zscore_matrix(raw_comp)

    rows_out: List[Dict[str, object]] = []
    for idx, s in enumerate(samples):
        pat = patterning_36_features(
            s.get("region_sequence", ""),
            num_scrambles=args.num_scrambles,
            seed=args.seed + idx,
        )
        vals = z_comp[idx] + pat
        row: Dict[str, object] = {
            "sample_id": s.get("sample_id", ""),
            "label": s.get("label", ""),
            "feature_version": args.feature_version,
            "random_seed": args.seed,
            "num_scrambles": args.num_scrambles,
            "computed_at": utc_timestamp(),
        }
        for name, val in zip(all_feature_names, vals):
            row[name] = float(val)
        rows_out.append(row)

    tsv_cols = [
        "sample_id",
        "label",
        "feature_version",
        "random_seed",
        "num_scrambles",
        "computed_at",
    ] + all_feature_names
    write_tsv(Path(args.output_tsv), rows_out, tsv_cols)

    conn = sqlite3.connect(Path(args.sqlite_file))
    try:
        ensure_table(conn, all_feature_names)
        insert_rows(conn, rows_out, tsv_cols)
        conn.commit()
    finally:
        conn.close()

    print(f"Computed NARDINI90 features for {len(rows_out)} samples")
    print(f"Wrote TSV: {args.output_tsv}")
    print(f"Wrote SQLite table: {args.sqlite_file}::nardini90_features")


if __name__ == "__main__":
    main()
