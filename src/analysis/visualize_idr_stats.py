from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.common.pipeline_utils import load_yaml, read_table


AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA_GROUPS = {
    "polar": set("STNQCH"),
    "hydrophobic": set("ILMV"),
    "charged": set("RKED"),
    "aromatic": set("FWY"),
    "proline": {"P"},
    "glycine": {"G"},
}


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_seq(seq: str) -> str:
    seq = (seq or "").upper()
    return "".join([x for x in seq if x in AA20])


def slice_idr(sequence: str, start: int, end: int) -> str:
    s = sanitize_seq(sequence)
    if not s:
        return ""
    if start <= 0 or end <= 0 or end < start:
        return ""
    left = max(start - 1, 0)
    right = min(end, len(s))
    if right <= left:
        return ""
    return s[left:right]


def seq_aa_freq(df: pd.DataFrame, seq_col: str) -> pd.Series:
    total = {aa: 0 for aa in AA20}
    length = 0
    for seq in df[seq_col].fillna(""):
        s = sanitize_seq(seq)
        length += len(s)
        for aa in AA20:
            total[aa] += s.count(aa)
    if length == 0:
        return pd.Series({aa: 0.0 for aa in AA20}, dtype=float)
    return pd.Series({aa: total[aa] / length for aa in AA20}, dtype=float)


def group_feature_df(df: pd.DataFrame, seq_col: str) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        s = sanitize_seq(r.get(seq_col, ""))
        n = len(s)
        if n == 0:
            continue
        pos = s.count("K") + s.count("R")
        neg = s.count("D") + s.count("E")
        rows.append(
            {
                "sample_id": r.get("sample_id", ""),
                "label": r.get("label", ""),
                "group": r.get("group", r.get("label", "")),
                "fcr": (pos + neg) / n,
                "ncpr": (pos - neg) / n,
                **{
                    f"group_{name}": sum(s.count(aa) for aa in aas) / n
                    for name, aas in AA_GROUPS.items()
                },
            }
        )
    return pd.DataFrame(rows)


def load_tables(config_path: Path, sqlite_path: Path) -> Dict[str, pd.DataFrame]:
    cfg = load_yaml(config_path)
    idr_regions = pd.DataFrame(read_table(Path(cfg["paths"]["idr_regions_file"])))
    samples = pd.DataFrame(read_table(Path(cfg["paths"]["samples_file"])))
    evidence = pd.DataFrame(read_table(Path(cfg["paths"]["evidence_file"])))
    splits = pd.DataFrame(read_table(Path(cfg["paths"]["splits_file"])))

    if not idr_regions.empty:
        idr_regions["idr_start"] = pd.to_numeric(idr_regions["idr_start"], errors="coerce").fillna(0).astype(int)
        idr_regions["idr_end"] = pd.to_numeric(idr_regions["idr_end"], errors="coerce").fillna(0).astype(int)
        idr_regions["idr_len"] = (idr_regions["idr_end"] - idr_regions["idr_start"] + 1).clip(lower=0)
        idr_regions["idr_seq"] = idr_regions.apply(
            lambda x: slice_idr(x.get("sequence", ""), int(x["idr_start"]), int(x["idr_end"])), axis=1
        )

    if not samples.empty:
        samples["region_start"] = pd.to_numeric(samples["region_start"], errors="coerce").fillna(0).astype(int)
        samples["region_end"] = pd.to_numeric(samples["region_end"], errors="coerce").fillna(0).astype(int)
        samples["region_len"] = (samples["region_end"] - samples["region_start"] + 1).clip(lower=0)
        samples["region_sequence"] = samples["region_sequence"].fillna("").map(sanitize_seq)
        # Three-way group: pos (idr_pos), neg (exp_neg), neg_pseudo (pseudo_neg)
        samples["group"] = "neg_pseudo"
        samples.loc[samples["label"] == "idr_pos", "group"] = "pos"
        samples.loc[
            (samples["label"] == "neg") & (samples["label_source"].fillna("") == "exp_neg"),
            "group",
        ] = "neg"

    nardini = pd.DataFrame()
    if sqlite_path.exists():
        conn = sqlite3.connect(sqlite_path)
        try:
            nardini = pd.read_sql_query("SELECT * FROM nardini90_features", conn)
        except Exception:
            nardini = pd.DataFrame()
        finally:
            conn.close()

    return {
        "idr_regions": idr_regions,
        "samples": samples,
        "evidence": evidence,
        "splits": splits,
        "nardini": nardini,
    }


def plot_length_distributions(idr: pd.DataFrame, samples: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(idr["idr_len"], bins=50, kde=True, ax=ax, color="#2a9d8f")
    ax.set_title("Global IDR Length Distribution")
    ax.set_xlabel("IDR Length")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out / "length_distribution_idr_regions.png", dpi=200)
    fig.savefig(out / "length_distribution_idr_regions.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(samples, x="region_len", hue="group", bins=50, kde=True, ax=ax, element="step", stat="density")
    ax.set_title("Sample Region Length Distribution (pos / neg / neg_pseudo)")
    ax.set_xlabel("Region Length")
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(out / "length_distribution_samples_by_label.png", dpi=200)
    fig.savefig(out / "length_distribution_samples_by_label.svg")
    plt.close(fig)

    bins = pd.cut(idr["idr_len"], bins=[0, 30, 100, 10_000], labels=["short", "medium", "long"], include_lowest=True)
    bin_counts = bins.value_counts().reindex(["short", "medium", "long"]).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=bin_counts.index, y=bin_counts.values, ax=ax, palette="viridis")
    ax.set_title("Global IDR Length Bins")
    ax.set_xlabel("Length Bin")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out / "length_bins_idr_regions.png", dpi=200)
    fig.savefig(out / "length_bins_idr_regions.svg")
    plt.close(fig)


def plot_amino_acid_preferences(idr: pd.DataFrame, samples: pd.DataFrame, out: Path) -> None:
    aa_all = seq_aa_freq(idr, "idr_seq").reset_index()
    aa_all.columns = ["aa", "freq"]
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=aa_all, x="aa", y="freq", ax=ax, color="#3a86ff")
    ax.set_title("Global IDR Amino Acid Frequency (20 AA)")
    ax.set_xlabel("Amino Acid")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(out / "aa_frequency_idr_regions.png", dpi=200)
    fig.savefig(out / "aa_frequency_idr_regions.svg")
    plt.close(fig)

    pos = samples[samples["group"] == "pos"]
    neg = samples[samples["group"] == "neg"]
    neg_pseudo = samples[samples["group"] == "neg_pseudo"]
    pos_freq = seq_aa_freq(pos, "region_sequence")
    neg_freq = seq_aa_freq(neg, "region_sequence")
    neg_pseudo_freq = seq_aa_freq(neg_pseudo, "region_sequence")
    l2fc = ((pos_freq + 1e-8) / (neg_freq + 1e-8)).map(lambda x: np.log2(x))
    l2fc_df = l2fc.reset_index()
    l2fc_df.columns = ["aa", "log2_fc_pos_vs_neg"]
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=l2fc_df, x="aa", y="log2_fc_pos_vs_neg", ax=ax, palette="coolwarm")
    ax.axhline(0, linestyle="--", linewidth=1, color="black")
    ax.set_title("Amino Acid Preference (log2FC, pos vs neg)")
    ax.set_xlabel("Amino Acid")
    ax.set_ylabel("log2(pos/neg)")
    fig.tight_layout()
    fig.savefig(out / "aa_log2fc_pos_vs_neg.png", dpi=200)
    fig.savefig(out / "aa_log2fc_pos_vs_neg.svg")
    plt.close(fig)

    l2fc_pseudo = ((pos_freq + 1e-8) / (neg_pseudo_freq + 1e-8)).map(lambda x: np.log2(x))
    l2fc_pseudo_df = l2fc_pseudo.reset_index()
    l2fc_pseudo_df.columns = ["aa", "log2_fc_pos_vs_neg_pseudo"]
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=l2fc_pseudo_df, x="aa", y="log2_fc_pos_vs_neg_pseudo", ax=ax, palette="coolwarm")
    ax.axhline(0, linestyle="--", linewidth=1, color="black")
    ax.set_title("Amino Acid Preference (log2FC, pos vs neg_pseudo)")
    ax.set_xlabel("Amino Acid")
    ax.set_ylabel("log2(pos/neg_pseudo)")
    fig.tight_layout()
    fig.savefig(out / "aa_log2fc_pos_vs_neg_pseudo.png", dpi=200)
    fig.savefig(out / "aa_log2fc_pos_vs_neg_pseudo.svg")
    plt.close(fig)

    gf = group_feature_df(samples, "region_sequence")
    melted = gf.melt(
        id_vars=["sample_id", "label", "group"],
        value_vars=[f"group_{k}" for k in AA_GROUPS.keys()],
        var_name="group_name",
        value_name="fraction",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=melted, x="group_name", y="fraction", hue="group", ax=ax)
    ax.set_title("Residue Group Fractions by Group (pos / neg / neg_pseudo)")
    ax.set_xlabel("Residue Group")
    ax.set_ylabel("Fraction")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out / "aa_group_fraction_by_label.png", dpi=200)
    fig.savefig(out / "aa_group_fraction_by_label.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.violinplot(data=gf, x="group", y="fcr", ax=axes[0], inner="quartile")
    axes[0].set_title("FCR Distribution")
    axes[0].set_xlabel("Group")
    axes[0].set_ylabel("FCR")
    sns.violinplot(data=gf, x="group", y="ncpr", ax=axes[1], inner="quartile")
    axes[1].set_title("NCPR Distribution")
    axes[1].set_xlabel("Group")
    axes[1].set_ylabel("NCPR")
    fig.tight_layout()
    fig.savefig(out / "charge_features_by_label.png", dpi=200)
    fig.savefig(out / "charge_features_by_label.svg")
    plt.close(fig)


def plot_species_and_sources(idr: pd.DataFrame, samples: pd.DataFrame, evidence: pd.DataFrame, out: Path) -> None:
    top_species = idr["organism"].fillna("unknown").value_counts().head(20).reset_index()
    top_species.columns = ["organism", "count"]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_species, y="organism", x="count", ax=ax, color="#8ecae6")
    ax.set_title("Top 20 Species by Global IDR Count")
    ax.set_xlabel("Count")
    ax.set_ylabel("Species")
    fig.tight_layout()
    fig.savefig(out / "species_top20_idr_regions.png", dpi=200)
    fig.savefig(out / "species_top20_idr_regions.svg")
    plt.close(fig)

    tg = samples["taxon_group"].fillna("unknown").value_counts().reset_index()
    tg.columns = ["taxon_group", "count"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=tg, x="taxon_group", y="count", ax=ax, palette="Set2")
    ax.set_title("taxon_group Distribution (samples)")
    ax.set_xlabel("taxon_group")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out / "taxon_group_samples.png", dpi=200)
    fig.savefig(out / "taxon_group_samples.svg")
    plt.close(fig)

    src = evidence["source_db"].fillna("unknown").value_counts().reset_index()
    src.columns = ["source_db", "count"]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=src, x="source_db", y="count", ax=ax, palette="muted")
    ax.set_title("Evidence Source Contribution")
    ax.set_xlabel("source_db")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out / "source_db_contribution.png", dpi=200)
    fig.savefig(out / "source_db_contribution.svg")
    plt.close(fig)


def plot_labels_quality_and_split(samples: pd.DataFrame, splits: pd.DataFrame, out: Path) -> Dict[str, int]:
    group_counts = samples["group"].value_counts().reindex(["pos", "neg", "neg_pseudo"]).fillna(0).reset_index()
    group_counts.columns = ["group", "count"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=group_counts, x="group", y="count", ax=ax, palette="Set1")
    ax.set_title("Sample Group Counts (pos / neg / neg_pseudo)")
    ax.set_xlabel("group")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out / "group_counts.png", dpi=200)
    fig.savefig(out / "group_counts.svg")
    plt.close(fig)

    label_counts = samples["label"].value_counts().reset_index()
    label_counts.columns = ["label", "count"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=label_counts, x="label", y="count", ax=ax, palette="Set1")
    ax.set_title("Sample Label Counts")
    ax.set_xlabel("label")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out / "label_counts.png", dpi=200)
    fig.savefig(out / "label_counts.svg")
    plt.close(fig)

    ls = samples["label_source"].fillna("unknown").value_counts().reset_index()
    ls.columns = ["label_source", "count"]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=ls, x="label_source", y="count", ax=ax, palette="pastel")
    ax.set_title("label_source Distribution")
    ax.set_xlabel("label_source")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out / "label_source_counts.png", dpi=200)
    fig.savefig(out / "label_source_counts.svg")
    plt.close(fig)

    merged = samples.merge(splits, on="sample_id", how="left")
    pivot = merged.pivot_table(index="split", columns="label", values="sample_id", aggfunc="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title("Label Distribution Across train/val/test")
    ax.set_xlabel("split")
    ax.set_ylabel("Sample Count")
    fig.tight_layout()
    fig.savefig(out / "split_label_distribution.png", dpi=200)
    fig.savefig(out / "split_label_distribution.svg")
    plt.close(fig)

    empty_seq = int((samples["region_sequence"].fillna("").str.len() == 0).sum())
    bad_region = int(((samples["region_start"] <= 0) | (samples["region_end"] < samples["region_start"])).sum())
    return {"empty_region_sequence": empty_seq, "bad_region_coordinates": bad_region}


def plot_nardini_features(samples: pd.DataFrame, nardini: pd.DataFrame, out: Path) -> None:
    if nardini.empty:
        return
    merged = samples[["sample_id", "label", "group"]].merge(nardini, on=["sample_id", "label"], how="inner")
    if merged.empty:
        return

    feature_cols = [c for c in merged.columns if c.startswith("comp_") or c.startswith("pat_")]
    means = merged.groupby("group")[feature_cols].mean().reindex(["pos", "neg", "neg_pseudo"]).fillna(0)
    # Top 12 by |pos - neg| difference
    delta_pos_neg = (means.loc["pos"] - means.loc["neg"]).abs()
    top = delta_pos_neg.sort_values(ascending=False).head(12).index.tolist()
    top_df = means[top].T.reset_index().rename(columns={"index": "feature"})
    top_df = top_df.melt(id_vars="feature", var_name="group", value_name="mean_val")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_df, y="feature", x="mean_val", hue="group", ax=ax)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Top 12 NARDINI90 Features by |pos−neg| (pos / neg / neg_pseudo means)")
    ax.set_xlabel("Mean feature value")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(out / "nardini90_top12_delta.png", dpi=200)
    fig.savefig(out / "nardini90_top12_delta.svg")
    plt.close(fig)


def generate_summary(
    tables: Dict[str, pd.DataFrame], qc: Dict[str, int], out_report: Path, fig_dir: Path
) -> None:
    idr = tables["idr_regions"]
    samples = tables["samples"]
    evidence = tables["evidence"]
    nardini = tables["nardini"]
    top_species = idr["organism"].fillna("unknown").value_counts().head(10)
    top_aa = seq_aa_freq(idr, "idr_seq").sort_values(ascending=False).head(10)
    label_ratio = samples["label"].value_counts(normalize=True).mul(100).round(2)

    lines = []
    lines.append("# IDR数据库统计可视化报告")
    lines.append("")
    lines.append("## 数据摘要")
    lines.append(f"- 全IDR条目数：{len(idr)}")
    lines.append(f"- 样本条目数：{len(samples)}")
    lines.append(f"- 证据条目数：{len(evidence)}")
    if not nardini.empty:
        lines.append(f"- NARDINI90条目数：{len(nardini)}")
    lines.append("")
    lines.append("## 标签比例（samples）")
    for k, v in label_ratio.items():
        lines.append(f"- {k}: {v}%")
    lines.append("")
    group_ratio = samples["group"].value_counts(normalize=True).mul(100).round(2)
    lines.append("## 三组比例（pos / neg / neg_pseudo）")
    for k, v in group_ratio.items():
        lines.append(f"- {k}: {v}%")
    lines.append("")
    lines.append("## 关键发现")
    lines.append("- 长度分布：IDR长度与样本区段长度均呈右偏分布，存在长尾。")
    lines.append("- 氨基酸偏好：给出全IDR的20AA频率和正负样本log2FC差异图。")
    lines.append("- 物种构成：列出Top物种与taxon_group分布。")
    lines.append("")
    lines.append("### Top10 物种")
    for org, n in top_species.items():
        lines.append(f"- {org}: {n}")
    lines.append("")
    lines.append("### Top10 高频氨基酸（全IDR）")
    for aa, f in top_aa.items():
        lines.append(f"- {aa}: {f:.4f}")
    lines.append("")
    lines.append("## 质量控制")
    lines.append(f"- 空区段序列条数：{qc['empty_region_sequence']}")
    lines.append(f"- 异常坐标条数：{qc['bad_region_coordinates']}")
    lines.append("")
    lines.append("## 图表目录")
    for p in sorted(fig_dir.glob("*.png")):
        lines.append(f"- `reports/figures/{p.name}`")
    out_report.write_text("\n".join(lines), encoding="utf-8")


def run_all(config: str, sqlite: str, output_dir: str) -> Tuple[Path, Path]:
    sns.set_theme(style="whitegrid")
    out_dir = Path(output_dir)
    fig_dir = out_dir / "figures"
    safe_mkdir(fig_dir)
    tables = load_tables(Path(config), Path(sqlite))

    plot_length_distributions(tables["idr_regions"], tables["samples"], fig_dir)
    plot_amino_acid_preferences(tables["idr_regions"], tables["samples"], fig_dir)
    plot_species_and_sources(tables["idr_regions"], tables["samples"], tables["evidence"], fig_dir)
    qc = plot_labels_quality_and_split(tables["samples"], tables["splits"], fig_dir)
    plot_nardini_features(tables["samples"], tables["nardini"], fig_dir)

    report = out_dir / "idr_stats_report.md"
    generate_summary(tables, qc, report, fig_dir)
    return fig_dir, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate detailed IDR statistics visualizations.")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--sqlite-file", default="data/processed/idr_llps.sqlite")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    fig_dir, report = run_all(config=args.config, sqlite=args.sqlite_file, output_dir=args.output_dir)
    print(f"Wrote figures to: {fig_dir}")
    print(f"Wrote report: {report}")


if __name__ == "__main__":
    main()
