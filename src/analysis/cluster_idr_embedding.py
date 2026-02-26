from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def load_labeled_feature_df(sqlite_file: Path) -> pd.DataFrame:
    conn = sqlite3.connect(sqlite_file)
    try:
        q = """
        SELECT
          s.sample_id,
          s.label AS sample_label,
          s.label_source,
          n.feature_version,
          n.random_seed,
          n.num_scrambles,
          n.computed_at,
          n.*
        FROM samples s
        JOIN nardini90_features n ON s.sample_id = n.sample_id
        WHERE s.label IN ('idr_pos', 'neg')
        """
        df = pd.read_sql_query(q, conn)
    finally:
        conn.close()
    df = df.loc[:, ~df.columns.duplicated()]
    # Three-way group: pos, neg (exp_neg), neg_pseudo (pseudo_neg)
    df["group"] = "neg_pseudo"
    df.loc[df["sample_label"] == "idr_pos", "group"] = "pos"
    df.loc[(df["sample_label"] == "neg") & (df["label_source"].fillna("") == "exp_neg"), "group"] = "neg"
    return df


def save_scatter(df: pd.DataFrame, x: str, y: str, hue: str, title: str, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, s=20, alpha=0.75, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=220)
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster IDR features and visualize PCA/t-SNE embeddings.")
    parser.add_argument("--sqlite-file", default="data/processed/idr_llps.sqlite")
    parser.add_argument("--output-dir", default="reports/figures")
    parser.add_argument("--metrics-file", default="reports/idr_cluster_metrics.json")
    parser.add_argument("--k", type=int, default=3, help="KMeans clusters (3 for pos/neg/neg_pseudo)")
    parser.add_argument("--pos-neg-only", action="store_true", help="Only pos+neg (exclude neg_pseudo); k=2, outputs *_pos_neg_only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tsne-perplexity", type=float, default=35.0)
    parser.add_argument("--umap-n-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.15)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_labeled_feature_df(Path(args.sqlite_file))
    pos_neg_only = args.pos_neg_only
    if pos_neg_only:
        df = df[df["group"].isin(["pos", "neg"])].copy()
        k = 2
        suffix = "_pos_neg_only"
        metrics_path = Path(args.metrics_file)
        metrics_file = metrics_path.parent / (metrics_path.stem + "_pos_neg_only.json")
    else:
        k = args.k
        suffix = ""
        metrics_file = Path(args.metrics_file)

    feat_cols = [c for c in df.columns if c.startswith("comp_") or c.startswith("pat_")]
    x = df[feat_cols].astype(float).values
    if pos_neg_only:
        group_to_int = {"pos": 0, "neg": 1}
    else:
        group_to_int = {"pos": 0, "neg": 1, "neg_pseudo": 2}
    y_true = df["group"].map(group_to_int).values

    x_scaled = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2, random_state=args.seed)
    pca_xy = pca.fit_transform(x_scaled)
    pca_df = pd.DataFrame(
        {
            "sample_id": df["sample_id"],
            "group": df["group"],
            "pca1": pca_xy[:, 0],
            "pca2": pca_xy[:, 1],
        }
    )
    save_scatter(
        pca_df,
        "pca1",
        "pca2",
        "group",
        "PCA of IDR NARDINI90 Features (pos / neg / neg_pseudo)" if not pos_neg_only else "PCA of IDR NARDINI90 Features (pos / neg only)",
        out_dir / ("idr_pca_pos_neg" + suffix),
    )

    tsne = TSNE(
        n_components=2,
        random_state=args.seed,
        perplexity=args.tsne_perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=1200,
    )
    tsne_xy = tsne.fit_transform(x_scaled)
    tsne_df = pd.DataFrame(
        {
            "sample_id": df["sample_id"],
            "group": df["group"],
            "tsne1": tsne_xy[:, 0],
            "tsne2": tsne_xy[:, 1],
        }
    )
    save_scatter(
        tsne_df,
        "tsne1",
        "tsne2",
        "group",
        "t-SNE of IDR NARDINI90 Features (pos / neg / neg_pseudo)" if not pos_neg_only else "t-SNE of IDR NARDINI90 Features (pos / neg only)",
        out_dir / ("idr_tsne_pos_neg" + suffix),
    )

    km = KMeans(n_clusters=k, random_state=args.seed, n_init=20)
    cluster_ids = km.fit_predict(x_scaled)
    pca_df["cluster"] = cluster_ids.astype(str)
    tsne_df["cluster"] = cluster_ids.astype(str)

    save_scatter(
        pca_df,
        "pca1",
        "pca2",
        "cluster",
        "PCA with KMeans Clusters (k={})".format(k),
        out_dir / ("idr_pca_kmeans" + suffix),
    )
    save_scatter(
        tsne_df,
        "tsne1",
        "tsne2",
        "cluster",
        "t-SNE with KMeans Clusters (k={})".format(k),
        out_dir / ("idr_tsne_kmeans" + suffix),
    )

    metrics = {
        "n_samples": int(len(df)),
        "n_features": int(len(feat_cols)),
        "group_counts": df["group"].value_counts().to_dict(),
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_[:2]],
        "kmeans_k": int(k),
        "ari_group_vs_kmeans": float(adjusted_rand_score(y_true, cluster_ids)),
        "nmi_group_vs_kmeans": float(normalized_mutual_info_score(y_true, cluster_ids)),
        "silhouette_kmeans": float(silhouette_score(x_scaled, cluster_ids)),
    }
    if HAS_UMAP:
        metrics["umap_params"] = {
            "n_neighbors": int(args.umap_n_neighbors),
            "min_dist": float(args.umap_min_dist),
        }
    metrics_path = metrics_file
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote metrics: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
