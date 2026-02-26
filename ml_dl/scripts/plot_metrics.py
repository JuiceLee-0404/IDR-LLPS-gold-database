from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    results = root / "ml_dl" / "results"
    results.mkdir(parents=True, exist_ok=True)

    ml_path = results / "ml_baselines_metrics.json"
    dl_path = results / "deep_baseline_metrics.json"
    if not ml_path.exists() or not dl_path.exists():
        raise FileNotFoundError("Please run run_balanced_training.py first.")

    ml = json.loads(ml_path.read_text(encoding="utf-8"))
    dl = json.loads(dl_path.read_text(encoding="utf-8"))

    rows = []
    for model, metrics in ml.items():
        for metric, value in metrics.items():
            rows.append({"model": model, "metric": metric, "value": value})
    for metric, value in dl.items():
        rows.append({"model": "mlp_baseline", "metric": metric, "value": value})
    df = pd.DataFrame(rows)

    plot_metrics = ["accuracy", "f1", "roc_auc", "pr_auc", "balanced_accuracy"]
    p = df[df["metric"].isin(plot_metrics)].copy()
    p["metric"] = pd.Categorical(p["metric"], categories=plot_metrics, ordered=True)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=p, x="metric", y="value", hue="model", ax=ax)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Baseline Model Metrics Comparison")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(results / "metrics_barplot.png", dpi=220)
    fig.savefig(results / "metrics_barplot.svg")
    plt.close(fig)

    ranking = p[p["metric"] == "roc_auc"].sort_values("value", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=ranking, x="model", y="value", ax=ax, palette="viridis")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Model Ranking by ROC-AUC")
    ax.set_xlabel("Model")
    ax.set_ylabel("ROC-AUC")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(results / "model_ranking.png", dpi=220)
    fig.savefig(results / "model_ranking.svg")
    plt.close(fig)

    summary_lines = [
        "# Baseline Metrics Summary",
        "",
        "## ROC-AUC ranking",
    ]
    for _, r in ranking.iterrows():
        summary_lines.append(f"- {r['model']}: {r['value']:.4f}")
    (results / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Wrote plots and summary to: {results}")


if __name__ == "__main__":
    main()
