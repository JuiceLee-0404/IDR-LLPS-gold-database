# ML/DL Workspace

This folder stores machine learning / deep learning scripts and outputs for the IDR dataset.

## Structure

- `scripts/run_balanced_training.py`: build balanced dataset and run ML/DL baselines.
- `scripts/plot_metrics.py`: visualize baseline metrics as figures.
- `results/`: metrics JSON files, plots, and summary.

## Usage

From project root:

```bash
python -m ml_dl.scripts.run_balanced_training
python -m ml_dl.scripts.plot_metrics
```

Outputs:

- `ml_dl/results/ml_baselines_metrics.json`
- `ml_dl/results/deep_baseline_metrics.json`
- `ml_dl/results/metrics_barplot.png`
- `ml_dl/results/metrics_barplot.svg`
- `ml_dl/results/model_ranking.png`
- `ml_dl/results/model_ranking.svg`
