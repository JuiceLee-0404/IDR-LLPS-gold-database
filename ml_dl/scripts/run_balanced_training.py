from __future__ import annotations

import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    results = root / "ml_dl" / "results"
    results.mkdir(parents=True, exist_ok=True)

    run(
        [
            "python3",
            "-m",
            "src.modeling.build_balanced_dataset",
            "--sqlite-file",
            str(root / "data/processed/idr_llps.sqlite"),
            "--output-dir",
            str(root / "ml_dl/results"),
            "--seed",
            "42",
            "--test-size",
            "0.2",
        ]
    )

    train_file = str(results / "balanced_nardini90_train.tsv")
    test_file = str(results / "balanced_nardini90_test.tsv")

    run(
        [
            "python3",
            "-m",
            "src.modeling.train_ml_baselines",
            "--train-file",
            train_file,
            "--test-file",
            test_file,
            "--metrics-file",
            str(results / "ml_baselines_metrics.json"),
            "--preds-file",
            str(results / "ml_baselines_preds.tsv"),
        ]
    )

    run(
        [
            "python3",
            "-m",
            "src.modeling.train_deep_baseline",
            "--train-file",
            train_file,
            "--test-file",
            test_file,
            "--metrics-file",
            str(results / "deep_baseline_metrics.json"),
            "--preds-file",
            str(results / "deep_baseline_preds.tsv"),
        ]
    )

    print(f"Done. Results saved under: {results}")


if __name__ == "__main__":
    main()
