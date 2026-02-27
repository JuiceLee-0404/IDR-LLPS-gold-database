from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse paired TAIR LLPS CSV into long-format labeled table."
    )
    parser.add_argument(
        "--input-csv",
        default="ml_dl/相分离蛋白汇总整理.csv",
        help="Path to the original TAIR LLPS CSV file.",
    )
    parser.add_argument(
        "--output-tsv",
        default="data/validation/tair_llps_labels.tsv",
        help="Path to write long-format TAIR labels TSV.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    in_path = root / args.input_csv
    out_path = root / args.output_tsv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, encoding="utf-8-sig")

    # Expect header like: Tair-ID,定位,是否相分离,系统,,,,Tair-ID,定位,是否相分离,系统
    # pandas will typically rename the second block as *.1
    left_cols = ["Tair-ID", "定位", "是否相分离", "系统"]
    right_cols = ["Tair-ID.1", "定位.1", "是否相分离.1", "系统.1"]

    for col in left_cols + right_cols:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in {in_path}")

    rows = []
    for idx, row in df.iterrows():
        pair_index = int(idx)

        # left (expected positive, 是否相分离 = 是)
        left_id = str(row[left_cols[0]]).strip()
        left_loc = str(row[left_cols[1]]).strip()
        left_label_raw = str(row[left_cols[2]]).strip()
        left_sys = str(row[left_cols[3]]).strip()
        if left_id and left_id != "nan":
            llps_label = 1 if left_label_raw == "是" else 0
            rows.append(
                {
                    "tair_id": left_id,
                    "location": left_loc,
                    "llps_label": llps_label,
                    "system": left_sys,
                    "pair_index": pair_index,
                    "side": "left",
                }
            )

        # right (expected negative, 是否相分离 = 否)
        right_id = str(row[right_cols[0]]).strip()
        right_loc = str(row[right_cols[1]]).strip()
        right_label_raw = str(row[right_cols[2]]).strip()
        right_sys = str(row[right_cols[3]]).strip()
        if right_id and right_id != "nan":
            llps_label = 1 if right_label_raw == "是" else 0
            rows.append(
                {
                    "tair_id": right_id,
                    "location": right_loc,
                    "llps_label": llps_label,
                    "system": right_sys,
                    "pair_index": pair_index,
                    "side": "right",
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()

