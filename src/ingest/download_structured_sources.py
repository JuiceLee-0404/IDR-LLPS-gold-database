from __future__ import annotations

import argparse
import csv
import json
import ssl
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from src.common.pipeline_utils import ensure_dirs, load_yaml, utc_date, write_tsv


STRUCTURED_SOURCES = [
    {
        "name": "drllps",
        "url": "https://llps.biocuckoo.cn/download/LLPS.txt",
        "filename": "LLPS.txt",
        "format": "tsv",
    },
    {
        "name": "disprot",
        "url": (
            "https://disprot.org/api/search?release=2025_06&show_ambiguous=true"
            "&show_obsolete=false&format=tsv&namespace=all&get_consensus=false"
        ),
        "filename": "disprot_regions.tsv",
        "format": "tsv",
    },
    {
        "name": "phasepro",
        "url": "https://phasepro.elte.hu/download_full.json",
        "filename": "download_full.json",
        "format": "json",
    },
    {
        "name": "llpsdatasets",
        "url": "https://llpsdatasets.ppmclab.com/downloads/datasets.tsv",
        "filename": "datasets.tsv",
        "format": "tsv",
        "fallback_url": "https://raw.githubusercontent.com/PPMC-lab/llps-datasets/main/datasets.tsv",
    },
]


def fetch(url: str, timeout: int = 60) -> str:
    try:
        with urlopen(url, timeout=timeout, context=ssl.create_default_context()) as r:
            return r.read().decode("utf-8", errors="replace")
    except URLError as exc:
        reason = str(exc.reason)
        if "CERTIFICATE_VERIFY_FAILED" not in reason:
            raise
    with urlopen(url, timeout=timeout, context=ssl._create_unverified_context()) as r:
        return r.read().decode("utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def count_rows_tsv(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return max(sum(1 for _ in csv.reader(f, delimiter="\t")) - 1, 0)


def count_rows_json(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return len(payload)
    if isinstance(payload, list):
        return len(payload)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Download structured source exports.")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    raw_dir = Path(config["paths"]["raw_dir"])
    ensure_dirs([raw_dir])

    manifest = []
    for src in STRUCTURED_SOURCES:
        source_dir = raw_dir / src["name"]
        ensure_dirs([source_dir])
        out_file = source_dir / src["filename"]
        status = "ok"
        message = "downloaded"
        row_count = 0
        try:
            try:
                content = fetch(src["url"], timeout=args.timeout)
            except Exception:  # noqa: BLE001
                if src.get("fallback_url"):
                    content = fetch(src["fallback_url"], timeout=args.timeout)
                    message = "downloaded_via_fallback"
                else:
                    raise
            write_text(out_file, content)
            if src["format"] == "tsv":
                row_count = count_rows_tsv(out_file)
            elif src["format"] == "json":
                row_count = count_rows_json(out_file)
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            message = str(exc)

        manifest.append(
            {
                "source_db": src["name"],
                "url": src["url"],
                "file_path": str(out_file),
                "status": status,
                "rows": str(row_count),
                "message": message,
                "download_date": utc_date(),
            }
        )

    write_tsv(
        raw_dir / "structured_manifest.tsv",
        manifest,
        ["source_db", "url", "file_path", "status", "rows", "message", "download_date"],
    )
    print(f"Wrote structured manifest with {len(manifest)} sources")


if __name__ == "__main__":
    main()
