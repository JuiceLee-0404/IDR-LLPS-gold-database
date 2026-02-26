from __future__ import annotations

import argparse
import ssl
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from src.common.pipeline_utils import ensure_dirs, load_yaml, utc_date, write_json, write_tsv


DATA_SOURCES = [
    {
        "name": "llpsdb",
        "url": "http://bio-comp.org.cn/llpsdb/help.html",
        "filename": "help.html",
        "db_version": "unknown",
    },
    {
        "name": "phasepro",
        "url": "https://phasepro.elte.hu/help",
        "filename": "help.html",
        "db_version": "unknown",
    },
    {
        "name": "drllps",
        "url": "https://llps.biocuckoo.cn/download.php",
        "filename": "download.html",
        "db_version": "unknown",
    },
    {
        "name": "disprot",
        "url": "https://disprot.org/download",
        "filename": "download.html",
        "db_version": "unknown",
    },
    {
        "name": "mobidb",
        "url": "https://mobidb.org/",
        "filename": "home.html",
        "db_version": "unknown",
    },
]


def fetch_url(url: str, timeout: int, verify_ssl: bool = True) -> tuple[bool, str, int]:
    context = ssl.create_default_context() if verify_ssl else ssl._create_unverified_context()
    with urlopen(url, timeout=timeout, context=context) as response:
        content = response.read().decode("utf-8", errors="replace")
        code = response.getcode() or 200
        return True, content, code


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch publicly available LLPS/IDR database pages into data/raw."
    )
    parser.add_argument(
        "--config",
        default="configs/dataset.yaml",
        help="Path to pipeline config file.",
    )
    parser.add_argument(
        "--timeout",
        default=30,
        type=int,
        help="Network timeout in seconds for each source.",
    )
    parser.add_argument(
        "--disable-ssl-fallback",
        action="store_true",
        help="Disable insecure fallback when SSL verification fails.",
    )
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    raw_dir = Path(config["paths"]["raw_dir"])
    ensure_dirs([raw_dir])

    manifest_rows = []
    for source in DATA_SOURCES:
        source_dir = raw_dir / source["name"]
        ensure_dirs([source_dir])
        out_file = source_dir / source["filename"]

        ok = False
        status_code = 0
        message = ""
        content = ""
        ssl_mode = "verified"
        try:
            ok, content, status_code = fetch_url(
                source["url"], timeout=args.timeout, verify_ssl=True
            )
            out_file.write_text(content, encoding="utf-8")
            message = "downloaded"
        except HTTPError as exc:
            status_code = exc.code
            message = f"http_error: {exc.reason}"
        except URLError as exc:
            reason = str(exc.reason)
            if "CERTIFICATE_VERIFY_FAILED" in reason and not args.disable_ssl_fallback:
                try:
                    ok, content, status_code = fetch_url(
                        source["url"], timeout=args.timeout, verify_ssl=False
                    )
                    out_file.write_text(content, encoding="utf-8")
                    ssl_mode = "insecure_fallback"
                    message = "downloaded_with_insecure_ssl_fallback"
                except URLError as fallback_exc:
                    message = f"url_error: {fallback_exc.reason}"
                except Exception as fallback_exc:  # noqa: BLE001
                    message = f"unexpected_error: {fallback_exc}"
            else:
                message = f"url_error: {exc.reason}"
        except Exception as exc:  # noqa: BLE001
            message = f"unexpected_error: {exc}"

        manifest_rows.append(
            {
                "source_db": source["name"],
                "url": source["url"],
                "raw_file": str(out_file),
                "status": "ok" if ok else "failed",
                "status_code": str(status_code),
                "message": message,
                "db_version": source["db_version"],
                "download_date": utc_date(),
                "bytes": str(len(content.encode("utf-8")) if content else 0),
                "ssl_mode": ssl_mode,
            }
        )

    manifest_path = raw_dir / "raw_manifest.tsv"
    write_tsv(
        manifest_path,
        manifest_rows,
        fieldnames=[
            "source_db",
            "url",
            "raw_file",
            "status",
            "status_code",
            "message",
            "db_version",
            "download_date",
            "bytes",
            "ssl_mode",
        ],
    )
    write_json(raw_dir / "raw_manifest.json", {"sources": manifest_rows})
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
