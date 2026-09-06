"""
One-time setup: fetch the DB-IP Lite GeoIP databases into data/geoip/.

The problem statement calls for an "open source downloadable Geo IP database".
DB-IP Lite (https://db-ip.com/db/) is published monthly under CC BY 4.0 and is
downloadable with no account or key -- unlike MaxMind GeoLite2, which needs a
signed-up licence key and forbids redistribution.

This script needs internet. Everything else in the project does not: once the
.mmdb files are on disk, src/geoip.py only ever reads them locally, so
scripts/offline_selfcheck.py still passes.

    python scripts/setup_geoip.py                # fetch the pinned month
    python scripts/setup_geoip.py --month 2026-10
    python scripts/setup_geoip.py --force        # re-fetch even if present

The repository already ships these files; run this only to refresh them.
"""

from __future__ import annotations

import argparse
import gzip
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_DIR = REPO_ROOT / "data" / "geoip"

# The build date of the .mmdb files currently committed. Keeping this pinned
# means a fresh clone + regenerate reproduces the committed dataset exactly.
PINNED_MONTH = "2026-09"

_BASE = "https://download.db-ip.com/free"
_FILES = {
    "dbip-country-lite.mmdb": "dbip-country-lite-{month}.mmdb.gz",
    "dbip-asn-lite.mmdb": "dbip-asn-lite-{month}.mmdb.gz",
}


def _download(url: str) -> bytes:
    print(f"  GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "btc-traffic-forensics/setup"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310  (fixed https host)
        return resp.read()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--month", default=PINNED_MONTH, help="DB-IP release, YYYY-MM")
    ap.add_argument("--dir", default=str(DB_DIR), help="target directory")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args()

    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for local_name, remote_tmpl in _FILES.items():
        dest = out_dir / local_name
        if dest.is_file() and not args.force:
            print(f"[skip] {dest} already present ({dest.stat().st_size:,} bytes)")
            continue
        url = f"{_BASE}/{remote_tmpl.format(month=args.month)}"
        try:
            blob = gzip.decompress(_download(url))
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {url}\n       {type(exc).__name__}: {exc}")
            if args.month == PINNED_MONTH:
                print("       DB-IP keeps only the last few months. Try a recent "
                      "--month YYYY-MM (the committed .mmdb files still work).")
            return 1
        dest.write_bytes(blob)
        print(f"[ ok ] {dest}  ({len(blob):,} bytes)")

    (out_dir / "SOURCE.txt").write_text(
        f"DB-IP Lite databases, release {args.month}\n"
        f"Source : https://db-ip.com/db/download/ip-to-country-lite\n"
        f"         https://db-ip.com/db/download/ip-to-asn-lite\n"
        f"Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
        f"         IP Geolocation by DB-IP (https://db-ip.com)\n"
        f"Fetched by scripts/setup_geoip.py\n",
        encoding="utf-8",
    )
    print(f"\nGeoIP databases ready in {out_dir}. The pipeline now runs offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
