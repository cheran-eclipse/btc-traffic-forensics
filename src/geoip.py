"""
SIH26146 / BitGuard AI -- offline GeoIP enrichment.

The problem statement asks for geo_country / asn to be derived by looking an IP
up against an "open source downloadable Geo IP database", not hand-assigned.

This wraps the DB-IP Lite databases (CC BY 4.0, https://db-ip.com/db/):
    data/geoip/dbip-country-lite.mmdb   IP -> ISO country code
    data/geoip/dbip-asn-lite.mmdb       IP -> autonomous system number

Both are local MaxMind-format (.mmdb) files. Every lookup is an in-process
b-tree read of that file -- no socket is ever opened, so this is safe under
scripts/offline_selfcheck.py. The files are fetched once by
scripts/setup_geoip.py; at runtime they are just data.

Private / reserved / unroutable addresses (and any address the database does
not cover) resolve to ("ZZ", "AS0") rather than raising -- a synthetic packet
from 10.x.x.x is a real thing to see on a capture and should not crash the
pipeline.
"""

from __future__ import annotations

import ipaddress
from pathlib import Path
from typing import Any

try:
    import maxminddb
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "maxminddb is required for GeoIP enrichment -- `pip install maxminddb` "
        "(it is in requirements.txt)"
    ) from exc

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_DIR = _REPO_ROOT / "data" / "geoip"
COUNTRY_DB_NAME = "dbip-country-lite.mmdb"
ASN_DB_NAME = "dbip-asn-lite.mmdb"

UNKNOWN_COUNTRY = "ZZ"   # ISO 3166 user-assigned "unknown"
UNKNOWN_ASN = "AS0"

_SETUP_HINT = (
    "GeoIP databases not found. Run once (needs internet):\n"
    "    python scripts/setup_geoip.py\n"
    "This downloads the DB-IP Lite .mmdb files into data/geoip/. After that the "
    "pipeline runs fully offline."
)


class GeoIPUnavailable(RuntimeError):
    """Raised when the local .mmdb files are missing."""


class GeoIPResolver:
    """Country + ASN lookups against the two local .mmdb files."""

    def __init__(self, country_db: Path, asn_db: Path) -> None:
        self._country = maxminddb.open_database(str(country_db))
        self._asn = maxminddb.open_database(str(asn_db))
        self.country_build_date = _build_date(self._country)
        self.asn_build_date = _build_date(self._asn)

    # -- lookups ---------------------------------------------------------

    def country(self, ip: str) -> str:
        if _is_non_public(ip):
            return UNKNOWN_COUNTRY
        rec = self._country.get(ip)
        if isinstance(rec, dict):
            iso = rec.get("country", {}).get("iso_code")
            if iso:
                return str(iso)
        return UNKNOWN_COUNTRY

    def asn(self, ip: str) -> str:
        if _is_non_public(ip):
            return UNKNOWN_ASN
        rec = self._asn.get(ip)
        if isinstance(rec, dict):
            num = rec.get("autonomous_system_number")
            if num is not None:
                return f"AS{int(num)}"
        return UNKNOWN_ASN

    def asn_org(self, ip: str) -> str | None:
        if _is_non_public(ip):
            return None
        rec = self._asn.get(ip)
        if isinstance(rec, dict):
            return rec.get("autonomous_system_organization")
        return None

    def lookup(self, ip: str) -> tuple[str, str]:
        """(country_iso2, 'AS<n>') -- the two values the CSV schema needs."""
        return self.country(ip), self.asn(ip)

    # -- lifecycle -----------------------------------------------------

    def close(self) -> None:
        self._country.close()
        self._asn.close()

    def __enter__(self) -> "GeoIPResolver":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _build_date(db: Any) -> str:
    import datetime

    try:
        return datetime.date.fromtimestamp(db.metadata().build_epoch).isoformat()
    except Exception:  # pragma: no cover
        return "unknown"


def _is_non_public(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True
    return not addr.is_global


def default_db_paths(db_dir: Path | str = DEFAULT_DB_DIR) -> tuple[Path, Path]:
    d = Path(db_dir)
    return d / COUNTRY_DB_NAME, d / ASN_DB_NAME


def databases_present(db_dir: Path | str = DEFAULT_DB_DIR) -> bool:
    country, asn = default_db_paths(db_dir)
    return country.is_file() and asn.is_file()


def default_resolver(db_dir: Path | str = DEFAULT_DB_DIR) -> GeoIPResolver:
    """Open the bundled databases, or raise GeoIPUnavailable with a fix hint."""
    country, asn = default_db_paths(db_dir)
    if not (country.is_file() and asn.is_file()):
        raise GeoIPUnavailable(_SETUP_HINT)
    return GeoIPResolver(country, asn)


def _demo() -> None:
    r = default_resolver()
    print(f"country db build date: {r.country_build_date}")
    print(f"asn db build date    : {r.asn_build_date}\n")
    for ip in ["8.8.8.8", "1.1.1.1", "52.95.110.1", "88.198.1.1",
               "139.59.1.1", "10.0.0.1"]:
        c, a = r.lookup(ip)
        print(f"  {ip:16s} -> {c:3s} {a:10s} {r.asn_org(ip) or ''}")
    r.close()


if __name__ == "__main__":
    _demo()
