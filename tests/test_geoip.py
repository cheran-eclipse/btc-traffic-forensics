"""Known-answer tests for src/geoip.py and the geo enrichment in the generator.

'Known answer' means: these IPs / prefixes have a real, stable geolocation
(Google DNS, Cloudflare, Amazon, Hetzner ...), so the test checks the lookup is
actually *correct*, not just that it returns something.

Skipped automatically if the DB-IP databases aren't present
(`python scripts/setup_geoip.py`).
"""

import ipaddress
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geoip  # noqa: E402

pytestmark = pytest.mark.skipif(
    not geoip.databases_present(),
    reason="GeoIP databases missing -- run: python scripts/setup_geoip.py",
)


@pytest.fixture(scope="module")
def resolver():
    r = geoip.default_resolver()
    yield r
    r.close()


# -- 1. hard known answers -------------------------------------------------

@pytest.mark.parametrize(
    "ip, country, asn, org_contains",
    [
        ("8.8.8.8", "US", "AS15169", "Google"),
        ("8.8.4.4", "US", "AS15169", "Google"),
        ("1.1.1.1", None, "AS13335", "Cloudflare"),      # anycast: ASN is the stable part
        ("52.95.110.1", "US", "AS16509", "Amazon"),
        ("88.198.1.1", "DE", "AS24940", "Hetzner"),
        ("193.0.6.139", "NL", "AS3333", "RIPE"),
        ("139.59.1.1", "IN", "AS14061", "DigitalOcean"),
    ],
)
def test_known_ip_lookups(resolver, ip, country, asn, org_contains):
    c, a = resolver.lookup(ip)
    if country is not None:
        assert c == country, f"{ip}: expected {country}, got {c}"
    assert a == asn, f"{ip}: expected {asn}, got {a}"
    assert org_contains.lower() in (resolver.asn_org(ip) or "").lower()


def test_private_and_bogus_addresses_resolve_to_unknown(resolver):
    for ip in ("10.0.0.1", "192.168.1.1", "127.0.0.1", "169.254.0.1", "not-an-ip"):
        assert resolver.lookup(ip) == (geoip.UNKNOWN_COUNTRY, geoip.UNKNOWN_ASN)


# -- 2. every prefix the generator draws from is what it claims ----------

def test_generator_prefixes_resolve_to_their_declared_country(resolver):
    from generate_dataset import IP_PREFIXES

    for cidr, expected_cc in IP_PREFIXES:
        net = ipaddress.ip_network(cidr)
        # sample 25 hosts spread across the block
        step = max(net.num_addresses // 25, 1)
        hosts = [str(net.network_address + off)
                 for off in range(1, net.num_addresses - 1, step)][:25]
        got = {resolver.country(h) for h in hosts}
        assert got == {expected_cc}, f"{cidr}: expected all {expected_cc}, got {got}"


# -- 3. the generated dataset's geo columns are real lookups ------------

@pytest.fixture(scope="module")
def dataset():
    from generate_dataset import build_dataset

    return build_dataset(seed=7)


def test_every_row_geo_matches_a_fresh_lookup(resolver, dataset):
    for tx in dataset["transactions"]:
        c, a = resolver.lookup(tx["src_ip"])
        assert (tx["geo_country"], tx["asn"]) == (c, a), tx["src_ip"]


def test_geo_columns_are_real_lookups_not_hand_assigned(dataset):
    countries = {tx["geo_country"] for tx in dataset["transactions"]}
    asns = {tx["asn"] for tx in dataset["transactions"]}

    assert len(countries) >= 5
    # real AS numbers that show up because the IPs really are in those networks
    assert {"AS16509", "AS15169", "AS24940"} <= asns   # Amazon, Google, Hetzner
    # the pre-GeoIP generator's invented AS numbers must be gone
    fake = {"AS9829", "AS60781", "AS3320", "AS7922", "AS12389", "AS9299", "AS5089", "AS3215"}
    assert not (fake & asns)
    # the vast majority resolve to a real AS (a few routable IPs aren't in the
    # ASN-lite db and fall back to AS0 -- that's a genuine lookup outcome)
    real = sum(1 for tx in dataset["transactions"] if tx["asn"] != "AS0")
    assert real / len(dataset["transactions"]) > 0.9


def test_geo_hopping_patterns_cross_multiple_countries(dataset):
    # the hop patterns force each hop into a different country; check every
    # planted instance genuinely spans several (real, looked-up) countries.
    for atype in ("rapid_movement", "layering", "circular_flow"):
        instances = [i for i in dataset["instances"] if i["anomaly_type"] == atype]
        assert instances
        for inst in instances:
            members = set(inst["members"]) | {inst["origin"]}
            countries = {
                tx["geo_country"]
                for tx in dataset["transactions"]
                if members & (set(tx["input_addresses"]) | set(tx["output_addresses"]))
            }
            assert len(countries) >= 3, f"{atype} instance spans only {countries}"


def test_resolver_reports_db_build_date(resolver):
    # a real database, dated -- not a stub
    assert resolver.country_build_date.startswith("202")
    assert resolver.asn_build_date.startswith("202")
