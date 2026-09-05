"""Smoke test for src/dashboard.py using Streamlit's AppTest harness.

Not a UI test -- just: the script runs end to end without raising, the command
-center metrics populate, and selecting a lead renders its case file.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(str(ROOT / "src" / "dashboard.py"), default_timeout=300).run()


def test_runs_without_exception(app):
    assert app.exception == []


def test_command_center_metrics_populate(app):
    labels = {m.label for m in app.metric}
    assert {"Transactions", "Entities", "Model-flagged",
            "Behaviour clusters", "High / Critical risk"} <= labels
    assert int(next(m.value for m in app.metric if m.label == "Transactions").replace(",", "")) > 0


def test_priority_alerts_table_present(app):
    assert len(app.dataframe) >= 1


def test_selecting_a_lead_renders_a_case_file(app):
    opts = app.selectbox[0].options
    assert opts
    app.selectbox[0].select(opts[0]).run()
    assert app.exception == []
    assert any("INVESTIGATIVE LEAD" in c.value for c in app.code)
