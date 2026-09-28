"""Dashboard smoke tests (stage 6).

Streamlit's own AppTest harness runs app.py headlessly - no browser, no server - so a crash in
the sidebar, the metric row, the Folium map building or the min-cut table shows up as a test
failure instead of a red box the demo audience gets to see.

These are slower than the algorithm tests (each run replans the whole network), so they live in
their own file: `pytest tests/test_planner.py` for the fast loop, plain `pytest` for everything.
"""
import pytest

from streamlit.testing.v1 import AppTest


@pytest.fixture(scope="module")
def planned():
    """Run app.py, click Plan network, return the finished app state."""
    at = AppTest.from_file("app.py", default_timeout=600)
    at.run()
    assert not at.exception, f"app.py raised on first load: {at.exception}"
    button = next(b for b in at.button if "Plan network" in str(b.label))
    button.click().run()
    assert not at.exception, f"planning raised: {at.exception}"
    return at


def test_app_loads_without_planning():
    """Before the button is pressed the app must show its prompt, not a stack trace."""
    at = AppTest.from_file("app.py", default_timeout=300)
    at.run()
    assert not at.exception
    assert any("Plan network" in str(i.value) for i in at.info)


def test_sidebar_controls_are_all_present():
    at = AppTest.from_file("app.py", default_timeout=300)
    at.run()
    labels = [str(s.label) for s in at.slider]
    for expected in ("Demand points", "Candidate towers", "Mbps per customer",
                     "Per-tower equipment budget"):
        assert any(expected in l for l in labels), f"missing slider: {expected}"
    assert any("Candidate towers" in str(s.label) and s.max >= 40 for s in at.slider), \
        "the |C| ceiling should reach 40: branch and bound handles it easily"


def test_planning_renders_the_metric_row(planned):
    values = [str(m.value) for m in planned.metric]
    labels = [str(m.label) for m in planned.metric]
    assert len(planned.metric) == 5, f"expected 5 metric tiles, got {labels}"
    for expected in ("Towers built", "Project cost", "Backbone cable", "Max flow", "Customers covered"):
        assert any(expected in l for l in labels), f"missing metric: {expected}"
    assert any("k$" in v for v in values)
    assert any("km" in v for v in values)


def test_project_cost_tile_includes_equipment_spend(planned):
    """The baseline reported build cost only; equipment spend must be in the headline number."""
    tile = next(m for m in planned.metric if "Project cost" in str(m.label))
    assert "build" in str(tile.delta) and "equipment" in str(tile.delta)
    total = float(str(tile.value).replace(" k$", ""))
    build = float(str(tile.delta).split(" build")[0])
    assert total > build, "total must exceed build cost once equipment is counted"


def test_coverage_tile_is_the_citywide_figure(planned):
    tile = next(m for m in planned.metric if "Customers covered" in str(m.label))
    assert "unreachable" in str(tile.delta)
    pct = float(str(tile.value).rstrip("%"))
    assert 0 < pct <= 100


def test_stage_tables_all_render(planned):
    """Stage 1 comparison, stage 2 per-tower equipment, and the stage 4 min-cut."""
    assert len(planned.table) >= 2, "stage 1 and stage 2 tables must both render"
    headers = [str(h.value) for h in planned.subheader]
    for stage in ("Stage 1", "Stage 2", "Stage 4", "Stage 5"):
        assert any(stage in h for h in headers), f"missing section: {stage}"


def test_stage_2_shows_per_tower_variation(planned):
    """The whole point of the stage-2 rework: capacities must differ between towers."""
    stage2 = planned.table[1].value
    caps = list(stage2["capacity Mbps"])
    assert len(caps) >= 2
    assert len(set(caps)) > 1, "every tower got identical equipment; stage 2 is decorative again"


def test_dijkstra_routing_section_reports_a_path(planned):
    assert any("Shortest road path" in str(m.value) for m in planned.markdown)


def test_no_streamlit_exceptions_anywhere(planned):
    assert not planned.exception
    assert len(planned.error) == 0, [str(e.value) for e in planned.error]
