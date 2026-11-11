"""Headless tests for the Streamlit app (``app.py``) using ``streamlit.testing``.

They cover the baseline workflow (save with a name, revert, restore the
reference, persistence across a refresh) and the compare-to-baseline switches
in the numbers and plots sections (independent of each other, shared between
the two tabs).

Every test points the app at a temporary baseline file through the
``VDL_BASELINE_PATH`` environment variable, and fails if the real
``user_data/baseline.json`` is changed.

Not covered (needs a real browser): pressing Enter in the name box submits
the form, and the name box gets keyboard focus when the popover opens.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from utils.setup_store import BASELINE_PATH_ENV, DEFAULT_BASELINE_PATH  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = 120


def _snapshot(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


@pytest.fixture
def baseline_file(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "baseline.json"
    monkeypatch.setenv(BASELINE_PATH_ENV, str(path))
    real_before = _snapshot(DEFAULT_BASELINE_PATH)
    yield path
    assert _snapshot(DEFAULT_BASELINE_PATH) == real_before, "test modified the real saved baseline"


def start_app(**state) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    assert_no_exceptions(at)
    return at


def assert_no_exceptions(at: AppTest) -> None:
    assert not at.exception, [e.message for e in at.exception]


def click(at: AppTest, label: str) -> None:
    [button] = [b for b in at.button if b.label == label]
    button.click().run()
    assert_no_exceptions(at)


def change_front_arb(at: AppTest, value: float = 40.0) -> None:
    at.sidebar.slider(key="w_arb_f").set_value(value).run()
    assert_no_exceptions(at)


def save_baseline(at: AppTest, name: str) -> None:
    at.sidebar.text_input(key="baseline_name_input").input(name)
    click(at, "Save baseline")


def saved_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# -----------------------------------------------------------------------------
# Save as new baseline
# -----------------------------------------------------------------------------
def test_starts_from_reference_without_saved_file(baseline_file):
    at = start_app()
    assert at.session_state.baseline_source == "Reference setup"
    assert not at.session_state.baseline_persisted
    assert not baseline_file.exists()


def test_save_with_name(baseline_file):
    at = start_app()
    change_front_arb(at)
    at.session_state["save_baseline_popover"] = True
    save_baseline(at, "  Stiff   front ARB ")

    ss = at.session_state
    assert ss.baseline_source == "Stiff front ARB"
    assert ss.baseline == ss.setup
    assert ss.baseline_persisted
    assert ss.baseline_name_input == ""
    assert ss.save_baseline_popover is False
    assert "Stiff front ARB" in ss.chat[-1]["content"]

    payload = saved_payload(baseline_file)
    assert payload["name"] == "Stiff front ARB"
    assert payload["setup"] == ss.baseline


def test_save_with_blank_name_uses_timestamp(baseline_file):
    at = start_app()
    change_front_arb(at)
    save_baseline(at, "   ")
    assert at.session_state.baseline_source.startswith("Setup saved ")
    assert saved_payload(baseline_file)["name"] is None


def test_named_baseline_survives_refresh(baseline_file):
    at = start_app()
    change_front_arb(at)
    save_baseline(at, "Stiff front ARB")

    fresh = start_app()
    ss = fresh.session_state
    assert ss.baseline_source == "Stiff front ARB"
    assert ss.baseline_persisted
    assert ss.baseline == at.session_state.baseline
    assert ss.setup == ss.baseline
    assert ss.chat[0]["content"].startswith("Welcome back! I loaded your saved baseline (Stiff front ARB)")


def test_revert_goes_back_to_saved_baseline(baseline_file):
    at = start_app()
    change_front_arb(at, 40.0)
    save_baseline(at, "ARB 40")
    change_front_arb(at, 60.0)
    assert at.session_state.setup != at.session_state.baseline

    click(at, "Revert to baseline")
    assert at.session_state.setup == at.session_state.baseline
    assert at.sidebar.slider(key="w_arb_f").value == 40.0


def test_restore_reference_clears_saved_baseline(baseline_file):
    at = start_app()
    change_front_arb(at)
    save_baseline(at, "Temporary")
    assert baseline_file.exists()

    click(at, "Restore reference setup")
    ss = at.session_state
    assert ss.baseline_source == "Reference setup"
    assert ss.baseline_saved_at is None
    assert not ss.baseline_persisted
    assert ss.setup == ss.baseline
    assert not baseline_file.exists()


def test_corrupt_saved_file_falls_back_to_reference(baseline_file):
    baseline_file.write_text("{not json", encoding="utf-8")
    at = start_app()
    assert at.session_state.baseline_source == "Reference setup"
    assert at.session_state.store_problem


def test_old_session_without_new_keys_still_runs(baseline_file):
    """A browser tab opened on an older app version must survive a Rerun."""
    reference = start_app().session_state.baseline
    at = start_app(setup=reference, baseline=reference, baseline_source="Custom, saved Nov 11, 11:18",
                   baseline_persisted=True, store_problem=None, result=None, show_deltas=True,
                   chat=[{"role": "assistant", "content": "old"}])
    for key in ("baseline_saved_at", "delta_metrics", "delta_plots", "baseline_name_input"):
        assert key in at.session_state


# -----------------------------------------------------------------------------
# Degrees in the UI, radians in the physics
# -----------------------------------------------------------------------------
DEG = math.pi / 180.0
RADIAN_UNIT = re.compile(r"/\s*rad\b|\brad\b|radian", re.IGNORECASE)


def visible_text(at: AppTest) -> list[str]:
    texts = []
    for kind in ("markdown", "caption", "info", "warning", "error", "subheader", "title"):
        texts += [e.value for e in at.get(kind)]
    for widget in (*at.slider, *at.toggle, *at.number_input, *at.radio, *at.selectbox, *at.button):
        texts += [widget.label, getattr(widget, "help", "") or ""]
    for metric in at.metric:
        texts += [metric.label, metric.value, metric.delta or "", metric.proto.help]
    texts += [chart.proto.spec for chart in at.get("plotly_chart")]
    return texts


def test_arb_slider_shows_degrees_but_setup_stays_per_radian(baseline_file):
    at = start_app()
    slider = at.sidebar.slider(key="w_arb_f")
    assert "N·m/deg" in slider.label
    stored = at.session_state.setup["front_suspension"]["anti_roll_bar_rate"]
    assert slider.value == pytest.approx(stored * DEG)

    change_front_arb(at, 500.0)
    assert at.session_state.setup["front_suspension"]["anti_roll_bar_rate"] == pytest.approx(500.0 / DEG)


def test_derived_roll_stiffness_in_degrees(baseline_file):
    at = start_app()
    roll = {m.label: m.value for m in at.sidebar.metric if m.label.startswith("Roll stiffness")}
    assert roll and all(v.endswith("N·m/deg") for v in roll.values())


def test_no_radians_anywhere_in_the_ui(baseline_file):
    at = start_app()
    click(at, "Run Simulation")
    change_front_arb(at)
    click(at, "Run Simulation")
    at.toggle(key="cmp_metrics_exp").set_value(True)
    at.toggle(key="cmp_plots_exp").set_value(True).run()
    at.chat_input(key="chat_box").set_value("stiffen the front arb by 20%").run()
    assert_no_exceptions(at)
    offenders = [t for t in visible_text(at) if RADIAN_UNIT.search(t)]
    assert not offenders, offenders


# -----------------------------------------------------------------------------
# Compare-to-baseline switches
# -----------------------------------------------------------------------------
def run_changed_setup() -> AppTest:
    at = start_app()
    click(at, "Run Simulation")
    change_front_arb(at)
    click(at, "Run Simulation")
    return at


def test_compare_switches_only_appear_with_results(baseline_file):
    at = start_app()
    assert not at.toggle
    click(at, "Run Simulation")
    assert {t.key for t in at.toggle} == {"cmp_metrics_exp", "cmp_plots_exp",
                                          "cmp_metrics_chat", "cmp_plots_chat"}


@pytest.mark.parametrize("numbers_on, plots_on", [(False, False), (True, False), (False, True), (True, True)])
def test_compare_switches_are_independent(baseline_file, numbers_on, plots_on):
    at = run_changed_setup()
    at.toggle(key="cmp_metrics_exp").set_value(numbers_on)
    at.toggle(key="cmp_plots_exp").set_value(plots_on).run()
    assert_no_exceptions(at)

    has_metric_deltas = any(m.delta for m in at.main.metric)
    assert has_metric_deltas == numbers_on

    chart_specs = " ".join(chart.proto.spec for chart in at.get("plotly_chart"))
    assert ("Change in vertical load" in chart_specs) == plots_on

    [header] = [m.value for m in at.main.markdown if m.value.startswith("**Results:**")]
    assert ("numbers" in header) == numbers_on
    assert ("plots compared" in header or "and plots" in header) == plots_on


@pytest.mark.parametrize("part", ["metrics", "plots"])
def test_compare_switch_is_shared_between_tabs(baseline_file, part):
    at = run_changed_setup()
    at.toggle(key=f"cmp_{part}_chat").set_value(True).run()
    assert_no_exceptions(at)
    assert at.session_state[f"delta_{part}"] is True
    assert at.toggle(key=f"cmp_{part}_exp").value is True

    at.toggle(key=f"cmp_{part}_exp").set_value(False).run()
    assert_no_exceptions(at)
    assert at.toggle(key=f"cmp_{part}_chat").value is False
