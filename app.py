"""
Vehicle Dynamics Lab: Streamlit front end.

Run from the repository root with:

    streamlit run app.py

All physics lives in ``physics/``; natural-language setup changes are parsed by
``utils/agent_parser.py``. This file only handles layout, state and charts.

Three setups
------------
* **Reference** -- the original, known-good car (``REFERENCE_SETUP``). Never changes.
* **Baseline**  -- what results are compared against. Starts as the reference;
  students can replace it with "Save as new baseline". A saved baseline is
  written to ``user_data/baseline.json`` (``utils.setup_store``) and reloaded
  on refresh or restart; the app opens with the current setup equal to it.
* **Current**   -- what the sliders and chat edit.

State model
-----------
All three are plain-number dictionaries (SI units) as used by
``utils.agent_parser``. Sidebar sliders are a *view* of the current setup in
friendlier units (mm, N/mm, and degrees instead of radians):

* slider moved          -> ``pull_widgets_into_setup`` copies slider values into the dict
* chat / revert / restore -> callbacks change the dict, then ``push_setup_to_widgets``
                           moves the sliders to match

Callbacks run *before* the script re-renders, which is the only time Streamlit
allows widget values to be changed programmatically.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from physics import (
    GRAVITY,
    VehicleSetup,
    find_max_cornering_speed,
    roll_moment_distribution,
    solve_aero_ride_heights,
    solve_steady_state_cornering,
    solve_straight_line,
    static_wheel_loads,
)
from utils.agent_parser import PARAMETERS as SETUP_PARAMETERS
from utils.agent_parser import dict_to_setup, interpret_setup_request, setup_to_dict
from utils.setup_store import SetupStoreError, delete_setup, load_setup, save_setup

st.set_page_config(page_title="Vehicle Dynamics Lab", layout="wide")

BASE_SETUP = VehicleSetup()
REFERENCE_SETUP = setup_to_dict(BASE_SETUP)

#: The physics works in radians; the UI only ever shows degrees.
#: A stiffness per radian (e.g. N·m/rad) times this is the stiffness per degree.
NM_PER_RAD_TO_DEG = math.pi / 180.0

CORNERING = "Constant-radius cornering"
STRAIGHT = "Straight-line acceleration"
EXPERIMENT_DEFAULTS = {
    "exp_mode": CORNERING,
    "exp_speed": 80.0,  # km/h
    "exp_radius": 50.0,  # m
    "exp_dir": "Left",
    "exp_accel": 0.3,  # g
}

CHART_LLTD = "Lateral load transfer"
CHART_LOADS = "Tire loads"
CHART_AERO = "Aero balance"
CHART_TIRES = "Tire slip vs. force"

FRONT_COLOR = "#1f77b4"
REAR_COLOR = "#ff7f0e"
WHEEL_COLORS = {"FL": "#6baed6", "FR": "#08519c", "RL": "#fdae6b", "RR": "#d94801"}
INCREASE_COLOR = "#2ca02c"
DECREASE_COLOR = "#d62728"

EXAMPLE_COMMANDS = [
    "Stiffen front springs by 20%",
    "Soften the rear anti-roll bar by 30%",
    "Add 50 kg of ballast",
    "Lower the CG by 30 mm",
    "Add 100 kg of downforce at 150 km/h",
    "Move weight forward 3%",
    "Double the front ARB",
]


# =============================================================================
# Sidebar slider definitions
# =============================================================================
@dataclass(frozen=True)
class SliderSpec:
    """A sidebar slider that edits one value in the setup dictionary.

    ``display = stored_value * factor`` (times ``aero[ref_key]`` if given, which
    turns an aero-map scale factor into an actual C_L / C_D value).
    """

    key: str
    label: str
    path: str
    unit: str
    factor: float
    min_value: float
    max_value: float
    step: float
    fmt: str
    help: str
    section: str
    ref_key: str | None = None

    @property
    def widget_key(self) -> str:
        return f"w_{self.key}"

    @property
    def decimals(self) -> int:
        match = re.search(r"\.(\d+)f", self.fmt)
        return int(match.group(1)) if match else 0


SLIDERS: tuple[SliderSpec, ...] = (
    SliderSpec("mass", "Vehicle mass", "mass", "kg", 1.0, 600.0, 2500.0, 10.0, "%.0f",
               "Total mass including driver. More mass = more weight on the tires but also more inertia.",
               "Mass & geometry"),
    SliderSpec("cg_height", "CG height", "cg_height", "mm", 1000.0, 250.0, 800.0, 5.0, "%.0f",
               "Height of the centre of gravity. A higher CG means more load transfer and more body roll.",
               "Mass & geometry"),
    SliderSpec("weight_dist", "Front weight distribution", "weight_distribution_front", "%", 100.0,
               35.0, 65.0, 0.5, "%.1f", "Share of the car's weight on the front axle when parked.",
               "Mass & geometry"),
    SliderSpec("track_f", "Front track width", "front_suspension.track_width", "mm", 1000.0,
               1300.0, 1800.0, 5.0, "%.0f", "Distance between the front tire centres. Wider = less load transfer.",
               "Mass & geometry"),
    SliderSpec("track_r", "Rear track width", "rear_suspension.track_width", "mm", 1000.0,
               1300.0, 1800.0, 5.0, "%.0f", "Distance between the rear tire centres.", "Mass & geometry"),
    SliderSpec("spring_f", "Front spring rate", "front_suspension.spring_rate", "N/mm", 1e-3,
               10.0, 200.0, 1.0, "%.1f", "Coil spring stiffness at the front. Newtons per millimetre of compression.",
               "Springs & anti-roll bars"),
    SliderSpec("spring_r", "Rear spring rate", "rear_suspension.spring_rate", "N/mm", 1e-3,
               10.0, 200.0, 1.0, "%.1f", "Coil spring stiffness at the rear.", "Springs & anti-roll bars"),
    SliderSpec("arb_f", "Front anti-roll bar", "front_suspension.anti_roll_bar_rate", "N·m/deg", NM_PER_RAD_TO_DEG,
               0.0, 1400.0, 5.0, "%.0f", "Resists body roll only (not bumps): newton-metres per degree of body roll. "
               "A stiffer front bar adds understeer.", "Springs & anti-roll bars"),
    SliderSpec("arb_r", "Rear anti-roll bar", "rear_suspension.anti_roll_bar_rate", "N·m/deg", NM_PER_RAD_TO_DEG,
               0.0, 1400.0, 5.0, "%.0f", "A stiffer rear bar moves the balance toward oversteer.",
               "Springs & anti-roll bars"),
    SliderSpec("cl", "Downforce coefficient [CL]", "aero.cl_scale", "", 1.0, 0.0, 3.0, 0.05, "%.2f",
               "How much downforce the car makes (at its static ride height). 0 = no wings.",
               "Aerodynamics", ref_key="reference_cl"),
    SliderSpec("cd", "Drag coefficient [CD]", "aero.cd_scale", "", 1.0, 0.20, 1.20, 0.01, "%.2f",
               "How 'slippery' the car is. Higher = more drag and more power needed.",
               "Aerodynamics", ref_key="reference_cd"),
)


def _get(d: dict[str, Any], path: str) -> Any:
    for key in path.split("."):
        d = d[key]
    return d


def _set(d: dict[str, Any], path: str, value: float) -> None:
    *parents, last = path.split(".")
    for key in parents:
        d = d[key]
    d[last] = value


def _differs(a: float, b: float) -> bool:
    return not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)


def _display_value(d: dict[str, Any], spec: SliderSpec) -> float:
    value = float(_get(d, spec.path)) * spec.factor
    return value * d["aero"][spec.ref_key] if spec.ref_key else value


def _stored_value(d: dict[str, Any], spec: SliderSpec, display: float) -> float:
    value = display / spec.factor
    return value / d["aero"][spec.ref_key] if spec.ref_key else value


def push_setup_to_widgets(d: dict[str, Any]) -> list[str]:
    """Move every slider to match ``d``. Values outside a slider's range are
    clamped (and written back to ``d``) so the sliders never lie."""
    notes = []
    for spec in SLIDERS:
        shown = _display_value(d, spec)
        clamped = min(max(shown, spec.min_value), spec.max_value)
        if _differs(clamped, shown):
            notes.append(
                f"{spec.label} limited to the slider range "
                f"({spec.fmt % spec.min_value}-{spec.fmt % spec.max_value} {spec.unit})."
            )
            _set(d, spec.path, _stored_value(d, spec, clamped))
        st.session_state[spec.widget_key] = float(clamped)
    return notes


def pull_widgets_into_setup(d: dict[str, Any]) -> None:
    """Copy slider values into the setup dictionary.

    Only values the student actually moved are written, so unit round-trips
    never make an untouched parameter drift away from its baseline value.
    """
    for spec in SLIDERS:
        if spec.widget_key in st.session_state:
            shown = float(st.session_state[spec.widget_key])
            if _differs(shown, _display_value(d, spec)):
                _set(d, spec.path, _stored_value(d, spec, shown))


# =============================================================================
# Setup comparison
# =============================================================================
@dataclass(frozen=True)
class SetupDiff:
    label: str
    baseline: str
    current: str
    change: str


def _fmt_number(x: float, decimals: int | None = None) -> str:
    if decimals is not None:
        return f"{x:,.{decimals}f}"
    if abs(x) >= 100:
        return f"{x:,.0f}"
    if abs(x) >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def _change_text(base: float, cur: float, unit: str, decimals: int | None, points: bool = False) -> str:
    delta = cur - base
    text = f"{'+' if delta >= 0 else '-'}{_fmt_number(abs(delta), decimals)}"
    if points:
        return f"{text} pts"
    text += f" {unit}" if unit else ""
    if base != 0:
        text += f" ({(cur / base - 1) * 100:+.1f}%)"
    return text


def _numeric_leaves(d: dict[str, Any], prefix: str = ""):
    for key, value in d.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from _numeric_leaves(value, path + ".")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            yield path, float(value)


def setup_differences(current: dict[str, Any], baseline: dict[str, Any]) -> list[SetupDiff]:
    """Every parameter that differs between two setups, in display units.

    Slider parameters come first, then values only reachable via chat (ride
    heights, tire grip, ...), then anything else in the dictionary.
    """
    diffs: list[SetupDiff] = []
    covered: set[str] = set()

    for spec in SLIDERS:
        covered.add(spec.path)
        b, c = _display_value(baseline, spec), _display_value(current, spec)
        if _differs(b, c):
            unit = f" {spec.unit}" if spec.unit else ""
            diffs.append(SetupDiff(
                spec.label, f"{spec.fmt % b}{unit}", f"{spec.fmt % c}{unit}",
                _change_text(b, c, spec.unit, spec.decimals, points=spec.unit == "%"),
            ))

    for p in SETUP_PARAMETERS:
        targets = [(p.path.format(axle=a), a) for a in ("front", "rear")] if p.per_axle else [(p.path, None)]
        for path, axle in targets:
            if path in covered:
                continue
            covered.add(path)
            b, c = float(_get(baseline, path)), float(_get(current, path))
            if _differs(b, c):
                label = f"{axle.capitalize()} {p.label}" if axle else p.label[0].upper() + p.label[1:]
                unit = f" {p.display_unit}" if p.display_unit else ""
                diffs.append(SetupDiff(
                    label, f"{_fmt_number(b * p.display_factor)}{unit}", f"{_fmt_number(c * p.display_factor)}{unit}",
                    _change_text(b * p.display_factor, c * p.display_factor, p.display_unit, None),
                ))

    base_leaves = dict(_numeric_leaves(baseline))
    for path, c in _numeric_leaves(current):
        if path in covered or path.startswith("aero.reference") or path not in base_leaves:
            continue
        b = base_leaves[path]
        if _differs(b, c):
            diffs.append(SetupDiff(path, _fmt_number(b), _fmt_number(c), _change_text(b, c, "", None)))
    return diffs


def setups_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return not setup_differences(a, b)


# =============================================================================
# Simulation (cached: identical setup + conditions are only solved once)
# =============================================================================
def _setup_json(d: dict[str, Any]) -> str:
    return json.dumps(d, sort_keys=True)


def current_conditions() -> dict[str, Any]:
    ss = st.session_state
    return {
        "mode": ss.exp_mode,
        "speed_kmh": float(ss.exp_speed),
        "radius": float(ss.exp_radius),
        "direction": ss.exp_dir,
        "accel_g": float(ss.exp_accel),
    }


def describe_conditions(cond: dict[str, Any]) -> str:
    if cond["mode"] == CORNERING:
        return f"{cond['speed_kmh']:.0f} km/h on a {cond['radius']:.0f} m {cond['direction'].lower()}-hand corner"
    return f"{cond['speed_kmh']:.0f} km/h, accelerating at {cond['accel_g']:+.2f} g"


@st.cache_data(show_spinner=False, max_entries=256)
def _simulate(setup_json: str, cond_json: str) -> dict[str, Any]:
    setup = dict_to_setup(json.loads(setup_json), BASE_SETUP)
    cond = json.loads(cond_json)
    speed = cond["speed_kmh"] / 3.6
    if cond["mode"] == CORNERING:
        radius = cond["radius"] if cond["direction"] == "Left" else -cond["radius"]
        return {
            "state": solve_steady_state_cornering(setup, speed, radius),
            "max_speed": find_max_cornering_speed(setup, abs(radius)),
        }
    return {
        "state": solve_straight_line(setup, speed, longitudinal_accel=cond["accel_g"] * GRAVITY),
        "max_speed": None,
    }


def run_experiment(setup_dict: dict[str, Any], cond: dict[str, Any]) -> dict[str, Any]:
    """Run the selected test and package everything the charts need."""
    setup_json = _setup_json(setup_dict)
    cond_json = json.dumps(cond, sort_keys=True)
    with st.spinner("Solving..."):
        out = _simulate(setup_json, cond_json)
    return {"cond": cond, "setup_json": setup_json, "cond_json": cond_json, **out}


@st.cache_data(show_spinner=False, max_entries=64)
def _aero_sweep(setup_json: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    setup = dict_to_setup(json.loads(setup_json), BASE_SETUP)
    speeds_kmh = np.linspace(0.0, 250.0, 26)
    front, rear = [], []
    for v_kmh in speeds_kmh:
        _, _, aero = solve_aero_ride_heights(setup, v_kmh / 3.6)
        front.append(aero.downforce_front)
        rear.append(aero.downforce_rear)
    return speeds_kmh, np.array(front), np.array(rear)


def car_of(res: dict[str, Any]) -> VehicleSetup:
    return dict_to_setup(json.loads(res["setup_json"]), BASE_SETUP)


def is_cornering(res: dict[str, Any]) -> bool:
    return res["cond"]["mode"] == CORNERING


# =============================================================================
# Session state & callbacks
# =============================================================================
REFERENCE_LABEL = "Reference setup"


def _baseline_label(name: str | None, saved_at: datetime) -> str:
    return name or f"Setup saved {saved_at:%b %d, %H:%M}"


def load_persisted_baseline() -> tuple[dict[str, Any], str, datetime | None, bool, str | None]:
    """Return ``(baseline, label, saved_at, persisted, problem)`` from disk, falling back to the reference."""
    try:
        saved = load_setup(REFERENCE_SETUP)
        if saved is not None:
            dict_to_setup(saved.setup, BASE_SETUP)  # must be a physically valid car
    except (SetupStoreError, ValueError, TypeError) as exc:
        return (copy.deepcopy(REFERENCE_SETUP), REFERENCE_LABEL, None, False,
                f"The saved baseline couldn't be loaded ({exc}), so the reference setup is used instead.")
    if saved is None:
        return copy.deepcopy(REFERENCE_SETUP), REFERENCE_LABEL, None, False, None
    return saved.setup, _baseline_label(saved.name, saved.saved_at), saved.saved_at, True, None


def init_state() -> None:
    """Fill in any missing session-state keys.

    Each key is checked on its own (not just on the first run) so that a
    browser session started on an older version of this file keeps working
    after the code changes and the user clicks Rerun.
    """
    ss = st.session_state
    if "baseline" not in ss:
        (ss.baseline, ss.baseline_source, ss.baseline_saved_at,
         ss.baseline_persisted, ss.store_problem) = load_persisted_baseline()
    for key, default in (("baseline_source", REFERENCE_LABEL), ("baseline_saved_at", None),
                         ("baseline_persisted", False), ("store_problem", None), ("result", None),
                         ("delta_metrics", False), ("delta_plots", False), ("baseline_name_input", "")):
        if key not in ss:
            ss[key] = default
    if "setup" not in ss:
        ss.setup = copy.deepcopy(ss.baseline)
        push_setup_to_widgets(ss.setup)
    if "chat" not in ss:
        greeting = (
            "Hi! I'm your setup engineer. Tell me what to change and I'll update the car "
            "and re-run your current experiment. Try:\n\n"
            + "\n".join(f"- *{c}*" for c in EXAMPLE_COMMANDS[:4])
        )
        if ss.baseline_persisted:
            greeting = f"Welcome back! I loaded your saved baseline ({ss.baseline_source}).\n\n" + greeting
        ss.chat = [{"role": "assistant", "content": greeting}]
    # Re-assigning keeps experiment inputs alive while hidden (Streamlit drops
    # the state of widgets that are not drawn on a run).
    for key, default in EXPERIMENT_DEFAULTS.items():
        ss[key] = ss.get(key, default)


def _rerun_if_results() -> None:
    ss = st.session_state
    if ss.result is not None:
        ss.result = run_experiment(ss.setup, current_conditions())


def on_run() -> None:
    ss = st.session_state
    pull_widgets_into_setup(ss.setup)
    ss.result = run_experiment(ss.setup, current_conditions())


def on_revert_to_baseline() -> None:
    ss = st.session_state
    ss.setup = copy.deepcopy(ss.baseline)
    push_setup_to_widgets(ss.setup)
    _rerun_if_results()
    ss.chat.append({"role": "assistant", "content": f"Setup reverted to the baseline ({ss.baseline_source})."})


def on_save_baseline() -> None:
    ss = st.session_state
    pull_widgets_into_setup(ss.setup)
    ss.baseline = copy.deepcopy(ss.setup)
    name = ss.get("baseline_name_input", "")
    ss.baseline_name_input = ""
    ss.save_baseline_popover = False
    try:
        saved = save_setup(ss.baseline, name=name)
    except OSError as exc:
        now = datetime.now().replace(microsecond=0)
        ss.baseline_source = _baseline_label(" ".join(name.split()) or None, now)
        ss.baseline_saved_at = now
        ss.baseline_persisted = False
        ss.store_problem = f"The baseline couldn't be saved to disk ({exc}). It will be lost on refresh."
        message = f"Set **{ss.baseline_source}** as the new baseline, but **couldn't save it to disk**, " \
                  "so it will only last until you refresh the page."
    else:
        ss.baseline_source = _baseline_label(saved.name, saved.saved_at)
        ss.baseline_saved_at = saved.saved_at
        ss.baseline_persisted = True
        ss.store_problem = None
        message = f"Saved the current setup as the new baseline **{ss.baseline_source}**. Results will now be " \
                  "compared against it, and it will still be here after a refresh or restart."
    ss.chat.append({"role": "assistant", "content": message})


def on_restore_reference() -> None:
    ss = st.session_state
    ss.setup = copy.deepcopy(REFERENCE_SETUP)
    ss.baseline = copy.deepcopy(REFERENCE_SETUP)
    ss.baseline_source = REFERENCE_LABEL
    ss.baseline_saved_at = None
    ss.baseline_persisted = False
    try:
        delete_setup()
        ss.store_problem = None
    except OSError as exc:
        ss.store_problem = f"The saved baseline file couldn't be deleted ({exc}). It may come back on refresh."
    push_setup_to_widgets(ss.setup)
    _rerun_if_results()
    ss.chat.append({"role": "assistant", "content": "Restored the reference setup as both the current "
                                                    "setup and the baseline, and cleared the saved baseline."})


def on_chat_submit() -> None:
    ss = st.session_state
    text = (ss.get("chat_box") or "").strip()
    if not text:
        return
    pull_widgets_into_setup(ss.setup)
    ss.chat.append({"role": "user", "content": text})

    old_setup = copy.deepcopy(ss.setup)
    parsed = interpret_setup_request(text, ss.setup)
    lines: list[str] = []

    if parsed.changes:
        ss.setup = parsed.setup_dict
        slider_notes = push_setup_to_widgets(ss.setup)
        lines.append("**Changes made:**")
        for change in parsed.changes:
            lines.append(f"- {change.description.replace('->', '→')}")
            lines.extend(f"    - *{note}*" for note in change.notes)
        lines.extend(f"- *{note}*" for note in slider_notes)
    else:
        lines.append("I couldn't find a setup change in that request, so nothing was changed.")
    if parsed.unparsed:
        lines.append("\n**Not understood:**")
        lines.extend(f"- {u}" for u in parsed.unparsed)
    if parsed.warnings:
        lines.append("\n**Warnings:**")
        lines.extend(f"- {w}" for w in parsed.warnings)

    if parsed.changes:
        cond = current_conditions()
        try:
            before = run_experiment(old_setup, cond)
            ss.result = run_experiment(ss.setup, cond)
            lines.append(f"\n**Re-ran the test** ({describe_conditions(cond)}):\n")
            lines.append("| Result | Before | After |")
            lines.append("|---|---|---|")
            after_metrics = key_metrics(ss.result)
            for name, value in key_metrics(before).items():
                lines.append(f"| {name} | {value} | {after_metrics[name]} |")
        except ValueError as exc:
            ss.setup = old_setup
            push_setup_to_widgets(ss.setup)
            lines.append(f"\nThat setup isn't physically valid ({exc}), so I undid the change.")
    else:
        lines.append("\nTry something like: " + ", ".join(f"*{c}*" for c in EXAMPLE_COMMANDS[4:]))

    ss.chat.append({"role": "assistant", "content": "\n".join(lines)})


# =============================================================================
# Numerical results
# =============================================================================
@dataclass(frozen=True)
class Metric:
    """One numerical result. ``points=True`` means a percentage whose change is
    shown in percentage points rather than percent-of-percent."""

    label: str
    value: float
    unit: str = ""
    decimals: int = 1
    help: str | None = None
    points: bool = False
    signed: bool = False

    def text(self) -> str:
        if self.value is None or math.isnan(self.value):
            return "—"
        sign = "+" if self.signed else ""
        return f"{self.value:{sign},.{self.decimals}f}{self.unit}"

    def delta(self, base: Metric) -> float:
        return self.value - base.value

    def delta_text(self, base: Metric) -> str:
        d = self.delta(base)
        if math.isnan(d):
            return "—"
        unit = " pts" if self.points else self.unit
        return f"{d:+,.{self.decimals}f}{unit}"


def _deg(x: float) -> float:
    return math.degrees(x) if x is not None and not math.isnan(x) else math.nan


def limiting_axle_text(axle: str | None) -> str:
    return "front and rear" if axle == "both" else str(axle)


def balance_label(understeer_angle: float) -> str:
    if math.isnan(understeer_angle):
        return "—"
    deg = math.degrees(understeer_angle)
    if deg > 0.05:
        return "Understeer"
    if deg < -0.05:
        return "Oversteer"
    return "Neutral"


def result_metrics(res: dict[str, Any]) -> list[Metric]:
    s = res["state"]
    if is_cornering(res):
        return [
            Metric("Lateral acceleration", abs(s.lateral_accel_g), " g", 2),
            Metric("Steering wheel angle", abs(_deg(s.handwheel_angle)), "°", 1),
            Metric("Body roll", abs(_deg(s.roll_angle)), "°", 2),
            Metric(f"Balance: {balance_label(s.understeer_angle)}", _deg(s.understeer_angle), "°", 2,
                   help="Understeer angle = front slip angle minus rear slip angle. "
                        "Positive = understeer, negative = oversteer.", signed=True),
            Metric("Max speed on this radius", res["max_speed"] * 3.6, " km/h", 1),
            Metric("Front LLTD", s.load_transfer.lltd_front * 100, "%", 1, points=True,
                   help="Front axle's share of the total lateral load transfer."),
            Metric("Total downforce", s.aero.downforce_total, " N", 0),
            Metric("Aero balance (front)", s.aero.front_balance * 100, "%", 1, points=True),
            Metric("Front slip angle", abs(_deg(s.front_slip_angle)), "°", 2),
            Metric("Rear slip angle", abs(_deg(s.rear_slip_angle)), "°", 2),
        ]
    return [
        Metric("Drive force needed", s.tractive_force_required, " N", 0),
        Metric("Power needed", s.power_required / 1000, " kW", 1),
        Metric("Aero drag", s.aero.drag, " N", 0),
        Metric("Rolling resistance", s.rolling_resistance, " N", 0),
        Metric("Load moved to rear", s.longitudinal_load_transfer, " N", 0),
        Metric("Total downforce", s.aero.downforce_total, " N", 0),
        Metric("Aero balance (front)", s.aero.front_balance * 100, "%", 1, points=True),
        Metric("Front ride height", s.front_ride_height * 1000, " mm", 1),
        Metric("Rear ride height", s.rear_ride_height * 1000, " mm", 1),
    ]


def key_metrics(res: dict[str, Any]) -> dict[str, str]:
    """Text metrics used in the chat before/after table."""
    out: dict[str, str] = {}
    s = res["state"]
    if is_cornering(res):
        out["Can make the corner?"] = (
            "Yes" if s.is_feasible else f"No ({limiting_axle_text(s.limiting_axle)} grip limit)"
        )
    for m in result_metrics(res):
        if m.label.startswith("Balance"):
            out["Balance"] = f"{balance_label(s.understeer_angle)} ({m.text()})"
        else:
            out[m.label] = m.text()
    return out


def render_metric_grid(metrics: list[Metric], base: list[Metric] | None, per_row: int = 5) -> None:
    for start in range(0, len(metrics), per_row):
        cols = st.columns(per_row)
        for col, i in zip(cols, range(start, min(start + per_row, len(metrics)))):
            m = metrics[i]
            if base is None:
                col.metric(m.label, m.text(), help=m.help)
            else:
                b = base[i]
                col.metric(m.label, m.text(), m.delta_text(b), delta_color="off",
                           help=f"Baseline: {b.text()}" + (f"\n\n{m.help}" if m.help else ""))


def comparison_rows(metrics: list[Metric], base: list[Metric]) -> list[dict[str, str]]:
    return [
        {"Result": m.label.split(":")[0], "Baseline": b.text(), "New": m.text(), "Change": m.delta_text(b)}
        for m, b in zip(metrics, base)
    ]


def render_markdown_table(rows: list[dict[str, str]], columns: tuple[str, ...]) -> None:
    def cell(value: str) -> str:
        return value.replace("|", r"\|").replace("\n", " ")

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(cell(row[column]) for column in columns) + " |" for row in rows]
    st.markdown("\n".join([header, separator, *body]))


# =============================================================================
# Charts: absolute views
# =============================================================================
def _base_layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.0, "font": {"size": 15}},
        height=height,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.3, "x": 0.0},
        template="plotly_white",
    )
    return fig


def _zero_line(fig: go.Figure) -> None:
    fig.add_hline(y=0, line_color="gray", line_width=1)


def fig_lltd(state, setup: VehicleSetup) -> go.Figure:
    lt = state.load_transfer
    axles = ["Front axle", "Rear axle"]
    geo = [abs(lt.front_geometric), abs(lt.rear_geometric)]
    ela = [abs(lt.front_elastic), abs(lt.rear_elastic)]
    fig = go.Figure([
        go.Bar(name="Geometric (through roll centres)", x=axles, y=geo, marker_color="#9ecae1",
               hovertemplate="%{x}<br>Geometric: %{y:,.0f} N<extra></extra>"),
        go.Bar(name="Elastic (springs + anti-roll bars)", x=axles, y=ela, marker_color="#3182bd",
               hovertemplate="%{x}<br>Elastic: %{y:,.0f} N<extra></extra>"),
    ])
    for axle, total in zip(axles, [g + e for g, e in zip(geo, ela)]):
        fig.add_annotation(x=axle, y=total, text=f"<b>{total:,.0f} N</b>", showarrow=False, yshift=12)
    fig.update_layout(barmode="stack", yaxis_title="Load moved inside → outside [N]")
    return _base_layout(fig, f"Lateral Load Transfer Distribution: front {lt.lltd_front * 100:.1f}% "
                             f"(static front weight {setup.weight_distribution_front * 100:.1f}%)")


_WHEEL_NAMES = {"FL": "Front left", "FR": "Front right", "RL": "Rear left", "RR": "Rear right"}


def fig_wheel_loads(state, setup: VehicleSetup) -> go.Figure:
    wheels = list(_WHEEL_NAMES)
    parked = static_wheel_loads(setup.mass, setup.weight_distribution_front, setup.gravity)
    loads = [state.tire_vertical_loads[w] for w in wheels]
    fig = go.Figure([
        go.Bar(name="Now", x=[_WHEEL_NAMES[w] for w in wheels], y=loads,
               marker_color=[WHEEL_COLORS[w] for w in wheels],
               text=[f"{v:,.0f} N" for v in loads], textposition="outside",
               hovertemplate="%{x}: %{y:,.0f} N<extra></extra>"),
        go.Scatter(name="Parked (no motion)", x=[_WHEEL_NAMES[w] for w in wheels], y=[parked[w] for w in wheels],
                   mode="markers", marker={"symbol": "line-ew-open", "size": 40, "color": "black",
                                           "line": {"width": 3}},
                   hovertemplate="%{x} parked: %{y:,.0f} N<extra></extra>"),
    ])
    fig.update_layout(yaxis_title="Vertical load F_z [N]")
    fig.update_yaxes(range=[0, max(loads + list(parked.values())) * 1.2])
    return _base_layout(fig, "Tire vertical loads")


def fig_aero_balance(setup_json: str, speed_kmh: float, aero) -> go.Figure:
    speeds, front, rear = _aero_sweep(setup_json)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.38, 0.62],
                        specs=[[{"type": "domain"}, {"type": "xy"}]],
                        subplot_titles=(f"Split at {speed_kmh:.0f} km/h", "Downforce vs. speed"))
    if aero.downforce_total > 1e-6:
        fig.add_trace(go.Pie(labels=["Front axle", "Rear axle"],
                             values=[aero.downforce_front, aero.downforce_rear], hole=0.55,
                             marker={"colors": [FRONT_COLOR, REAR_COLOR]}, sort=False,
                             texttemplate="%{label}<br>%{percent}", textposition="outside",
                             hovertemplate="%{label}: %{value:,.0f} N<extra></extra>", showlegend=False),
                      row=1, col=1)
        fig.add_annotation(text=f"<b>{aero.downforce_total:,.0f} N</b><br>total", x=0.155, y=0.5,
                           xref="paper", yref="paper", showarrow=False)
    else:
        fig.add_annotation(text="No downforce<br>at this speed", x=0.155, y=0.5,
                           xref="paper", yref="paper", showarrow=False)
    fig.add_trace(go.Scatter(x=speeds, y=front, name="Front axle", mode="lines",
                             line={"color": FRONT_COLOR, "width": 3}, stackgroup="aero",
                             hovertemplate="%{x:.0f} km/h: %{y:,.0f} N front<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=speeds, y=rear, name="Rear axle", mode="lines",
                             line={"color": REAR_COLOR, "width": 3}, stackgroup="aero",
                             hovertemplate="%{x:.0f} km/h: %{y:,.0f} N rear<extra></extra>"), row=1, col=2)
    fig.add_shape(type="line", x0=speed_kmh, x1=speed_kmh, y0=0, y1=1, xref="x", yref="y domain",
                  line={"dash": "dash", "color": "gray"})
    fig.update_xaxes(title_text="Speed [km/h]", row=1, col=2)
    fig.update_yaxes(title_text="Downforce [N] (stacked)", row=1, col=2)
    balance = aero.front_balance * 100
    return _base_layout(fig, f"Aero balance: {balance:.1f}% front / {100 - balance:.1f}% rear", height=400)


def _tire_operating_point(state, wheel: str) -> tuple[float, float]:
    slip = state.front_slip_angle if wheel.startswith("F") else state.rear_slip_angle
    return abs(math.degrees(slip)), abs(state.tire_lateral_forces[wheel]) / 1000.0


def fig_tire_curves(state, setup: VehicleSetup, cornering: bool, base_state=None,
                    base_setup: VehicleSetup | None = None) -> go.Figure:
    """Tire curves at each tire's current load, with operating points.

    If a baseline is given, its curves are overlaid faded and its operating
    points are hollow, with an arrow to where each tire works now.
    """
    alpha_deg = np.linspace(0.0, 15.0, 151)
    alpha = np.radians(alpha_deg)
    compare = base_state is not None and base_setup is not None
    outside_right = cornering and state.lateral_accel >= 0
    show_points = cornering and state.is_feasible
    show_base_points = compare and cornering and base_state.is_feasible
    fig = go.Figure()

    for i, wheel in enumerate(("FL", "FR", "RL", "RR")):
        color = WHEEL_COLORS[wheel]
        is_outside = cornering and (wheel.endswith("R") == outside_right)
        side = "" if not cornering else (" outside" if is_outside else " inside")
        dash = "solid" if (compare or is_outside or not cornering) else "dot"

        if compare:
            b_tire = base_setup.front_tire if wheel.startswith("F") else base_setup.rear_tire
            b_fz = base_state.tire_vertical_loads[wheel]
            b_fy = np.array([b_tire.lateral_force(a, b_fz) for a in alpha]) / 1000.0
            fig.add_trace(go.Scatter(
                x=alpha_deg, y=b_fy, mode="lines", name="Baseline (faded)", legendgroup="baseline",
                showlegend=i == 0, opacity=0.35, line={"color": color, "width": 2, "dash": "dash"},
                hovertemplate=f"{wheel} baseline: " + "α = %{x:.1f}°, F_y = %{y:.2f} kN<extra></extra>",
            ))

        tire = setup.front_tire if wheel.startswith("F") else setup.rear_tire
        fz = state.tire_vertical_loads[wheel]
        fy = np.array([tire.lateral_force(a, fz) for a in alpha]) / 1000.0
        fig.add_trace(go.Scatter(
            x=alpha_deg, y=fy, mode="lines", name=f"{wheel}{side} (F_z = {fz / 1000:.1f} kN)",
            line={"color": color, "width": 3, "dash": dash},
            hovertemplate=f"{wheel}: " + "α = %{x:.1f}°, F_y = %{y:.2f} kN<extra></extra>",
        ))

        if show_base_points:
            bx, by = _tire_operating_point(base_state, wheel)
            fig.add_trace(go.Scatter(
                x=[bx], y=[by], mode="markers", showlegend=False,
                marker={"size": 12, "symbol": "circle-open", "color": color, "line": {"width": 2}},
                hovertemplate=f"{wheel} baseline<br>" + "α = %{x:.2f}°<br>F_y = %{y:.2f} kN<extra></extra>",
            ))
        if show_points:
            x, y = _tire_operating_point(state, wheel)
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers", showlegend=False,
                marker={"size": 13, "color": color, "line": {"color": "black", "width": 2}},
                hovertemplate=f"{wheel} now<br>" + "α = %{x:.2f}°<br>F_y = %{y:.2f} kN<extra></extra>",
            ))
            if show_base_points and (abs(x - bx) > 0.01 or abs(y - by) > 0.005):
                fig.add_annotation(x=x, y=y, ax=bx, ay=by, xref="x", yref="y", axref="x", ayref="y",
                                   showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                                   arrowcolor=color, text="")

    fig.update_layout(xaxis_title="Slip angle α [deg]", yaxis_title="Cornering force F_y [kN]")
    if compare:
        title = "Tire slip vs. cornering force: baseline (hollow, faded) → new (filled)"
    elif cornering:
        title = "Tire slip vs. cornering force (dots = where each tire is working now)"
    else:
        title = "Tire slip vs. cornering force at the current loads"
    return _base_layout(fig, title, height=430)


# =============================================================================
# Charts: delta views (new minus baseline)
# =============================================================================
def _delta_colors(values: list[float]) -> list[str]:
    return [INCREASE_COLOR if v >= 0 else DECREASE_COLOR for v in values]


def fig_lltd_delta(state, base_state) -> go.Figure:
    lt, bl = state.load_transfer, base_state.load_transfer
    axles = ["Front axle", "Rear axle"]
    d_geo = [abs(lt.front_geometric) - abs(bl.front_geometric), abs(lt.rear_geometric) - abs(bl.rear_geometric)]
    d_ela = [abs(lt.front_elastic) - abs(bl.front_elastic), abs(lt.rear_elastic) - abs(bl.rear_elastic)]
    d_tot = [g + e for g, e in zip(d_geo, d_ela)]
    fig = go.Figure([
        go.Bar(name="Geometric Δ", x=axles, y=d_geo, marker_color="#9ecae1",
               hovertemplate="%{x}<br>Geometric change: %{y:+,.0f} N<extra></extra>"),
        go.Bar(name="Elastic Δ", x=axles, y=d_ela, marker_color="#3182bd",
               hovertemplate="%{x}<br>Elastic change: %{y:+,.0f} N<extra></extra>"),
        go.Bar(name="Total Δ", x=axles, y=d_tot, marker_color="#08306b",
               text=[f"{v:+,.0f} N" for v in d_tot], textposition="outside",
               hovertemplate="%{x}<br>Total change: %{y:+,.0f} N<extra></extra>"),
    ])
    _zero_line(fig)
    fig.update_layout(barmode="group", yaxis_title="Change in load transfer [N]")
    new, base = lt.lltd_front * 100, bl.lltd_front * 100
    return _base_layout(fig, f"Change in load transfer: front LLTD {new:.1f}% "
                             f"({new - base:+.1f} pts vs baseline {base:.1f}%)")


def fig_wheel_loads_delta(state, base_state) -> go.Figure:
    wheels = list(_WHEEL_NAMES)
    deltas = [state.tire_vertical_loads[w] - base_state.tire_vertical_loads[w] for w in wheels]
    fig = go.Figure(go.Bar(
        x=[_WHEEL_NAMES[w] for w in wheels], y=deltas, marker_color=_delta_colors(deltas),
        text=[f"{v:+,.0f} N" for v in deltas], textposition="outside",
        customdata=[[base_state.tire_vertical_loads[w], state.tire_vertical_loads[w]] for w in wheels],
        hovertemplate="%{x}<br>Baseline %{customdata[0]:,.0f} N → new %{customdata[1]:,.0f} N"
                      "<br>Change %{y:+,.0f} N<extra></extra>",
        showlegend=False,
    ))
    _zero_line(fig)
    fig.update_layout(yaxis_title="Change in vertical load [N]")
    span = max(abs(v) for v in deltas) or 1.0
    fig.update_yaxes(range=[-span * 1.35, span * 1.35])
    return _base_layout(fig, "Change in tire vertical loads vs baseline (green = more load)")


def fig_aero_delta(setup_json: str, base_json: str, speed_kmh: float, aero, base_aero) -> go.Figure:
    speeds, front, rear = _aero_sweep(setup_json)
    _, b_front, b_rear = _aero_sweep(base_json)
    d_front, d_rear = front - b_front, rear - b_rear
    fig = go.Figure([
        go.Scatter(x=speeds, y=d_front, name="Front axle Δ", mode="lines", line={"color": FRONT_COLOR, "width": 3},
                   hovertemplate="%{x:.0f} km/h: %{y:+,.0f} N front<extra></extra>"),
        go.Scatter(x=speeds, y=d_rear, name="Rear axle Δ", mode="lines", line={"color": REAR_COLOR, "width": 3},
                   hovertemplate="%{x:.0f} km/h: %{y:+,.0f} N rear<extra></extra>"),
        go.Scatter(x=speeds, y=d_front + d_rear, name="Total Δ", mode="lines",
                   line={"color": "gray", "width": 2, "dash": "dash"},
                   hovertemplate="%{x:.0f} km/h: %{y:+,.0f} N total<extra></extra>"),
    ])
    _zero_line(fig)
    fig.add_vline(x=speed_kmh, line_dash="dot", line_color="gray")
    fig.update_layout(xaxis_title="Speed [km/h]", yaxis_title="Change in downforce [N]")
    d_total = aero.downforce_total - base_aero.downforce_total
    bal, b_bal = aero.front_balance * 100, base_aero.front_balance * 100
    return _base_layout(fig, f"Change in downforce: {d_total:+,.0f} N at {speed_kmh:.0f} km/h; "
                             f"aero balance {bal:.1f}% front ({bal - b_bal:+.1f} pts)", height=400)


def build_figure(name: str, res: dict[str, Any], base_res: dict[str, Any] | None) -> go.Figure:
    """The chart ``name`` in absolute view, or delta view if ``base_res`` is given."""
    state, car, cornering = res["state"], car_of(res), is_cornering(res)
    speed = res["cond"]["speed_kmh"]
    if base_res is None:
        if name == CHART_LLTD:
            return fig_lltd(state, car)
        if name == CHART_LOADS:
            return fig_wheel_loads(state, car)
        if name == CHART_AERO:
            return fig_aero_balance(res["setup_json"], speed, state.aero)
        return fig_tire_curves(state, car, cornering)

    b_state = base_res["state"]
    if name == CHART_LLTD:
        return fig_lltd_delta(state, b_state)
    if name == CHART_LOADS:
        return fig_wheel_loads_delta(state, b_state)
    if name == CHART_AERO:
        return fig_aero_delta(res["setup_json"], base_res["setup_json"], speed, state.aero, b_state.aero)
    return fig_tire_curves(state, car, cornering, b_state, car_of(base_res))


# =============================================================================
# Educational text
# =============================================================================
THEORY_LLTD = r"""
**What it shows.** When the car corners, the sideways force $m\,a_y$ acts at the centre of
gravity, which is *above* the ground. That tries to tip the car outward, so load moves from the
inside tires to the outside tires. This chart shows **how much load moves at each axle**.

**The total is fixed** by physics:

$$\Delta F_{z,f}\,t_f + \Delta F_{z,r}\,t_r = m\,a_y\,h$$

Only mass $m$, CG height $h$ and track width $t$ change the total. Springs and anti-roll bars
only change **how it is shared** between front and rear.

**Two paths:**
- *Geometric:* through the suspension links at the roll centre height. Happens instantly.
- *Elastic:* through the springs and bars, shared by roll stiffness
  $\lambda_f = \dfrac{K_{\phi,f}}{K_{\phi,f}+K_{\phi,r}}$, where the roll stiffness (in N·m per degree of roll) is
  $K_\phi = \tfrac{\pi}{180}\cdot\tfrac12 k_w t^2 + K_{ARB}$. The $\tfrac{\pi}{180}$ turns the springs' part
  into "per degree", the same unit as the anti-roll bar sliders.

**Why it matters (tire load sensitivity).** Tires get less grippy per newton as load increases.
The axle with *more* load transfer loses more total grip. If the front LLTD is higher than the
static front weight distribution, the front loses grip first and the car **understeers**. If it is
lower, the car tends to **oversteer**.

**Delta view:** bars above zero mean more load moves at that axle than in the baseline. If you only
changed springs or bars, the front and rear changes roughly cancel, because the total is fixed!

**Try it:** stiffen the front anti-roll bar and watch the front bar grow and the balance move
toward understeer.
"""

THEORY_WHEEL_LOADS = r"""
**What it shows.** The vertical load on each tire now, compared with the car parked (black lines).

$$F_z = \underbrace{\tfrac12 m g\,w}_{\text{static}} + \underbrace{\tfrac12 F_{aero,axle}}_{\text{downforce}}
\pm \underbrace{\Delta F_z}_{\text{load transfer}}$$

In a **left** turn the **right** tires are on the outside and gain load. Under acceleration, load
moves to the rear: $\Delta F_{z} = m\,a_x\,h / L$. Downforce adds load at speed **without adding
mass**, which is why it increases cornering grip.

**Delta view:** green bars gained load compared with the baseline, red bars lost load.

If an inside tire reaches zero, that wheel lifts off the ground!
"""

THEORY_AERO = r"""
**What it shows.** How much downforce the car makes and how it is split between the axles.

$$F_{z,aero} = \tfrac12\,\rho\,V^2\,A\,C_L$$

- $\rho$ = air density (1.225 kg/m³), $V$ = speed (m/s), $A$ = frontal area (m²), $C_L$ = downforce coefficient.
- Downforce grows with the **square** of speed: double the speed gives **four times** the downforce.
  That is why the right-hand curve bends upward.

**Aero balance** is the front share of the downforce. More front downforce helps the front tires
(less understeer at high speed). More rear downforce makes the car more stable (less oversteer).
Engineers usually set the aero balance close to the weight distribution.

**Ride height matters.** Downforce compresses the springs, the car sits lower, and a lower floor
makes even more downforce (ground effect). The simulator solves this loop automatically. That is
why changing *springs* can change downforce in the delta view, even with the same wings.
"""

THEORY_TIRES = r"""
**What it shows.** How much sideways (cornering) force each tire makes as its **slip angle**
grows. The slip angle is the difference between where the tire points and where it actually travels.

- **Linear region** (small angles): $F_y \approx C_\alpha\,\alpha$, where $C_\alpha$ is the cornering stiffness
  (newtons per degree of slip) and $\alpha$ is in degrees.
- **Peak:** the most force the tire can make, about $\mu F_z$. Beyond it the tire is sliding.
- Each tire has its own curve because each carries a different load $F_z$. Heavily loaded
  (outside, solid) tires make more force, but *less than proportionally* more. That is
  **load sensitivity**.

**The dots** show where each tire is working right now. Both tires on an axle share the same slip
angle. If the front dots are closer to their peaks than the rear dots, the front will run out of
grip first, which means **understeer**.

**Delta view:** faded dashed curves and hollow dots are the baseline; arrows show how each tire's
working point moved. A difference between two curves isn't meaningful, so this chart overlays them.

The steering angle you need (in degrees) is $\delta = \dfrac{180}{\pi}\cdot\dfrac{L}{R} + \alpha_f - \alpha_r$.
"""

THEORY_EXPERIMENT = r"""
**Constant-radius cornering** (steady state): the car drives around a circle of radius $R$ at a
constant speed $V$. It needs a sideways acceleration $a_y = V^2/R$, so the tires must
provide $F_y = m\,a_y$. The simulator finds the loads, slip angles, steering and roll that balance
every force and moment. If the tires can't provide enough force, the car can't make the corner.

**Straight-line acceleration** (one instant): the drive force must overcome drag, rolling
resistance **and** accelerate the mass: $F_x = m\,a_x + \tfrac12\rho V^2 A C_D + C_{rr}\sum F_z$.
Power is $P = F_x V$.

*Units:* 1 g = 9.81 m/s². 100 km/h = 27.8 m/s.
"""

THEORY_COMPARE = """
- **Reference setup:** the original, known-good car. It never changes, so you can always get back to a car that works.
- **Baseline:** the setup your results are compared against. It starts as the reference.
  When you find a setup you like, press **Save as new baseline** in the sidebar and give it a name.
  A saved baseline stays after a refresh or restart, and the app opens with it loaded.
- **Current setup:** what the sliders and the chat change.

After you run an experiment, each results section has its own **Compare to baseline** switch, and the
two work independently:
- In the **numbers** section it shows each number's change from the baseline underneath it, plus a
  comparison table.
- In the **plots** section it switches the charts to showing *changes* (new minus baseline).

The switches are shared between the Experiment Runner and the chat tab's live results.

For example, keep the plots absolute to see what the car is doing, while the numbers show what your
change did. Both setups are run under exactly the same test conditions, so any difference comes from
the setup alone.
"""


# The popover content is drawn a moment after it opens, so keep trying briefly.
FOCUS_BASELINE_NAME_JS = """
<script>
(() => {
  let tries = 0;
  const focusName = () => {
    const box = document.querySelector(".st-key-baseline_name_input input");
    if (box && box.offsetParent !== null) { box.focus(); return; }
    if (++tries < 40) setTimeout(focusName, 25);
  };
  focusName();
})();
</script>
"""


# Streamlit reserves a full header row in the sidebar for an app logo. This app
# has no logo, so that row is empty space above "Vehicle Setup". Collapse it.
# The collapse control stays an overlay so the sidebar can still be closed.
SIDEBAR_HEADER_CSS = """
<style>
[data-testid="stSidebarHeader"] {
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: visible !important;
    position: relative;
    z-index: 2;
}
[data-testid="stLogoSpacer"] {
    display: none !important;
}
[data-testid="stSidebarCollapseButton"] {
    position: absolute;
    top: 0.55rem;
    right: 0.25rem;
}
</style>
"""


# =============================================================================
# Layout pieces
# =============================================================================
def render_sidebar() -> None:
    ss = st.session_state
    sb = st.sidebar
    pull_widgets_into_setup(ss.setup)
    diffs = setup_differences(ss.setup, ss.baseline)

    sb.title("Vehicle Setup")
    sb.caption("Adjust the car, then run an experiment. The chat can change these too.")

    with sb.container(border=True):
        st.markdown(f"**Baseline:** {ss.baseline_source}")
        if ss.baseline_persisted and ss.baseline_saved_at:
            st.caption(f"Saved {ss.baseline_saved_at:%b %d, %H:%M} on this computer, "
                       "so it stays after a refresh or restart.")
        if ss.store_problem:
            st.warning(ss.store_problem)
        if diffs:
            n = len(diffs)
            st.markdown(f":orange[**{n} change{'s' if n != 1 else ''}** from the baseline]")
            with st.expander("Show changes"):
                for d in diffs:
                    st.markdown(f"**{d.label}**  \n{d.baseline} → {d.current}  \n:orange[{d.change}]")
        else:
            st.caption("The current setup matches the baseline.")
        c1, c2 = st.columns(2)
        c1.button("Revert to baseline", on_click=on_revert_to_baseline, disabled=not diffs, width="stretch",
                  help="Throw away the current changes and go back to the baseline.")
        with c2.popover("Save as new baseline", disabled=not diffs, width="stretch",
                        help="Name the current setup and make it the new comparison point.",
                        key="save_baseline_popover", on_change="rerun") as save_popover:
            with st.form("save_baseline_form", border=False, enter_to_submit=True):
                st.text_input("Baseline name", key="baseline_name_input", max_chars=60,
                              placeholder="e.g. Stiff front ARB",
                              help="Leave blank to name it by the date and time it was saved.")
                st.form_submit_button("Save baseline", type="primary",
                                      on_click=on_save_baseline, width="stretch")
            if save_popover.open:
                st.html(FOCUS_BASELINE_NAME_JS, unsafe_allow_javascript=True)

    for section in dict.fromkeys(s.section for s in SLIDERS):
        sb.subheader(section)
        for spec in (s for s in SLIDERS if s.section == section):
            label = f"{spec.label} [{spec.unit}]" if spec.unit else spec.label
            sb.slider(label, min_value=spec.min_value, max_value=spec.max_value, step=spec.step,
                      format=spec.fmt, help=spec.help, key=spec.widget_key)
            base = _display_value(ss.baseline, spec)
            cur = _display_value(ss.setup, spec)
            if _differs(base, cur):
                change = _change_text(base, cur, spec.unit, spec.decimals, points=spec.unit == "%")
                sb.caption(f":orange[{change} · baseline {spec.fmt % base}]")

    car = dict_to_setup(ss.setup, BASE_SETUP)
    k_f = car.front_suspension.roll_stiffness
    k_r = car.rear_suspension.roll_stiffness
    sb.divider()
    sb.markdown("**Derived values**")
    c1, c2 = sb.columns(2)
    c1.metric("Roll stiffness F", f"{k_f * NM_PER_RAD_TO_DEG:,.0f} N·m/deg")
    c2.metric("Roll stiffness R", f"{k_r * NM_PER_RAD_TO_DEG:,.0f} N·m/deg")
    c1.metric("Front roll share", f"{roll_moment_distribution(k_f, k_r) * 100:.1f}%")
    c2.metric("Front wheel rate", f"{car.front_suspension.wheel_rate / 1000:.1f} N/mm")

    sb.divider()
    sb.markdown("**Reference setup**")
    sb.caption("The original, known-good car. Restoring it resets both the current setup and the baseline, "
               "and clears the saved baseline.")
    at_reference = setups_equal(ss.setup, REFERENCE_SETUP) and setups_equal(ss.baseline, REFERENCE_SETUP)
    sb.button("Restore reference setup", on_click=on_restore_reference, disabled=at_reference, width="stretch")


def render_experiment_inputs() -> None:
    st.radio("Test type", [CORNERING, STRAIGHT], key="exp_mode", horizontal=True)
    c1, c2, c3 = st.columns(3)
    c1.slider("Speed [km/h]", 0.0, 250.0, step=1.0, format="%.0f", key="exp_speed")
    if st.session_state.exp_mode == CORNERING:
        c2.number_input("Corner radius [m]", min_value=10.0, max_value=1000.0, step=5.0, format="%.0f",
                        key="exp_radius")
        c3.radio("Turn direction", ["Left", "Right"], key="exp_dir", horizontal=True)
    else:
        c2.slider("Acceleration [g]", -1.5, 1.0, step=0.05, key="exp_accel",
                  help="Positive = speeding up, negative = braking.")
    st.button("Run Simulation", type="primary", on_click=on_run)
    with st.expander("What does this experiment do?"):
        st.markdown(THEORY_EXPERIMENT)


COMPARE_HELP = {
    "metrics": "Show each numerical result with its change from the baseline setup.",
    "plots": "Switch the charts from absolute values to changes from the baseline setup.",
}


def _sync_compare(part: str, widget_key: str) -> None:
    st.session_state[f"delta_{part}"] = st.session_state[widget_key]


def compare_toggle(part: str, where: str) -> None:
    """Draw the "Compare to baseline" switch for ``part`` ("metrics" or "plots").

    The same switch appears in both tabs, and a widget key can only be drawn
    once per run, so each copy gets its own key and writes through to the
    shared ``delta_<part>`` setting.
    """
    ss = st.session_state
    widget_key = f"cmp_{part}_{where}"
    ss[widget_key] = ss[f"delta_{part}"]
    st.toggle("Compare to baseline", key=widget_key, help=COMPARE_HELP[part],
              on_change=_sync_compare, args=(part, widget_key))


def baseline_results(res: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Baseline run under the same conditions as ``res``, split by what is compared.

    Returns ``(for_metrics, for_plots)``; each is ``None`` when that part of the
    page should show absolute values.
    """
    ss = st.session_state
    if not (ss.delta_metrics or ss.delta_plots):
        return None, None
    base_res = run_experiment(ss.baseline, res["cond"])
    return (base_res if ss.delta_metrics else None), (base_res if ss.delta_plots else None)


def comparison_caption(metric_base: dict | None, plot_base: dict | None) -> str:
    parts = [p for p, on in (("numbers", metric_base), ("plots", plot_base)) if on]
    if not parts:
        return ""
    return f" · {' and '.join(parts)} compared with baseline ({st.session_state.baseline_source})"


def render_results(res: dict[str, Any]) -> None:
    ss = st.session_state
    state = res["state"]
    cornering = is_cornering(res)
    metric_base, plot_base = baseline_results(res)
    base_res = metric_base or plot_base

    stale = res["setup_json"] != _setup_json(ss.setup) or res["cond_json"] != json.dumps(
        current_conditions(), sort_keys=True)
    if stale:
        st.info("The setup or test conditions have changed since this run. Press **Run Simulation** to update.")

    st.markdown(f"**Results:** {describe_conditions(res['cond'])}" + comparison_caption(metric_base, plot_base))
    if base_res is not None and setups_equal(json.loads(res["setup_json"]), ss.baseline):
        st.info("This result used the same setup as the baseline, so every change is zero. "
                "Change the setup and run again to see differences.")

    if cornering:
        if not state.is_feasible:
            st.error(f"The car **cannot** hold this corner: the **{limiting_axle_text(state.limiting_axle)}** "
                     f"tires ran out of grip. Maximum speed on this radius is about "
                     f"**{res['max_speed'] * 3.6:.1f} km/h**.")
        if base_res is not None and not base_res["state"].is_feasible:
            st.warning("The **baseline** setup cannot hold this corner, so some changes can't be calculated.")
        if state.wheel_lift:
            st.warning("An inside wheel has lifted off the ground. The results are beyond what this model covers.")

    compare_toggle("metrics", "exp")
    metrics = result_metrics(res)
    base_metrics = result_metrics(metric_base) if metric_base else None
    render_metric_grid(metrics, base_metrics)
    if base_metrics:
        with st.expander("Comparison table: baseline vs new"):
            render_markdown_table(comparison_rows(metrics, base_metrics),
                                  ("Result", "Baseline", "New", "Change"))
            diffs = setup_differences(json.loads(res["setup_json"]), ss.baseline)
            if diffs:
                st.markdown("**Setup changes behind these results**")
                render_markdown_table(
                    [{"Parameter": d.label, "Baseline": d.baseline, "New": d.current, "Change": d.change}
                     for d in diffs],
                    ("Parameter", "Baseline", "New", "Change"),
                )

    st.divider()
    compare_toggle("plots", "exp")
    left, right = st.columns(2)
    with left:
        if cornering:
            st.plotly_chart(build_figure(CHART_LLTD, res, plot_base), width="stretch", key="exp_lltd")
        else:
            st.info("Lateral load transfer only happens in a corner. Run a cornering test to see this chart.")
        with st.expander("Theory: Lateral Load Transfer Distribution"):
            st.markdown(THEORY_LLTD)
    with right:
        st.plotly_chart(build_figure(CHART_LOADS, res, plot_base), width="stretch", key="exp_loads")
        with st.expander("Theory: Tire vertical loads"):
            st.markdown(THEORY_WHEEL_LOADS)

    st.plotly_chart(build_figure(CHART_AERO, res, plot_base), width="stretch", key="exp_aero")
    with st.expander("Theory: Aero balance and downforce"):
        st.markdown(THEORY_AERO)

    st.plotly_chart(build_figure(CHART_TIRES, res, plot_base), width="stretch", key="exp_tires")
    if cornering and not state.is_feasible:
        st.caption("No operating points are shown for the new setup because its tires cannot produce "
                   "the force this corner needs.")
    with st.expander("Theory: Tire slip vs. cornering force"):
        st.markdown(THEORY_TIRES)


def render_chat_tab() -> None:
    ss = st.session_state
    chat_col, live_col = st.columns([5, 4], gap="large")
    with chat_col:
        st.caption("Type setup changes in plain English. Each change re-runs the current experiment "
                   "(set it up in the Experiment Runner tab).")
        history = st.container(height=520)
        with history:
            for msg in ss.chat:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        st.chat_input("e.g. Stiffen front springs by 20% and add 50 kg of ballast",
                      key="chat_box", on_submit=on_chat_submit)
        with st.expander("What can I ask for?"):
            st.markdown(
                "You can change **springs, anti-roll bars, ride height, CG height, roll centre height, "
                "track width, mass/ballast, weight distribution, tire grip, downforce and drag**.\n\n"
                "- Use *front*, *rear* or *front and rear* (default: both axles).\n"
                "- Use percentages (*by 20%*), amounts (*by 5 mm*, *50 kg*), targets (*to 80 N/mm*) "
                "or *double* / *halve*.\n"
                "- Split several changes with *and* or commas.\n\n"
                "**Note:** *kg of downforce* means aerodynamic force (kilogram-force) at a reference speed. "
                "To make the car heavier, ask for *ballast* or *mass*."
            )
    with live_col:
        st.markdown("#### Live results")
        res = ss.result
        if res is None:
            st.info("No results yet. Send a setup change, or press **Run Simulation** in the Experiment Runner tab.")
            return
        metric_base, plot_base = baseline_results(res)
        st.markdown(f"*{describe_conditions(res['cond'])}*" + comparison_caption(metric_base, plot_base))
        compare_toggle("metrics", "chat")
        metrics = result_metrics(res)
        render_metric_grid(metrics, result_metrics(metric_base) if metric_base else None, per_row=2)
        options = ([CHART_LLTD, CHART_TIRES, CHART_AERO, CHART_LOADS] if is_cornering(res)
                   else [CHART_LOADS, CHART_AERO, CHART_TIRES])
        chart_col, toggle_col = st.columns([3, 2], vertical_alignment="bottom")
        choice = chart_col.selectbox("Chart", options, key="chat_chart")
        with toggle_col:
            compare_toggle("plots", "chat")
        st.plotly_chart(build_figure(choice, res, plot_base), width="stretch", key="chat_live_chart")


# =============================================================================
# Page
# =============================================================================
def main() -> None:
    init_state()
    st.html(SIDEBAR_HEADER_CSS)
    render_sidebar()

    st.title("Vehicle Dynamics Lab")
    st.caption("Explore how mass, springs, anti-roll bars and aerodynamics change the way a race car corners.")

    with st.expander("Reference, baseline and current setup: how comparison works"):
        st.markdown(THEORY_COMPARE)

    tab_exp, tab_chat = st.tabs(["Experiment Runner", "Setup Assistant (Chat)"])
    with tab_exp:
        render_experiment_inputs()
        st.divider()
        if st.session_state.result is None:
            st.info("Choose a test above and press **Run Simulation**.")
        else:
            render_results(st.session_state.result)
    with tab_chat:
        render_chat_tab()


main()
