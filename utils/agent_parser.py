"""
Natural-language setup assistant: turn "stiffen the front springs by 20 %"
into concrete, validated changes to a vehicle setup.

Pipeline
--------
::

    request text ──► IntentParser ──► [SetupChange, ...] ──► apply_changes ──► new setup dict
                     (rules or LLM)    (structured intent)    (units, maths,     + summary
                                                               limits)

1. **Intent parsing** turns free text into a list of :class:`SetupChange`
   objects (parameter, axle, operation, value, unit). Two interchangeable
   parsers are provided:

   * :class:`RuleBasedIntentParser` -- regex-based, offline, deterministic.
     Good for classrooms with no internet and for unit tests.
   * :class:`LLMIntentParser` -- sends a structured-JSON prompt to *any* LLM
     through a plain callable, then validates the JSON it gets back. Handles
     looser phrasing ("the back end feels too stiff, take 10 % out").

2. **Applying changes** is always done here in plain Python, never by the
   LLM: unit conversion, percentage maths, physical limits and the
   human-readable summary are deterministic and testable.

The setup dictionary
--------------------
:class:`physics.vehicle.VehicleSetup` contains tire-model and aero-map objects
that do not serialise nicely, so this module works on a plain-number
dictionary instead (all SI units). Use :func:`setup_to_dict` and
:func:`dict_to_setup` to convert::

    {
      "mass": 1200.0, "wheelbase": 2.6, "weight_distribution_front": 0.52,
      "cg_height": 0.45, "static_front_ride_height": 0.07,
      "static_rear_ride_height": 0.09, "steering_ratio": 15.0,
      "front_suspension": {"spring_rate": 60000.0, "motion_ratio": 0.8, ...},
      "rear_suspension":  {...},
      "front_tire": {"cornering_stiffness": 60000.0, "peak_friction": 1.2, ...},
      "rear_tire":  {...},
      "aero": {"frontal_area": 1.8, "air_density": 1.225,
               "cl_scale": 1.0, "cd_scale": 1.0,
               "reference_cl": 1.12, "reference_cd": 0.37,
               "reference_speed": 40.0}
    }

Downforce and drag are speed dependent, so "add 50 kg of downforce" is
interpreted **at a reference speed** (``aero.reference_speed``, or a speed in
the request such as "at 120 km/h") and implemented by scaling the whole aero
map's :math:`C_L` (``aero.cl_scale``).

Using an LLM
------------
Pass any function ``complete(system_prompt, user_prompt) -> str``. For example
with the OpenAI SDK::

    from openai import OpenAI
    client = OpenAI()

    def complete(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content

    new_setup, summary = parse_setup_request(text, setup_dict, llm=complete)

For function-calling APIs, :data:`SETUP_CHANGES_TOOL_SCHEMA` is a ready-made
JSON schema; pass the tool-call arguments (as a JSON string) to
:func:`parse_llm_response`.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from physics.aero import AeroMap, AeroModel, dynamic_pressure
from physics.suspension import GRAVITY, AxleSuspension
from physics.tires import TireParameters
from physics.vehicle import VehicleSetup

__all__ = [
    "DEFAULT_REFERENCE_SPEED",
    "PARAMETERS",
    "SETUP_CHANGES_TOOL_SCHEMA",
    "AppliedChange",
    "IntentParser",
    "LLMIntentParser",
    "LLMResponseError",
    "ParamSpec",
    "ParseResult",
    "RuleBasedIntentParser",
    "SetupChange",
    "SetupChangeError",
    "apply_changes",
    "build_system_prompt",
    "build_user_prompt",
    "dict_to_setup",
    "interpret_setup_request",
    "parse_llm_response",
    "parse_setup_request",
    "setup_to_dict",
]

#: Speed at which "N of downforce/drag" requests are interpreted [m/s] (144 km/h).
DEFAULT_REFERENCE_SPEED = 40.0

OPERATIONS = ("scale_percent", "add", "set", "multiply")
AXLES = ("front", "rear", "both")


class SetupChangeError(ValueError):
    """A change could not be applied (unknown unit, missing key, ...)."""


class LLMResponseError(ValueError):
    """The LLM's reply was not valid JSON in the expected shape."""


# =============================================================================
# Units
# =============================================================================
_LBF = 4.4482216152605  # N
#: Setups store angular stiffnesses per radian (SI); people (and the app) use
#: per degree. Per-radian value x this = per-degree value.
_PER_RAD_TO_PER_DEG = math.pi / 180.0
_UNIT_FACTORS: dict[str, dict[str, float]] = {
    "linear_stiffness": {"n/m": 1.0, "n/mm": 1000.0, "kn/m": 1000.0, "lb/in": _LBF / 0.0254, "lbf/in": _LBF / 0.0254},
    "roll_stiffness": {"nm/rad": 1.0, "nm/deg": 180.0 / math.pi,
                       "knm/rad": 1000.0, "knm/deg": 1000.0 * 180.0 / math.pi},
    "cornering_stiffness": {"n/rad": 1.0, "n/deg": 180.0 / math.pi},
    "length": {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "in": 0.0254},
    "mass": {"kg": 1.0, "lb": 0.45359237},
    # "kg" / "lb" of force mean kilogram-force / pound-force (what people say for downforce).
    "force": {"n": 1.0, "kn": 1e3, "kg": GRAVITY, "kgf": GRAVITY, "lb": _LBF, "lbf": _LBF},
    "area": {"m2": 1.0},
    "dimensionless": {"": 1.0},
    "fraction": {"": 1.0, "%": 0.01},
}
_SPEED_FACTORS = {"m/s": 1.0, "km/h": 1.0 / 3.6, "kph": 1.0 / 3.6, "mph": 0.44704}

_UNIT_SYNONYMS = (
    ("percent", "%"),
    ("millimetres", "mm"),
    ("millimeters", "mm"),
    ("inches", "in"),
    ("inch", "in"),
    ("degrees", "deg"),
    ("degree", "deg"),
    ("°", "deg"),
    ("pounds", "lb"),
    ("pound", "lb"),
    ("lbs", "lb"),
    ("newtons", "n"),
    ("m²", "m2"),
    ("m^2", "m2"),
)


def _normalise_unit(unit: str | None) -> str:
    """Lower-case and canonicalise a unit string, e.g. ``"N·m/deg"`` -> ``"nm/deg"``."""
    u = re.sub(r"[\s·*.\-]", "", (unit or "").lower())
    for old, new in _UNIT_SYNONYMS:
        u = u.replace(old, new)
    return u


# =============================================================================
# Parameter registry
# =============================================================================
@dataclass(frozen=True)
class ParamSpec:
    """Describes one adjustable setup parameter.

    Attributes
    ----------
    key : str            Canonical name used in :class:`SetupChange`.
    label : str          Human-readable name for summaries.
    path : str           Dotted path into the setup dict; ``{axle}`` is
                         replaced by ``front``/``rear`` for per-axle values.
    dimension : str      Key into the unit table (``"length"``, ``"force"``, ...).
    si_unit : str        Unit stored in the dict.
    default_unit : str   Unit assumed when the request gives a bare number.
    bounds : tuple       (min, max) physically sensible range, SI units.
    display_unit : str   Unit used in the summary.
    display_factor : float  SI value x factor = display value.
    aliases : str        Regex matching how people refer to this parameter.
    """

    key: str
    label: str
    path: str
    dimension: str
    si_unit: str
    default_unit: str
    bounds: tuple[float, float]
    display_unit: str
    display_factor: float = 1.0
    aliases: str = ""

    @property
    def per_axle(self) -> bool:
        return "{axle}" in self.path


#: Ordered by matching priority: more specific phrases come first so that
#: "weight distribution" is not mistaken for "weight" (mass), etc.
PARAMETERS: tuple[ParamSpec, ...] = (
    ParamSpec("weight_distribution_front", "front weight distribution", "weight_distribution_front",
              "fraction", "-", "", (0.30, 0.70), "%", 100.0,
              r"weight\s*(?:distribution|bias|balance)|front\s*weight|(?:move|shift)\s+(?:the\s+)?weight"),
    ParamSpec("cg_height", "CG height", "cg_height", "length", "m", "mm", (0.15, 1.0), "mm", 1000.0,
              r"\bc\.?g\.?\b|\bcog\b|cent(?:er|re)\s+of\s+(?:gravity|mass)"),
    ParamSpec("roll_center_height", "roll centre height", "{axle}_suspension.roll_center_height",
              "length", "m", "mm", (0.0, 0.30), "mm", 1000.0, r"roll\s*cent(?:er|re)s?"),
    ParamSpec("anti_roll_bar_rate", "anti-roll bar rate", "{axle}_suspension.anti_roll_bar_rate",
              "roll_stiffness", "N·m/rad", "nm/deg", (0.0, 200_000.0), "N·m/deg", _PER_RAD_TO_PER_DEG,
              r"anti[-\s]?roll\s*bars?|\barbs?\b|sway\s*bars?|roll\s*bars?|stabili[sz]ers?"),
    ParamSpec("ride_height", "static ride height", "static_{axle}_ride_height",
              "length", "m", "mm", (0.0, 0.20), "mm", 1000.0, r"ride\s*heights?"),
    ParamSpec("cornering_stiffness", "tire cornering stiffness", "{axle}_tire.cornering_stiffness",
              "cornering_stiffness", "N/rad", "n/deg", (5_000.0, 300_000.0), "N/deg", _PER_RAD_TO_PER_DEG,
              r"cornering\s*stiffness"),
    ParamSpec("tire_vertical_stiffness", "tire vertical stiffness", "{axle}_suspension.tire_vertical_stiffness",
              "linear_stiffness", "N/m", "n/m", (50_000.0, 1_000_000.0), "N/m", 1.0,
              r"(?:tire|tyre)s?\s+(?:vertical\s+)?(?:stiffness|rate|springs?)"),
    ParamSpec("load_sensitivity", "tire load sensitivity", "{axle}_tire.load_sensitivity",
              "dimensionless", "-", "", (0.0, 0.5), "", 1.0, r"load\s*sensitivity"),
    ParamSpec("peak_friction", "tire peak friction", "{axle}_tire.peak_friction",
              "dimensionless", "-", "", (0.3, 3.0), "", 1.0, r"\bgrip\b|friction|\bmu\b"),
    ParamSpec("spring_rate", "spring rate", "{axle}_suspension.spring_rate",
              "linear_stiffness", "N/m", "n/m", (5_000.0, 500_000.0), "N/m", 1.0, r"springs?(?:\s*rates?)?"),
    ParamSpec("motion_ratio", "motion ratio", "{axle}_suspension.motion_ratio",
              "dimensionless", "-", "", (0.3, 1.5), "", 1.0, r"motion\s*ratio|installation\s*ratio"),
    ParamSpec("track_width", "track width", "{axle}_suspension.track_width",
              "length", "m", "mm", (1.0, 2.2), "mm", 1000.0, r"\btrack(?:\s*width)?\b"),
    ParamSpec("downforce", "downforce", "aero.cl_scale", "force", "N", "n", (0.0, 5.0), "N", 1.0,
              r"downforce"),
    ParamSpec("drag", "drag", "aero.cd_scale", "force", "N", "n", (0.1, 5.0), "N", 1.0, r"\bdrag\b"),
    ParamSpec("frontal_area", "frontal area", "aero.frontal_area", "area", "m²", "m2", (0.5, 4.0), "m²", 1.0,
              r"frontal\s*area"),
    ParamSpec("steering_ratio", "steering ratio", "steering_ratio", "dimensionless", "-", "", (5.0, 30.0), "", 1.0,
              r"steering\s*ratio"),
    ParamSpec("wheelbase", "wheelbase", "wheelbase", "length", "m", "mm", (1.5, 4.0), "mm", 1000.0,
              r"wheel\s*base"),
    ParamSpec("mass", "vehicle mass", "mass", "mass", "kg", "kg", (300.0, 3000.0), "kg", 1.0,
              r"\bmass\b|\bweight\b|ballast"),
)
PARAMS_BY_KEY: dict[str, ParamSpec] = {p.key: p for p in PARAMETERS}

#: Parameters stored as a scale on the aero map but requested as a force.
_AERO_FORCE_PARAMS = {"downforce": ("cl_scale", "reference_cl", "C_L"), "drag": ("cd_scale", "reference_cd", "C_D")}


# =============================================================================
# Data structures
# =============================================================================
@dataclass(frozen=True)
class SetupChange:
    """One structured change request (the output of an intent parser).

    Attributes
    ----------
    parameter : str
        Key from :data:`PARAMETERS`, e.g. ``"spring_rate"``.
    operation : str
        ``"scale_percent"`` -- change by ``value`` percent (signed, e.g. -10).
        ``"add"``           -- add ``value`` (signed) in ``unit``.
        ``"set"``           -- set to ``value`` in ``unit``.
        ``"multiply"``      -- multiply by ``value`` (e.g. 2 for "double").
    value : float
        Signed amount; meaning depends on ``operation``.
    unit : str
        Unit of ``value`` for ``add``/``set`` (``""`` = parameter default).
    axle : str or None
        ``"front"``, ``"rear"``, ``"both"`` or ``None`` (= both for per-axle
        parameters; ignored for whole-vehicle parameters).
    reference_speed : float or None
        Speed [m/s] at which downforce/drag amounts are meant.
    source_text : str
        The fragment of the request this change came from.
    """

    parameter: str
    operation: str
    value: float
    unit: str = ""
    axle: str | None = None
    reference_speed: float | None = None
    source_text: str = ""


@dataclass(frozen=True)
class AppliedChange:
    """Record of one value actually written to the setup dictionary."""

    label: str
    path: str
    old_value: float
    new_value: float
    description: str
    notes: tuple[str, ...] = ()


@dataclass
class ParseResult:
    """Everything produced by :func:`interpret_setup_request`."""

    setup_dict: dict[str, Any]
    summary: str
    changes: list[AppliedChange] = field(default_factory=list)
    unparsed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    parser: str = ""


# =============================================================================
# VehicleSetup <-> dict
# =============================================================================
def setup_to_dict(setup: VehicleSetup, reference_speed: float = DEFAULT_REFERENCE_SPEED) -> dict[str, Any]:
    """Convert a :class:`VehicleSetup` into the plain-number setup dictionary.

    ``aero.reference_cl`` / ``reference_cd`` are the map coefficients at the
    static ride heights; they let downforce requests in newtons be converted
    to a map scale factor without needing the full aero map.
    """
    amap = setup.aero.aero_map
    hf, hr = setup.static_front_ride_height, setup.static_rear_ride_height
    return {
        "mass": setup.mass,
        "wheelbase": setup.wheelbase,
        "weight_distribution_front": setup.weight_distribution_front,
        "cg_height": setup.cg_height,
        "static_front_ride_height": hf,
        "static_rear_ride_height": hr,
        "steering_ratio": setup.steering_ratio,
        "front_suspension": asdict(setup.front_suspension),
        "rear_suspension": asdict(setup.rear_suspension),
        "front_tire": asdict(setup.front_tire.params),
        "rear_tire": asdict(setup.rear_tire.params),
        "aero": {
            "frontal_area": setup.aero.frontal_area,
            "air_density": setup.aero.air_density,
            "cl_scale": 1.0,
            "cd_scale": 1.0,
            "reference_cl": amap.cl(hf, hr),
            "reference_cd": amap.cd(hf, hr),
            "reference_speed": reference_speed,
        },
    }


def dict_to_setup(setup_dict: dict[str, Any], base: VehicleSetup | None = None) -> VehicleSetup:
    """Build a :class:`VehicleSetup` from a setup dictionary.

    Tire model *types* (linear / Pacejka / lookup) and the aero map *shape*
    are taken from ``base`` (default: ``VehicleSetup()``); the dictionary
    supplies every number. ``aero.cl_scale`` / ``cd_scale`` multiply the base
    map's coefficient tables, so ``base`` should be the setup the dictionary
    was originally created from.
    """
    base = base if base is not None else VehicleSetup()
    d = setup_dict
    aero_d = d["aero"]
    amap = base.aero.aero_map
    new_map = AeroMap(
        front_ride_heights=amap.front_ride_heights,
        rear_ride_heights=amap.rear_ride_heights,
        cl_table=amap.cl_table * aero_d.get("cl_scale", 1.0),
        cd_table=amap.cd_table * aero_d.get("cd_scale", 1.0),
        front_balance_table=amap.front_balance_table,
    )

    front_tire = copy.copy(base.front_tire)
    front_tire.params = TireParameters(**d["front_tire"])
    rear_tire = copy.copy(base.rear_tire)
    rear_tire.params = TireParameters(**d["rear_tire"])

    return VehicleSetup(
        mass=d["mass"],
        wheelbase=d["wheelbase"],
        weight_distribution_front=d["weight_distribution_front"],
        cg_height=d["cg_height"],
        front_suspension=AxleSuspension(**d["front_suspension"]),
        rear_suspension=AxleSuspension(**d["rear_suspension"]),
        front_tire=front_tire,
        rear_tire=rear_tire,
        aero=AeroModel(aero_map=new_map, frontal_area=aero_d["frontal_area"], air_density=aero_d["air_density"]),
        static_front_ride_height=d["static_front_ride_height"],
        static_rear_ride_height=d["static_rear_ride_height"],
        steering_ratio=d["steering_ratio"],
        gravity=base.gravity,
    )


# =============================================================================
# Intent parsers
# =============================================================================
class IntentParser(Protocol):
    """Anything that turns request text into structured changes."""

    name: str

    def parse(self, request: str, setup_dict: dict[str, Any]) -> tuple[list[SetupChange], list[str]]:
        """Return ``(changes, unparsed_fragments_with_reasons)``."""
        ...


_UNIT_PATTERN = (
    r"%|percent"
    r"|k?n\s*[·*.\-]?\s*m\s*/\s*(?:deg|rad|°)"
    r"|n/mm|kn/m|n/m(?![a-z])|n/rad|n/deg"
    r"|lbf?s?\s*/\s*in"
    r"|kgf|kg|lbf|lbs?|pounds?|kn"
    r"|mm|cm|inch(?:es)?|in(?=\s*$)"
    r"|m²|m\^?2|m(?![a-z/])"
    r"|n(?![a-z])|deg|°"
)
_NUMBER_RE = re.compile(r"(?P<num>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>" + _UNIT_PATTERN + r")?")
_SPEED_RE = re.compile(r"\bat\s+(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>km/h|kph|mph|m/s)")
_MULTIPLIER_RE = re.compile(r"\b(?P<word>double|twice|triple|halve|half)\b")
_MULTIPLIERS = {"double": 2.0, "twice": 2.0, "triple": 3.0, "halve": 0.5, "half": 0.5}
_POSITIVE_RE = re.compile(
    r"\b(?:stiffen\w*|stiffer|increase\w*|raise\w*|add\w*|more|up|boost\w*|bump\w*|higher|forward"
    r"|gain|strengthen\w*|extra|stronger|bigger|harder|heavier|larger|wider|longer)\b"
)
_NEGATIVE_RE = re.compile(
    r"\b(?:soften\w*|softer|decrease\w*|reduce\w*|lower\w*|remove\w*|less|cut|drop\w*|down|weaken\w*"
    r"|subtract\w*|smaller|lighter|weaker|narrower|shorter|rearward|backward|back|take\s+off)\b"
)
_SET_BEFORE_NUMBER_RE = re.compile(r"\bto\s*$")
_CLAUSE_SPLIT_RE = re.compile(r"\s*(?:,|;|\.(?=\s|$)|\band\b|\bthen\b|\balso\b|\bplus\b|&)\s*")
_BOTH_AXLES_RE = re.compile(r"\b(?:front\s*(?:and|&)\s*rear|rear\s*(?:and|&)\s*front|all\s*four|both\s*axles|both\s*ends)\b")


class RuleBasedIntentParser:
    """Offline regex parser for short, imperative setup requests.

    Understands clauses like:

    * "stiffen front springs by 20%"        -> spring_rate, front, scale +20 %
    * "add 50kg of downforce at 150 km/h"   -> downforce, add 50 kgf at 41.7 m/s
    * "lower the rear ride height 5 mm"     -> ride_height, rear, add -5 mm
    * "set rear spring rate to 80 N/mm"     -> spring_rate, rear, set 80 N/mm
    * "double the front anti-roll bar"      -> anti_roll_bar_rate, front, x2
    * "move weight forward 2%"              -> weight_distribution_front, +2 points

    Requests are split into clauses on commas, "and", "then", ...; each clause
    must name exactly one parameter. Anything it cannot understand is
    reported back rather than guessed. Use :class:`LLMIntentParser` for
    looser, conversational phrasing.
    """

    name = "rule-based parser"

    def parse(self, request: str, setup_dict: dict[str, Any]) -> tuple[list[SetupChange], list[str]]:
        changes: list[SetupChange] = []
        unparsed: list[str] = []
        for clause in self._split(self._normalise(request)):
            try:
                changes.append(self._parse_clause(clause))
            except SetupChangeError as exc:
                unparsed.append(f'"{clause}" ({exc})')
        return changes, unparsed

    @staticmethod
    def _normalise(text: str) -> str:
        t = text.lower().strip()
        t = re.sub(r"down[\s\-]+force", "downforce", t)
        t = _BOTH_AXLES_RE.sub("both", t)
        return t

    @staticmethod
    def _split(text: str) -> list[str]:
        return [c.strip() for c in _CLAUSE_SPLIT_RE.split(text) if c and c.strip()]

    @staticmethod
    def _match_parameter(clause: str) -> ParamSpec | None:
        for spec in PARAMETERS:
            if re.search(spec.aliases, clause):
                return spec
        return None

    @staticmethod
    def _match_axle(clause: str) -> str | None:
        front = re.search(r"\bfront\b", clause) is not None
        rear = re.search(r"\brear\b", clause) is not None
        if re.search(r"\bboth\b", clause) or (front and rear):
            return "both"
        if front:
            return "front"
        if rear:
            return "rear"
        return None

    @staticmethod
    def _direction(clause: str) -> int | None:
        pos = _POSITIVE_RE.search(clause)
        neg = _NEGATIVE_RE.search(clause)
        if pos and neg:
            return 1 if pos.start() < neg.start() else -1
        if pos:
            return 1
        if neg:
            return -1
        return None

    def _parse_clause(self, clause: str) -> SetupChange:
        spec = self._match_parameter(clause)
        if spec is None:
            raise SetupChangeError("no recognised setup parameter")

        reference_speed = None
        body = clause
        speed_match = _SPEED_RE.search(body)
        if speed_match:
            reference_speed = float(speed_match["num"]) * _SPEED_FACTORS[speed_match["unit"]]
            body = body[: speed_match.start()] + body[speed_match.end() :]

        axle = self._match_axle(body) if (spec.per_axle or spec.key in _AERO_FORCE_PARAMS) else None
        common = dict(parameter=spec.key, axle=axle, reference_speed=reference_speed, source_text=clause)

        mult = _MULTIPLIER_RE.search(body)
        if mult:
            return SetupChange(operation="multiply", value=_MULTIPLIERS[mult["word"]], **common)

        num = _NUMBER_RE.search(body)
        if num is None:
            raise SetupChangeError("no amount found")
        raw = num["num"]
        value = float(raw)
        unit = _normalise_unit(num["unit"])

        if _SET_BEFORE_NUMBER_RE.search(body[: num.start()]):
            if unit == "%" and spec.dimension != "fraction":
                return SetupChange(operation="multiply", value=value / 100.0, **common)
            return SetupChange(operation="set", value=value, unit=unit, **common)

        if raw[0] not in "+-":
            sign = self._direction(body)
            if sign is None:
                raise SetupChangeError("could not tell whether to increase or decrease")
            value = sign * abs(value)
        operation = "scale_percent" if unit == "%" else "add"
        return SetupChange(operation=operation, value=value, unit="" if unit == "%" else unit, **common)


# ---------------------------------------------------------------------------
# LLM parser
# ---------------------------------------------------------------------------
LLMCallable = Callable[[str, str], str]
"""``complete(system_prompt, user_prompt) -> response_text``."""

_SYSTEM_PROMPT_TEMPLATE = """\
You are a race-car setup assistant. Convert the engineer's request into a list
of structured setup changes. You do NOT compute new values; you only describe
the requested changes precisely.

Adjustable parameters (key: description; scope; SI unit; accepted units):
<<CATALOG>>

Rules:
- "operation" is one of:
    "scale_percent": change by a signed percentage, e.g. "20% stiffer" -> 20, "soften 10%" -> -10.
    "add":           add a signed amount in "unit", e.g. "lower ride height 5 mm" -> -5, "mm".
    "set":           set an absolute value in "unit", e.g. "set rear springs to 80 N/mm" -> 80, "N/mm".
    "multiply":      multiply by a factor, e.g. "double" -> 2, "halve" -> 0.5.
- "value" is a signed number. Decrease/soften/lower/remove means negative.
- Always give "unit" for "add" and "set" using one of the accepted units ("" for dimensionless).
- "axle" is "front", "rear" or "both" for per-axle parameters (use "both" if not stated), null otherwise.
- For weight distribution, percentages are percentage points of front weight.
- "kg of downforce" means kilogram-force: use unit "kg" with parameter "downforce".
- If a speed is given for downforce/drag (e.g. "at 150 km/h"), put it in
  "reference_speed_mps" converted to m/s; otherwise null.
- Put any part of the request you cannot map to a parameter in "unparsed".
- Respond with JSON only, no prose, in exactly this shape:

{"changes": [{"parameter": "spring_rate", "axle": "front", "operation": "scale_percent",
              "value": 20, "unit": "", "reference_speed_mps": null,
              "source_text": "stiffen front springs by 20%"}],
 "unparsed": []}
"""


def build_system_prompt() -> str:
    """System prompt listing every adjustable parameter and the JSON contract."""
    lines = []
    for spec in PARAMETERS:
        scope = "per-axle" if spec.per_axle else "whole vehicle"
        units = sorted(u for u in _UNIT_FACTORS[spec.dimension] if u)
        lines.append(
            f"- {spec.key}: {spec.label}; {scope}; {spec.si_unit}; "
            f"units: {', '.join(units) if units else 'none (dimensionless)'}"
        )
    return _SYSTEM_PROMPT_TEMPLATE.replace("<<CATALOG>>", "\n".join(lines))


def build_user_prompt(request: str, setup_dict: dict[str, Any]) -> str:
    """User prompt: current setup (for context) plus the request."""
    return f"Current setup (SI units):\n{json.dumps(setup_dict, indent=2)}\n\nRequest: {request}"


#: JSON schema for LLM function-calling / structured-output APIs.
SETUP_CHANGES_TOOL_SCHEMA: dict[str, Any] = {
    "name": "apply_setup_changes",
    "description": "Structured list of vehicle setup changes requested by the user.",
    "parameters": {
        "type": "object",
        "properties": {
            "changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "parameter": {"type": "string", "enum": [p.key for p in PARAMETERS]},
                        "axle": {"type": ["string", "null"], "enum": [*AXLES, None]},
                        "operation": {"type": "string", "enum": list(OPERATIONS)},
                        "value": {"type": "number"},
                        "unit": {"type": "string"},
                        "reference_speed_mps": {"type": ["number", "null"]},
                        "source_text": {"type": "string"},
                    },
                    "required": ["parameter", "operation", "value"],
                },
            },
            "unparsed": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["changes"],
    },
}


def parse_llm_response(raw: str) -> tuple[list[SetupChange], list[str]]:
    """Validate an LLM JSON reply and convert it to :class:`SetupChange` objects.

    Tolerates Markdown code fences and surrounding prose. Individual invalid
    items are reported in the returned ``unparsed`` list rather than failing
    the whole reply.

    Raises
    ------
    LLMResponseError
        If no JSON object with a ``"changes"`` list can be found.
    """
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise LLMResponseError("no JSON object found in LLM response")
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("changes"), list):
        raise LLMResponseError('response must be an object with a "changes" list')

    changes: list[SetupChange] = []
    unparsed = [str(u) for u in data.get("unparsed") or []]
    for item in data["changes"]:
        try:
            changes.append(_change_from_json(item))
        except (SetupChangeError, TypeError, ValueError) as exc:
            unparsed.append(f"LLM item {item!r} rejected ({exc})")
    return changes, unparsed


def _change_from_json(item: Any) -> SetupChange:
    if not isinstance(item, dict):
        raise SetupChangeError("item is not an object")
    parameter = item.get("parameter")
    if parameter not in PARAMS_BY_KEY:
        raise SetupChangeError(f"unknown parameter {parameter!r}")
    operation = item.get("operation")
    if operation not in OPERATIONS:
        raise SetupChangeError(f"unknown operation {operation!r}")
    value = item.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SetupChangeError(f"value must be a finite number, got {value!r}")
    axle = item.get("axle")
    axle = axle.lower() if isinstance(axle, str) else None
    if axle is not None and axle not in AXLES:
        raise SetupChangeError(f"unknown axle {axle!r}")
    speed = item.get("reference_speed_mps")
    if speed is not None and (not isinstance(speed, (int, float)) or speed <= 0):
        raise SetupChangeError("reference_speed_mps must be a positive number")
    return SetupChange(
        parameter=parameter,
        operation=operation,
        value=float(value),
        unit=_normalise_unit(item.get("unit")),
        axle=axle,
        reference_speed=float(speed) if speed is not None else None,
        source_text=str(item.get("source_text", "")),
    )


class LLMIntentParser:
    """Intent parser that delegates language understanding to an LLM.

    Parameters
    ----------
    complete : callable
        ``complete(system_prompt, user_prompt) -> str``. Any provider works;
        ask it for JSON-only output if the API supports it.
    """

    name = "LLM parser"

    def __init__(self, complete: LLMCallable) -> None:
        self.complete = complete

    def parse(self, request: str, setup_dict: dict[str, Any]) -> tuple[list[SetupChange], list[str]]:
        raw = self.complete(build_system_prompt(), build_user_prompt(request, setup_dict))
        return parse_llm_response(raw)


# =============================================================================
# Applying changes
# =============================================================================
def _get(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise SetupChangeError(f"setup dictionary is missing '{path}'")
        cur = cur[key]
    return cur


def _set(d: dict[str, Any], path: str, value: float) -> None:
    *parents, last = path.split(".")
    cur = d
    for key in parents:
        cur = cur[key]
    cur[last] = value


def _to_si(value: float, unit: str, spec: ParamSpec) -> float:
    """Convert ``value`` in ``unit`` to the parameter's SI unit."""
    u = _normalise_unit(unit)
    if spec.dimension == "fraction" and u == "":
        return value if abs(value) <= 1.0 else value / 100.0  # bare 55 means 55 %
    if u == "":
        u = spec.default_unit
    factors = _UNIT_FACTORS[spec.dimension]
    if u not in factors:
        accepted = ", ".join(sorted(k for k in factors if k)) or "no unit"
        raise SetupChangeError(f"unit '{unit}' is not valid for {spec.label} (use {accepted})")
    return value * factors[u]


def _new_value(old: float, spec: ParamSpec, change: SetupChange) -> float:
    op, v = change.operation, change.value
    if op == "multiply":
        return old * v
    if op == "scale_percent":
        return old + v / 100.0 if spec.dimension == "fraction" else old * (1.0 + v / 100.0)
    amount = _to_si(v, change.unit, spec)
    return old + amount if op == "add" else amount


def _clamp(value: float, spec: ParamSpec, label: str) -> tuple[float, list[str]]:
    lo, hi = spec.bounds
    if value < lo or value > hi:
        clamped = min(max(value, lo), hi)
        return clamped, [
            f"{label} limited to {_fmt(clamped * spec.display_factor)} {spec.display_unit}".rstrip()
            + f" (requested {_fmt(value * spec.display_factor)}; allowed range "
            f"{_fmt(lo * spec.display_factor)}-{_fmt(hi * spec.display_factor)})"
        ]
    return value, []


def _fmt(x: float) -> str:
    if abs(x) >= 100:
        return f"{x:,.0f}"
    if abs(x) >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def _describe(spec: ParamSpec, label: str, old: float, new: float) -> str:
    unit = f" {spec.display_unit}" if spec.display_unit and spec.display_unit != "%" else spec.display_unit
    text = f"{label}: {_fmt(old * spec.display_factor)}{unit} -> {_fmt(new * spec.display_factor)}{unit}"
    if spec.dimension == "fraction":
        text += f" ({(new - old) * 100:+.1f} percentage points)"
    elif old != 0:
        text += f" ({(new / old - 1) * 100:+.1f}%)"
    return text


def _apply_regular(d: dict[str, Any], spec: ParamSpec, change: SetupChange) -> list[AppliedChange]:
    if spec.per_axle:
        axles = ["front", "rear"] if change.axle in (None, "both") else [change.axle]
    else:
        axles = [None]
    applied = []
    for axle in axles:
        path = spec.path.format(axle=axle) if axle else spec.path
        label = f"{axle.capitalize()} {spec.label}" if axle else spec.label[0].upper() + spec.label[1:]
        old = float(_get(d, path))
        new, notes = _clamp(_new_value(old, spec, change), spec, label)
        if spec.per_axle and change.axle is None:
            notes.append("no axle specified, so the change was applied to both axles")
        _set(d, path, new)
        applied.append(AppliedChange(label, path, old, new, _describe(spec, label, old, new), tuple(notes)))
    return applied


def _apply_aero_force(d: dict[str, Any], spec: ParamSpec, change: SetupChange) -> list[AppliedChange]:
    """Downforce/drag: convert the requested force change into an aero-map scale.

    .. math::

        F = \\tfrac12 \\rho V_{ref}^2 A \\, C_{ref} \\, s
        \\quad\\Rightarrow\\quad
        s_{new} = \\frac{F_{new}}{\\tfrac12 \\rho V_{ref}^2 A \\, C_{ref}}
    """
    scale_key, ref_key, coeff_name = _AERO_FORCE_PARAMS[spec.key]
    aero = _get(d, "aero")
    notes: list[str] = []

    v_ref = change.reference_speed
    if v_ref is None:
        v_ref = float(aero.get("reference_speed", DEFAULT_REFERENCE_SPEED))
        if change.operation in ("add", "set"):
            notes.append(f"no speed given; amount interpreted at {v_ref:.1f} m/s ({v_ref * 3.6:.0f} km/h)")
    if v_ref <= 0:
        raise SetupChangeError("reference speed must be positive")
    if change.axle in ("front", "rear"):
        notes.append(
            f"{spec.label} is changed for the whole car; use aero balance (aero.py) to shift it {change.axle}"
        )

    force_per_unit_scale = (
        dynamic_pressure(v_ref, float(_get(d, "aero.air_density")))
        * float(_get(d, "aero.frontal_area"))
        * float(_get(d, f"aero.{ref_key}"))
    )
    if force_per_unit_scale <= 0:
        raise SetupChangeError(f"reference {coeff_name} is zero, cannot scale {spec.label}")

    old_scale = float(aero.get(scale_key, 1.0))
    if change.operation == "multiply":
        new_scale = old_scale * change.value
    elif change.operation == "scale_percent":
        new_scale = old_scale * (1.0 + change.value / 100.0)
    else:
        amount = _to_si(change.value, change.unit, spec)
        old_force = force_per_unit_scale * old_scale
        new_force = old_force + amount if change.operation == "add" else amount
        new_scale = new_force / force_per_unit_scale

    label = spec.label[0].upper() + spec.label[1:]
    new_scale, clamp_notes = _clamp(new_scale, spec, f"{label} scale")
    notes.extend(clamp_notes)
    aero[scale_key] = new_scale

    old_f, new_f = force_per_unit_scale * old_scale, force_per_unit_scale * new_scale
    ratio = new_scale / old_scale if old_scale else math.inf
    description = (
        f"{label} at {v_ref:.1f} m/s ({v_ref * 3.6:.0f} km/h): {_fmt(old_f)} N -> {_fmt(new_f)} N "
        f"({new_f - old_f:+,.0f} N); aero map {coeff_name} scaled x{ratio:.3f}"
    )
    return [AppliedChange(label, f"aero.{scale_key}", old_scale, new_scale, description, tuple(notes))]


def apply_changes(
    setup_dict: dict[str, Any], changes: Sequence[SetupChange]
) -> tuple[dict[str, Any], list[AppliedChange], list[str]]:
    """Apply structured changes to a **copy** of ``setup_dict``.

    Changes are applied in order, so "stiffen springs 10% then set front
    springs to 70 N/mm" ends with the front at exactly 70 N/mm. Every result
    is clamped to the parameter's physical bounds (with a note).

    Returns
    -------
    (new_setup_dict, applied_changes, warnings)
        ``warnings`` lists changes that could not be applied at all.
    """
    new_dict = copy.deepcopy(setup_dict)
    applied: list[AppliedChange] = []
    warnings: list[str] = []
    for change in changes:
        spec = PARAMS_BY_KEY.get(change.parameter)
        if spec is None:
            warnings.append(f"Unknown parameter '{change.parameter}' skipped.")
            continue
        try:
            if spec.key in _AERO_FORCE_PARAMS:
                applied.extend(_apply_aero_force(new_dict, spec, change))
            else:
                applied.extend(_apply_regular(new_dict, spec, change))
        except SetupChangeError as exc:
            source = f' "{change.source_text}"' if change.source_text else ""
            warnings.append(f"Could not apply {spec.label} change{source}: {exc}.")
    return new_dict, applied, warnings


# =============================================================================
# Summary and top-level API
# =============================================================================
def _build_summary(
    request: str, parser_name: str, applied: list[AppliedChange], unparsed: list[str], warnings: list[str]
) -> str:
    lines = [f'Request: "{request.strip()}"', f"Interpreted by: {parser_name}"]
    if applied:
        lines.append("Changes made:")
        for change in applied:
            lines.append(f"  - {change.description}")
            lines.extend(f"      note: {n}" for n in change.notes)
    else:
        lines.append("No changes were made.")
    if unparsed:
        lines.append("Not understood (no change made):")
        lines.extend(f"  - {u}" for u in unparsed)
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in warnings)
    return "\n".join(lines)


def interpret_setup_request(
    request: str,
    setup_dict: dict[str, Any],
    llm: LLMCallable | None = None,
    fallback_to_rules: bool = True,
) -> ParseResult:
    """Parse a natural-language request and apply it to a copy of ``setup_dict``.

    Parameters
    ----------
    request : str
        e.g. ``"Stiffen front springs by 20% and add 50kg of downforce"``.
    setup_dict : dict
        Current setup (see :func:`setup_to_dict`). Never modified.
    llm : callable, optional
        ``complete(system_prompt, user_prompt) -> str``. If omitted, the
        offline :class:`RuleBasedIntentParser` is used.
    fallback_to_rules : bool
        If the LLM call fails or returns invalid JSON, fall back to the
        rule-based parser (with a warning) instead of raising.
    """
    warnings: list[str] = []
    parser: IntentParser = LLMIntentParser(llm) if llm is not None else RuleBasedIntentParser()
    try:
        changes, unparsed = parser.parse(request, setup_dict)
    except Exception as exc:
        if llm is None or not fallback_to_rules:
            raise
        warnings.append(f"LLM parsing failed ({exc}); used the rule-based parser instead.")
        parser = RuleBasedIntentParser()
        changes, unparsed = parser.parse(request, setup_dict)

    new_dict, applied, apply_warnings = apply_changes(setup_dict, changes)
    warnings.extend(apply_warnings)
    summary = _build_summary(request, parser.name, applied, unparsed, warnings)
    return ParseResult(new_dict, summary, applied, unparsed, warnings, parser.name)


def parse_setup_request(
    request: str, setup_dict: dict[str, Any], llm: LLMCallable | None = None
) -> tuple[dict[str, Any], str]:
    """Convenience wrapper: return ``(modified_setup_dict, summary_text)``.

    Example
    -------
    >>> from physics import VehicleSetup
    >>> setup = setup_to_dict(VehicleSetup())
    >>> new, summary = parse_setup_request("Stiffen front springs by 20%", setup)
    >>> new["front_suspension"]["spring_rate"]
    72000.0
    """
    result = interpret_setup_request(request, setup_dict, llm=llm)
    return result.setup_dict, result.summary
