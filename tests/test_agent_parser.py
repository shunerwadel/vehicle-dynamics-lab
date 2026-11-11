"""Tests for the natural-language setup parser in ``utils/agent_parser.py``."""

from __future__ import annotations

import copy
import json
import math

import pytest

from physics import GRAVITY, VehicleSetup, solve_aero_ride_heights
from physics.aero import dynamic_pressure
from utils.agent_parser import (
    SETUP_CHANGES_TOOL_SCHEMA,
    LLMResponseError,
    RuleBasedIntentParser,
    SetupChange,
    apply_changes,
    build_system_prompt,
    dict_to_setup,
    interpret_setup_request,
    parse_llm_response,
    parse_setup_request,
    setup_to_dict,
)


@pytest.fixture
def setup_dict() -> dict:
    return setup_to_dict(VehicleSetup())


def downforce_at_reference(d: dict, speed: float | None = None) -> float:
    aero = d["aero"]
    v = speed if speed is not None else aero["reference_speed"]
    return dynamic_pressure(v, aero["air_density"]) * aero["frontal_area"] * aero["reference_cl"] * aero["cl_scale"]


class TestExampleRequest:
    def test_springs_and_downforce(self, setup_dict):
        new, summary = parse_setup_request("Stiffen front springs by 20% and add 50kg of downforce", setup_dict)

        assert new["front_suspension"]["spring_rate"] == pytest.approx(1.2 * setup_dict["front_suspension"]["spring_rate"])
        assert new["rear_suspension"]["spring_rate"] == setup_dict["rear_suspension"]["spring_rate"]
        added = downforce_at_reference(new) - downforce_at_reference(setup_dict)
        assert added == pytest.approx(50.0 * GRAVITY)
        assert "Front spring rate" in summary and "Downforce" in summary

    def test_input_dict_is_not_mutated(self, setup_dict):
        original = copy.deepcopy(setup_dict)
        parse_setup_request("Stiffen front springs by 20% and add 50kg of downforce", setup_dict)
        assert setup_dict == original

    def test_result_converts_back_to_a_working_vehicle(self, setup_dict):
        new, _ = parse_setup_request("add 50kg of downforce", setup_dict)
        base = VehicleSetup()
        _, _, aero_base = solve_aero_ride_heights(dict_to_setup(setup_dict, base), 40.0)
        _, _, aero_new = solve_aero_ride_heights(dict_to_setup(new, base), 40.0)
        assert aero_new.downforce_total > aero_base.downforce_total


class TestRuleBasedParser:
    @pytest.mark.parametrize(
        ("request_text", "parameter", "axle", "operation", "value", "unit"),
        [
            ("stiffen front springs by 20%", "spring_rate", "front", "scale_percent", 20.0, ""),
            ("soften the rear anti-roll bar 10%", "anti_roll_bar_rate", "rear", "scale_percent", -10.0, ""),
            ("lower the front ride height by 5 mm", "ride_height", "front", "add", -5.0, "mm"),
            ("set rear spring rate to 80 N/mm", "spring_rate", "rear", "set", 80.0, "n/mm"),
            ("double the front arb", "anti_roll_bar_rate", "front", "multiply", 2.0, ""),
            ("move weight forward 2%", "weight_distribution_front", None, "scale_percent", 2.0, ""),
            ("front springs +15%", "spring_rate", "front", "scale_percent", 15.0, ""),
            ("add 20 kg of ballast", "mass", None, "add", 20.0, "kg"),
            ("raise the cg by 10mm", "cg_height", None, "add", 10.0, "mm"),
            ("more grip on the rear by 5%", "peak_friction", "rear", "scale_percent", 5.0, ""),
        ],
    )
    def test_clause_parsing(self, setup_dict, request_text, parameter, axle, operation, value, unit):
        changes, unparsed = RuleBasedIntentParser().parse(request_text, setup_dict)
        assert unparsed == []
        assert len(changes) == 1
        c = changes[0]
        assert (c.parameter, c.axle, c.operation, c.unit) == (parameter, axle, operation, unit)
        assert c.value == pytest.approx(value)

    def test_front_and_rear_is_not_split_into_two_clauses(self, setup_dict):
        changes, unparsed = RuleBasedIntentParser().parse("soften front and rear springs by 10%", setup_dict)
        assert unparsed == []
        assert len(changes) == 1 and changes[0].axle == "both"

    def test_reference_speed_extracted(self, setup_dict):
        changes, _ = RuleBasedIntentParser().parse("add 300 lb of downforce at 150 km/h", setup_dict)
        assert changes[0].reference_speed == pytest.approx(150.0 / 3.6)

    def test_unrecognised_text_is_reported_not_guessed(self, setup_dict):
        result = interpret_setup_request("make it faster", setup_dict)
        assert result.setup_dict == setup_dict
        assert result.changes == []
        assert result.unparsed and "No changes were made" in result.summary

    def test_missing_direction_is_reported(self, setup_dict):
        _, unparsed = RuleBasedIntentParser().parse("front springs 20%", setup_dict)
        assert unparsed and "increase or decrease" in unparsed[0]


class TestApplyChanges:
    def test_unspecified_axle_applies_to_both(self, setup_dict):
        new, applied, _ = apply_changes(setup_dict, [SetupChange("spring_rate", "scale_percent", 10.0)])
        assert len(applied) == 2
        for axle in ("front", "rear"):
            key = f"{axle}_suspension"
            assert new[key]["spring_rate"] == pytest.approx(1.1 * setup_dict[key]["spring_rate"])

    def test_unit_conversions(self, setup_dict):
        new, _, _ = apply_changes(
            setup_dict,
            [
                SetupChange("spring_rate", "set", 400.0, unit="lb/in", axle="front"),
                SetupChange("anti_roll_bar_rate", "set", 500.0, unit="nm/deg", axle="rear"),
                SetupChange("ride_height", "add", -1.0, unit="cm", axle="rear"),
            ],
        )
        assert new["front_suspension"]["spring_rate"] == pytest.approx(400.0 * 175.126835, rel=1e-6)
        assert new["rear_suspension"]["anti_roll_bar_rate"] == pytest.approx(500.0 * 180.0 / math.pi)
        assert new["static_rear_ride_height"] == pytest.approx(setup_dict["static_rear_ride_height"] - 0.01)

    def test_weight_distribution_set_as_percentage(self, setup_dict):
        new, _, _ = apply_changes(setup_dict, [SetupChange("weight_distribution_front", "set", 55.0, unit="%")])
        assert new["weight_distribution_front"] == pytest.approx(0.55)

    def test_values_clamped_to_physical_bounds(self, setup_dict):
        result = interpret_setup_request("add 5000 kg of ballast", setup_dict)
        assert result.setup_dict["mass"] == pytest.approx(3000.0)
        assert "limited" in result.summary

    def test_downforce_set_and_percent(self, setup_dict):
        new, _, _ = apply_changes(setup_dict, [SetupChange("downforce", "set", 3000.0, unit="n", reference_speed=50.0)])
        assert downforce_at_reference(new, 50.0) == pytest.approx(3000.0)
        new, _, _ = apply_changes(setup_dict, [SetupChange("downforce", "scale_percent", 10.0)])
        assert new["aero"]["cl_scale"] == pytest.approx(1.1)

    def test_invalid_unit_becomes_warning(self, setup_dict):
        new, applied, warnings = apply_changes(setup_dict, [SetupChange("spring_rate", "add", 5.0, unit="kg")])
        assert applied == [] and new == setup_dict
        assert warnings and "not valid" in warnings[0]

    def test_changes_apply_in_order(self, setup_dict):
        new, _, _ = apply_changes(
            setup_dict,
            [
                SetupChange("spring_rate", "scale_percent", 10.0, axle="front"),
                SetupChange("spring_rate", "set", 70.0, unit="n/mm", axle="front"),
            ],
        )
        assert new["front_suspension"]["spring_rate"] == pytest.approx(70_000.0)


class TestLLMParser:
    LLM_REPLY = """Here you go:
```json
{"changes": [
  {"parameter": "spring_rate", "axle": "rear", "operation": "scale_percent", "value": -10,
   "unit": "", "reference_speed_mps": null, "source_text": "back end too stiff"},
  {"parameter": "flux_capacitor", "operation": "add", "value": 1}
 ],
 "unparsed": ["make it look cooler"]}
```"""

    def test_llm_reply_is_validated_and_applied(self, setup_dict):
        calls = {}

        def fake_llm(system: str, user: str) -> str:
            calls["system"], calls["user"] = system, user
            return self.LLM_REPLY

        result = interpret_setup_request("the back end is too stiff, take 10% out", setup_dict, llm=fake_llm)
        assert result.parser == "LLM parser"
        assert result.setup_dict["rear_suspension"]["spring_rate"] == pytest.approx(
            0.9 * setup_dict["rear_suspension"]["spring_rate"]
        )
        assert any("flux_capacitor" in u for u in result.unparsed)
        assert "make it look cooler" in result.unparsed
        assert "spring_rate" in calls["system"] and "take 10% out" in calls["user"]

    def test_invalid_llm_reply_falls_back_to_rules(self, setup_dict):
        result = interpret_setup_request("stiffen front springs by 20%", setup_dict, llm=lambda s, u: "sorry!")
        assert result.parser == "rule-based parser"
        assert any("LLM parsing failed" in w for w in result.warnings)
        assert result.setup_dict["front_suspension"]["spring_rate"] == pytest.approx(72_000.0)

    def test_invalid_llm_reply_raises_without_fallback(self, setup_dict):
        with pytest.raises(LLMResponseError):
            interpret_setup_request("x", setup_dict, llm=lambda s, u: "{}", fallback_to_rules=False)

    def test_tool_schema_and_prompt_cover_every_parameter(self):
        keys = SETUP_CHANGES_TOOL_SCHEMA["parameters"]["properties"]["changes"]["items"]["properties"]["parameter"]["enum"]
        prompt = build_system_prompt()
        assert all(f"- {k}:" in prompt for k in keys)
        json.dumps(SETUP_CHANGES_TOOL_SCHEMA)  # must be serialisable for API calls

    def test_parse_llm_response_rejects_bad_values(self):
        raw = json.dumps({"changes": [{"parameter": "mass", "operation": "add", "value": "lots"}]})
        changes, unparsed = parse_llm_response(raw)
        assert changes == [] and "finite number" in unparsed[0]


class TestDegreesInUserFacingText:
    """Setups store angular stiffness per radian, but people only ever see degrees."""

    DEG = math.pi / 180.0

    def test_bare_arb_number_means_per_degree(self, setup_dict):
        new, summary = parse_setup_request("set rear arb to 350", setup_dict)
        assert new["rear_suspension"]["anti_roll_bar_rate"] == pytest.approx(350.0 / self.DEG)
        assert "N·m/deg" in summary

    @pytest.mark.parametrize(
        ("text", "expected_si"),
        [
            ("set front arb to 20 kN·m/rad", 20_000.0),
            ("set front arb to 0.5 kNm/deg", 500.0 / (math.pi / 180.0)),
            ("set front arb to 400 N·m/deg", 400.0 / (math.pi / 180.0)),
        ],
    )
    def test_explicit_angular_units(self, setup_dict, text, expected_si):
        new, _ = parse_setup_request(text, setup_dict)
        assert new["front_suspension"]["anti_roll_bar_rate"] == pytest.approx(expected_si)

    @pytest.mark.parametrize(
        "text", ["stiffen the front arb by 20%", "set rear arb to 99999", "increase cornering stiffness by 10%"]
    )
    def test_summaries_never_mention_radians(self, setup_dict, text):
        _, summary = parse_setup_request(text, setup_dict)
        assert "rad" not in summary.replace("radius", "")
