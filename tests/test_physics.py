"""
Unit tests for the physics engine.

Each test checks a physical law or an edge case. If a student edits one of the
"TODO (Student Exercise)" blocks and breaks the physics, these tests will say
which law is no longer satisfied.

Run from the repository root with:  ``pytest -v``
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from physics import (
    GRAVITY,
    AeroMap,
    AeroModel,
    AxleSuspension,
    LinearTire,
    LookupTire,
    PacejkaTire,
    TireParameters,
    VehicleSetup,
    default_aero_map,
    dynamic_pressure,
    find_max_cornering_speed,
    lateral_load_transfer,
    longitudinal_load_transfer,
    roll_moment_distribution,
    solve_aero_ride_heights,
    solve_steady_state_cornering,
    solve_straight_line,
    static_wheel_loads,
    understeer_gradient,
)
from physics.suspension import (
    axle_heave_stiffness,
    axle_roll_stiffness,
    ride_height_change,
    ride_rate,
    roll_angle,
    roll_axis_height,
    wheel_rate,
)

ALL_TIRES = [LinearTire, PacejkaTire, LookupTire]


def make_tire(cls):
    return cls() if cls is not LookupTire else LookupTire()


# =============================================================================
# Tires
# =============================================================================
class TestTires:
    @pytest.mark.parametrize("cls", ALL_TIRES)
    def test_zero_slip_gives_zero_force(self, cls):
        assert make_tire(cls).lateral_force(0.0, 4000.0) == 0.0

    @pytest.mark.parametrize("cls", ALL_TIRES)
    def test_zero_or_negative_load_gives_zero_force(self, cls):
        tire = make_tire(cls)
        assert tire.lateral_force(0.05, 0.0) == 0.0
        assert tire.lateral_force(0.05, -100.0) == 0.0

    @pytest.mark.parametrize("cls", ALL_TIRES)
    @pytest.mark.parametrize("alpha", [0.005, 0.03, 0.1, 0.3])
    def test_force_is_odd_function_of_slip(self, cls, alpha):
        tire = make_tire(cls)
        assert tire.lateral_force(-alpha, 4000.0) == pytest.approx(-tire.lateral_force(alpha, 4000.0))

    def test_linear_tire_obeys_fy_equals_c_alpha_times_alpha(self):
        tire = LinearTire(TireParameters(cornering_stiffness=60_000.0, nominal_load=4000.0))
        alpha = math.radians(1.0)
        assert tire.lateral_force(alpha, 4000.0) == pytest.approx(60_000.0 * alpha)

    def test_linear_tire_saturates_at_mu_fz(self):
        params = TireParameters(peak_friction=1.2, nominal_load=4000.0, load_sensitivity=0.0)
        tire = LinearTire(params)
        assert tire.lateral_force(math.radians(25.0), 4000.0) == pytest.approx(1.2 * 4000.0)

    def test_pacejka_initial_slope_matches_cornering_stiffness(self):
        tire = PacejkaTire()
        fz = 4000.0
        alpha = 1e-6
        slope = tire.lateral_force(alpha, fz) / alpha
        assert slope == pytest.approx(tire.cornering_stiffness(fz), rel=1e-6)

    @pytest.mark.parametrize("fz", [1500.0, 4000.0, 7000.0])
    def test_pacejka_peak_equals_mu_fz(self, fz):
        tire = PacejkaTire()
        peak = tire.lateral_force(tire.peak_slip_angle(fz), fz)
        assert peak == pytest.approx(tire.max_lateral_force(fz), rel=1e-6)

    def test_load_sensitivity_reduces_friction_coefficient(self):
        tire = LinearTire(TireParameters(load_sensitivity=0.2))
        assert tire.friction_coefficient(6000.0) < tire.friction_coefficient(4000.0)
        assert tire.friction_coefficient(4000.0) == pytest.approx(tire.params.peak_friction)
        # Doubling the load gives less than double the peak force.
        assert tire.max_lateral_force(8000.0) < 2.0 * tire.max_lateral_force(4000.0)

    @pytest.mark.parametrize("cls", ALL_TIRES)
    def test_inverse_model_round_trip(self, cls):
        tire = make_tire(cls)
        fz = 3500.0
        for fraction in (0.1, 0.5, 0.9):
            fy = fraction * tire.lateral_force(tire.peak_slip_angle(fz), fz)
            alpha = tire.slip_angle_for_force(fy, fz)
            assert tire.lateral_force(alpha, fz) == pytest.approx(fy, rel=1e-6)
            assert tire.slip_angle_for_force(-fy, fz) == pytest.approx(-alpha)

    @pytest.mark.parametrize("cls", ALL_TIRES)
    def test_inverse_model_rejects_impossible_force(self, cls):
        tire = make_tire(cls)
        with pytest.raises(ValueError):
            tire.slip_angle_for_force(10.0 * tire.max_lateral_force(4000.0), 4000.0)

    def test_invalid_parameters_rejected(self):
        with pytest.raises(ValueError):
            TireParameters(cornering_stiffness=-1.0)
        with pytest.raises(ValueError):
            LookupTire(slip_angles_deg=np.array([0.0, 2.0, 1.0]), normalised_force=np.array([0.0, 0.5, 1.0]))


# =============================================================================
# Aero
# =============================================================================
class TestAero:
    def test_dynamic_pressure(self):
        assert dynamic_pressure(10.0, 1.2) == pytest.approx(0.5 * 1.2 * 100.0)

    def test_zero_speed_gives_zero_aero_forces(self):
        forces = AeroModel().forces(0.0, 0.06, 0.08)
        assert forces.downforce_total == 0.0
        assert forces.drag == 0.0

    def test_downforce_formula(self):
        model = AeroModel()
        v, hf, hr = 40.0, 0.06, 0.08
        cl = model.aero_map.cl(hf, hr)
        expected = 0.5 * model.air_density * v**2 * model.frontal_area * cl
        assert model.downforce(v, hf, hr) == pytest.approx(expected)

    def test_forces_scale_with_speed_squared(self):
        model = AeroModel()
        f1 = model.forces(20.0, 0.06, 0.08)
        f2 = model.forces(40.0, 0.06, 0.08)
        assert f2.downforce_total == pytest.approx(4.0 * f1.downforce_total)
        assert f2.drag == pytest.approx(4.0 * f1.drag)

    def test_map_is_exact_at_grid_points(self):
        m = default_aero_map()
        for i, hf in enumerate(m.front_ride_heights):
            for j, hr in enumerate(m.rear_ride_heights):
                assert m.cl(hf, hr) == pytest.approx(m.cl_table[i, j])
                assert m.cd(hf, hr) == pytest.approx(m.cd_table[i, j])

    def test_bilinear_interpolation_midpoint(self):
        m = AeroMap(
            front_ride_heights=[0.0, 1.0],
            rear_ride_heights=[0.0, 1.0],
            cl_table=[[1.0, 2.0], [3.0, 4.0]],
            cd_table=[[0.3, 0.3], [0.3, 0.3]],
            front_balance_table=[[0.5, 0.5], [0.5, 0.5]],
        )
        assert m.cl(0.5, 0.5) == pytest.approx(2.5)

    def test_out_of_range_ride_heights_are_clamped(self):
        m = default_aero_map()
        lo_f, lo_r = m.front_ride_heights[0], m.rear_ride_heights[0]
        assert m.cl(-1.0, -1.0) == pytest.approx(m.cl(lo_f, lo_r))
        assert m.cl(5.0, 5.0) == pytest.approx(m.cl_table[-1, -1])

    def test_lower_ride_height_gives_more_downforce(self):
        m = default_aero_map()
        assert m.cl(0.04, 0.07) > m.cl(0.08, 0.10)

    def test_axle_downforce_sums_to_total(self):
        f = AeroModel().forces(50.0, 0.05, 0.09)
        assert f.downforce_front + f.downforce_rear == pytest.approx(f.downforce_total)
        assert f.downforce_front / f.downforce_total == pytest.approx(f.front_balance)

    def test_bad_table_shape_rejected(self):
        with pytest.raises(ValueError):
            AeroMap([0.0, 1.0], [0.0, 1.0], [[1.0, 2.0]], [[0.3, 0.3]], [[0.5, 0.5]])


# =============================================================================
# Suspension
# =============================================================================
class TestSuspension:
    def test_wheel_rate_uses_motion_ratio_squared(self):
        assert wheel_rate(100_000.0, 0.5) == pytest.approx(25_000.0)

    def test_ride_rate_is_series_combination(self):
        assert ride_rate(50_000.0, 50_000.0) == pytest.approx(25_000.0)
        assert ride_rate(30_000.0, 250_000.0) < 30_000.0

    def test_heave_stiffness_is_two_corners_in_parallel(self):
        assert axle_heave_stiffness(30_000.0) == pytest.approx(60_000.0)

    def test_roll_stiffness_formula(self):
        # K_phi = k_w t^2 / 2 + K_arb
        assert axle_roll_stiffness(40_000.0, 1.5, 10_000.0) == pytest.approx(
            0.5 * 40_000.0 * 1.5**2 + 10_000.0
        )

    def test_anti_roll_bar_does_not_change_heave_stiffness(self):
        soft = AxleSuspension(anti_roll_bar_rate=0.0)
        stiff = replace(soft, anti_roll_bar_rate=50_000.0)
        assert soft.heave_stiffness == pytest.approx(stiff.heave_stiffness)
        assert stiff.roll_stiffness > soft.roll_stiffness

    def test_roll_moment_distribution(self):
        assert roll_moment_distribution(30_000.0, 10_000.0) == pytest.approx(0.75)
        assert roll_moment_distribution(1.0, 1.0) == pytest.approx(0.5)

    def test_static_loads_sum_to_weight(self):
        loads = static_wheel_loads(1000.0, 0.6)
        assert sum(loads.values()) == pytest.approx(1000.0 * GRAVITY)
        assert loads["FL"] + loads["FR"] == pytest.approx(0.6 * 1000.0 * GRAVITY)
        assert loads["FL"] == loads["FR"]

    def test_zero_lateral_accel_gives_zero_transfer(self):
        llt = lateral_load_transfer(1200.0, 0.0, 0.45, 0.5, AxleSuspension(), AxleSuspension())
        assert llt.front == 0.0 and llt.rear == 0.0

    def test_lateral_load_transfer_moment_balance(self):
        """Total transferred moment must equal m * ay * h, whatever the tuning."""
        m, ay, h, wf = 1200.0, 9.0, 0.45, 0.55
        front = AxleSuspension(track_width=1.6, anti_roll_bar_rate=30_000.0, roll_center_height=0.04)
        rear = AxleSuspension(track_width=1.5, anti_roll_bar_rate=5_000.0, roll_center_height=0.10)
        llt = lateral_load_transfer(m, ay, h, wf, front, rear)
        moment = llt.front * front.track_width + llt.rear * rear.track_width
        assert moment == pytest.approx(m * ay * h)
        assert llt.front == pytest.approx(llt.front_geometric + llt.front_elastic)

    def test_stiffer_front_bar_moves_transfer_forward(self):
        base = AxleSuspension()
        stiff_front = replace(base, anti_roll_bar_rate=60_000.0)
        llt_base = lateral_load_transfer(1200.0, 9.0, 0.45, 0.5, base, base)
        llt_stiff = lateral_load_transfer(1200.0, 9.0, 0.45, 0.5, stiff_front, base)
        assert llt_stiff.front > llt_base.front
        assert llt_stiff.rear < llt_base.rear
        assert llt_base.lltd_front == pytest.approx(0.5)

    def test_roll_axis_height_and_roll_angle(self):
        assert roll_axis_height(0.0, 0.1, 0.5) == pytest.approx(0.05)
        phi = roll_angle(1000.0, 10.0, 0.5, 0.1, 100_000.0)
        assert phi == pytest.approx(1000.0 * 10.0 * 0.4 / 100_000.0)

    def test_longitudinal_load_transfer(self):
        assert longitudinal_load_transfer(1000.0, 5.0, 0.5, 2.5) == pytest.approx(1000.0)
        assert longitudinal_load_transfer(1000.0, -5.0, 0.5, 2.5) == pytest.approx(-1000.0)

    def test_ride_height_change_hookes_law(self):
        assert ride_height_change(1000.0, 100_000.0) == pytest.approx(0.01)

    def test_invalid_inputs_rejected(self):
        with pytest.raises(ValueError):
            wheel_rate(50_000.0, 0.0)
        with pytest.raises(ValueError):
            static_wheel_loads(1000.0, 1.2)
        with pytest.raises(ValueError):
            AxleSuspension(track_width=0.0)


# =============================================================================
# Vehicle: steady-state solvers
# =============================================================================
@pytest.fixture
def setup() -> VehicleSetup:
    return VehicleSetup()


def speed_for_lateral_accel(ay: float, radius: float) -> float:
    return math.sqrt(ay * abs(radius))


class TestVehicle:
    def test_zero_speed_corner_is_pure_geometry(self, setup):
        radius = 30.0
        state = solve_steady_state_cornering(setup, 0.0, radius)
        assert state.is_feasible
        assert state.lateral_accel == 0.0
        assert state.roll_angle == 0.0
        assert state.steer_angle == pytest.approx(setup.wheelbase / radius)
        assert state.understeer_angle == 0.0
        static = static_wheel_loads(setup.mass, setup.weight_distribution_front, setup.gravity)
        for wheel, fz in static.items():
            assert state.tire_vertical_loads[wheel] == pytest.approx(fz)
            assert state.tire_lateral_forces[wheel] == 0.0

    def test_straight_line_via_infinite_radius(self, setup):
        state = solve_steady_state_cornering(setup, 30.0, math.inf)
        assert state.steer_angle == 0.0
        assert state.total_lateral_force == 0.0
        assert state.tire_vertical_loads["FL"] == pytest.approx(state.tire_vertical_loads["FR"])

    def test_invalid_inputs_rejected(self, setup):
        with pytest.raises(ValueError):
            solve_steady_state_cornering(setup, 10.0, 0.0)
        with pytest.raises(ValueError):
            solve_steady_state_cornering(setup, -1.0, 50.0)
        with pytest.raises(ValueError):
            VehicleSetup(weight_distribution_front=1.0)

    def test_left_and_right_turns_are_mirror_images(self, setup):
        left = solve_steady_state_cornering(setup, 18.0, 40.0)
        right = solve_steady_state_cornering(setup, 18.0, -40.0)
        assert right.lateral_accel == pytest.approx(-left.lateral_accel)
        assert right.steer_angle == pytest.approx(-left.steer_angle)
        assert right.roll_angle == pytest.approx(-left.roll_angle)
        assert right.sideslip_angle == pytest.approx(-left.sideslip_angle)
        assert right.understeer_angle == pytest.approx(left.understeer_angle)
        for l_wheel, r_wheel in (("FL", "FR"), ("RL", "RR")):
            assert right.tire_vertical_loads[l_wheel] == pytest.approx(left.tire_vertical_loads[r_wheel])
            assert right.tire_lateral_forces[l_wheel] == pytest.approx(-left.tire_lateral_forces[r_wheel])

    def test_symmetric_car_is_neutral_steer(self):
        """50/50 weight, identical axles and tires -> equal slip angles front and rear."""
        axle = AxleSuspension(roll_center_height=0.05)
        symmetric = VehicleSetup(
            weight_distribution_front=0.5,
            front_suspension=axle,
            rear_suspension=axle,
            front_tire=PacejkaTire(),
            rear_tire=PacejkaTire(),
            aero=AeroModel(frontal_area=1e-9),  # effectively no aero
        )
        state = solve_steady_state_cornering(symmetric, 15.0, 40.0)
        assert state.front_slip_angle == pytest.approx(state.rear_slip_angle, abs=1e-9)
        assert state.steer_angle == pytest.approx(symmetric.wheelbase / 40.0, abs=1e-9)
        assert understeer_gradient(symmetric) == pytest.approx(0.0, abs=1e-12)

    def test_equilibrium_at_one_g(self, setup):
        radius = 50.0
        speed = speed_for_lateral_accel(GRAVITY, radius)
        state = solve_steady_state_cornering(setup, speed, radius)

        assert state.is_feasible and not state.wheel_lift
        assert state.lateral_accel_g == pytest.approx(1.0)

        # Vertical equilibrium: tires carry weight + downforce.
        assert state.total_vertical_load == pytest.approx(setup.weight + state.aero.downforce_total)
        # Lateral equilibrium: tires provide m * ay.
        assert state.total_lateral_force == pytest.approx(setup.mass * GRAVITY, rel=1e-6)
        # Yaw equilibrium: front force * a == rear force * b.
        fy_f = state.tire_lateral_forces["FL"] + state.tire_lateral_forces["FR"]
        fy_r = state.tire_lateral_forces["RL"] + state.tire_lateral_forces["RR"]
        assert fy_f * setup.cg_to_front_axle == pytest.approx(fy_r * setup.cg_to_rear_axle, rel=1e-6)
        # Roll moment balance: overturning moment m*ay*h equals load transfer moment.
        dfz_f = 0.5 * (state.tire_vertical_loads["FR"] - state.tire_vertical_loads["FL"])
        dfz_r = 0.5 * (state.tire_vertical_loads["RR"] - state.tire_vertical_loads["RL"])
        moment = dfz_f * setup.front_suspension.track_width + dfz_r * setup.rear_suspension.track_width
        assert moment == pytest.approx(setup.mass * GRAVITY * setup.cg_height, rel=1e-9)

        # Left turn: outside (right) tires loaded, body rolls right, steer positive.
        assert state.tire_vertical_loads["FR"] > state.tire_vertical_loads["FL"]
        assert state.roll_angle > 0.0
        assert state.steer_angle > 0.0
        # Kinematics: delta = L/R + alpha_f - alpha_r.
        assert state.steer_angle == pytest.approx(
            setup.wheelbase / radius + state.front_slip_angle - state.rear_slip_angle
        )
        assert state.yaw_rate == pytest.approx(speed / radius)

    def test_impossible_corner_is_flagged(self, setup):
        state = solve_steady_state_cornering(setup, 40.0, 20.0)  # ~8 g
        assert not state.is_feasible
        assert state.limiting_axle in {"front", "rear", "both"}
        assert math.isnan(state.steer_angle)

    def test_stiffer_front_bar_increases_understeer(self, setup):
        radius = 50.0
        speed = speed_for_lateral_accel(0.9 * GRAVITY, radius)
        base = solve_steady_state_cornering(setup, speed, radius)
        stiff_front = replace(
            setup, front_suspension=replace(setup.front_suspension, anti_roll_bar_rate=60_000.0)
        )
        stiff = solve_steady_state_cornering(stiff_front, speed, radius)
        assert stiff.understeer_angle > base.understeer_angle

    def test_downforce_lowers_ride_height_and_adds_load(self, setup):
        hf0, hr0, aero0 = solve_aero_ride_heights(setup, 0.0)
        hf, hr, aero = solve_aero_ride_heights(setup, 60.0)
        assert (hf0, hr0) == (setup.static_front_ride_height, setup.static_rear_ride_height)
        assert aero0.downforce_total == 0.0
        assert hf < hf0 and hr < hr0
        # Converged: the ride height drop matches Hooke's law for the final downforce.
        assert hf0 - hf == pytest.approx(aero.downforce_front / setup.front_suspension.heave_stiffness, abs=1e-6)
        assert hr0 - hr == pytest.approx(aero.downforce_rear / setup.rear_suspension.heave_stiffness, abs=1e-6)

    def test_straight_line_force_balance(self, setup):
        result = solve_straight_line(setup, 40.0, rolling_resistance_coeff=0.012)
        total_fz = sum(result.tire_vertical_loads.values())
        assert total_fz == pytest.approx(setup.weight + result.aero.downforce_total)
        assert result.rolling_resistance == pytest.approx(0.012 * total_fz)
        assert result.tractive_force_required == pytest.approx(result.aero.drag + result.rolling_resistance)
        assert result.power_required == pytest.approx(result.tractive_force_required * 40.0)

    def test_straight_line_acceleration(self, setup):
        ax = 0.5 * GRAVITY
        cruise = solve_straight_line(setup, 30.0)
        accel = solve_straight_line(setup, 30.0, longitudinal_accel=ax)
        transfer = setup.mass * ax * setup.cg_height / setup.wheelbase
        assert accel.longitudinal_load_transfer == pytest.approx(transfer)
        assert accel.tire_vertical_loads["RL"] == pytest.approx(cruise.tire_vertical_loads["RL"] + 0.5 * transfer)
        assert accel.tire_vertical_loads["FL"] == pytest.approx(cruise.tire_vertical_loads["FL"] - 0.5 * transfer)
        assert sum(accel.tire_vertical_loads.values()) == pytest.approx(sum(cruise.tire_vertical_loads.values()))
        assert accel.tractive_force_required == pytest.approx(cruise.tractive_force_required + setup.mass * ax)

    def test_straight_line_at_zero_speed(self, setup):
        result = solve_straight_line(setup, 0.0)
        assert result.tractive_force_required == 0.0
        assert result.power_required == 0.0

    def test_max_cornering_speed_is_grip_boundary(self, setup):
        radius = 40.0
        v_max = find_max_cornering_speed(setup, radius)
        assert solve_steady_state_cornering(setup, v_max, radius).is_feasible
        assert not solve_steady_state_cornering(setup, v_max + 0.05, radius).is_feasible
        # A car on ~1.2-mu tires should peak somewhere between 1.0 and 1.5 g.
        assert 1.0 < v_max**2 / radius / GRAVITY < 1.5

    @pytest.mark.parametrize("tire_cls", ALL_TIRES)
    def test_solver_works_with_every_tire_model(self, tire_cls):
        car = VehicleSetup(front_tire=make_tire(tire_cls), rear_tire=make_tire(tire_cls))
        state = solve_steady_state_cornering(car, 15.0, 40.0)
        assert state.is_feasible
        assert state.total_lateral_force == pytest.approx(car.mass * 15.0**2 / 40.0, rel=1e-6)

    def test_default_car_understeers(self, setup):
        assert understeer_gradient(setup) > 0.0
