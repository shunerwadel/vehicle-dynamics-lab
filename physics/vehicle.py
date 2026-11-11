"""
Vehicle: steady-state solvers for cornering and straight-line driving.

What "steady state" means
-------------------------
We assume the car has been driving around a circle of radius :math:`R` at a
constant speed :math:`V` for long enough that nothing is changing any more:
the steering wheel is still, the body has finished rolling, and the forces
are in perfect balance. That lets us replace differential equations with
plain algebra.

The bicycle-model equations
---------------------------
Lump the two front tires into one and the two rear tires into one. With
:math:`a` = CG-to-front-axle distance, :math:`b` = CG-to-rear-axle distance and
:math:`L = a + b`:

1. **Centripetal acceleration** needed to follow the circle

   .. math:: a_y = \\frac{V^2}{R}

2. **Lateral force balance** (Newton's 2nd law sideways)

   .. math:: F_{y,f} + F_{y,r} = m a_y

3. **Yaw moment balance** (no spinning up)

   .. math:: F_{y,f}\\, a = F_{y,r}\\, b
             \\;\\Rightarrow\\;
             F_{y,f} = m a_y \\frac{b}{L},\\quad F_{y,r} = m a_y \\frac{a}{L}

4. **Steering geometry** (small angles)

   .. math:: \\delta = \\frac{L}{R} + \\alpha_f - \\alpha_r

   :math:`L/R` is the *Ackermann* angle (what you would need with perfect,
   non-slipping tires). If the front tires need more slip than the rears
   (:math:`\\alpha_f > \\alpha_r`) you need extra steering: **understeer**.

The solver adds a 4-wheel layer on top: per-tire vertical loads from static
weight, aero downforce and lateral load transfer, then finds the slip angle
at which each axle's *pair* of tires makes the required force. Because of tire
load sensitivity, the inside/outside load split directly changes the slip
angle each axle needs -- which is how suspension tuning changes the balance.

Coordinate system (ISO 8855)
----------------------------
x forward, y left, z up. A **positive radius means a left turn** (positive
:math:`a_y`, positive steer, load moves to the right-hand tires). A negative
radius means a right turn.

Units
-----
kg, m, s, N, rad (angles are reported in radians; use ``numpy.degrees`` for
display).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .aero import AeroForces, AeroModel
from .suspension import (
    GRAVITY,
    AxleSuspension,
    LateralLoadTransfer,
    lateral_load_transfer,
    longitudinal_load_transfer,
    roll_angle,
    roll_axis_height,
    static_wheel_loads,
)
from .tires import PacejkaTire, TireModel

__all__ = [
    "WHEELS",
    "VehicleSetup",
    "VehicleState",
    "StraightLineState",
    "solve_aero_ride_heights",
    "solve_steady_state_cornering",
    "solve_straight_line",
    "find_max_cornering_speed",
    "understeer_gradient",
]

#: Wheel labels used throughout: front-left, front-right, rear-left, rear-right.
WHEELS = ("FL", "FR", "RL", "RR")

#: Largest slip angle the axle solver will consider [rad] (30 deg).
_MAX_SLIP_ANGLE = math.radians(30.0)


# ----------------------------------------------------------------------------
# Inputs and outputs
# ----------------------------------------------------------------------------
@dataclass
class VehicleSetup:
    """Everything needed to describe the car. Defaults are a ~1200 kg club racer.

    Attributes
    ----------
    mass : float
        Total vehicle mass including driver :math:`m` [kg].
    wheelbase : float
        Distance between front and rear axles :math:`L` [m].
    weight_distribution_front : float
        Fraction of static weight on the front axle :math:`w_f` [-].
    cg_height : float
        Height of the centre of gravity above the ground :math:`h` [m].
    front_suspension, rear_suspension : AxleSuspension
        Spring, anti-roll bar, track and roll-centre data for each axle.
    front_tire, rear_tire : TireModel
        Tire model used on each axle (both tires on an axle are identical).
    aero : AeroModel
        Aero map, frontal area and air density.
    static_front_ride_height, static_rear_ride_height : float
        Ride heights at rest with no downforce [m].
    steering_ratio : float
        Handwheel angle / road-wheel angle [-].
    gravity : float
        Gravitational acceleration :math:`g` [m/s^2].
    """

    mass: float = 1200.0
    wheelbase: float = 2.60
    weight_distribution_front: float = 0.52
    cg_height: float = 0.45
    front_suspension: AxleSuspension = field(
        default_factory=lambda: AxleSuspension(
            spring_rate=60_000.0,
            motion_ratio=0.8,
            track_width=1.55,
            anti_roll_bar_rate=20_000.0,
            roll_center_height=0.05,
        )
    )
    rear_suspension: AxleSuspension = field(
        default_factory=lambda: AxleSuspension(
            spring_rate=50_000.0,
            motion_ratio=0.8,
            track_width=1.53,
            anti_roll_bar_rate=10_000.0,
            roll_center_height=0.08,
        )
    )
    front_tire: TireModel = field(default_factory=PacejkaTire)
    rear_tire: TireModel = field(default_factory=PacejkaTire)
    aero: AeroModel = field(default_factory=AeroModel)
    static_front_ride_height: float = 0.070
    static_rear_ride_height: float = 0.090
    steering_ratio: float = 15.0
    gravity: float = GRAVITY

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError("mass must be positive [kg].")
        if self.wheelbase <= 0:
            raise ValueError("wheelbase must be positive [m].")
        if not 0.0 < self.weight_distribution_front < 1.0:
            raise ValueError("weight_distribution_front must be strictly between 0 and 1.")
        if self.cg_height <= 0:
            raise ValueError("cg_height must be positive [m].")
        if self.static_front_ride_height < 0 or self.static_rear_ride_height < 0:
            raise ValueError("Static ride heights must be >= 0 [m].")
        if self.steering_ratio <= 0:
            raise ValueError("steering_ratio must be positive.")

    @property
    def weight(self) -> float:
        """Vehicle weight :math:`W = m g` [N]."""
        return self.mass * self.gravity

    @property
    def cg_to_front_axle(self) -> float:
        """:math:`a = L (1 - w_f)` [m]. More weight on the front -> CG closer to the front."""
        return self.wheelbase * (1.0 - self.weight_distribution_front)

    @property
    def cg_to_rear_axle(self) -> float:
        """:math:`b = L w_f` [m]."""
        return self.wheelbase * self.weight_distribution_front


@dataclass(frozen=True)
class VehicleState:
    """Steady-state cornering result.

    Attributes
    ----------
    speed : float                 :math:`V` [m/s].
    corner_radius : float         :math:`R` [m] (positive = left turn).
    lateral_accel : float         :math:`a_y = V^2/R` [m/s^2].
    yaw_rate : float              :math:`r = V/R` [rad/s].
    steer_angle : float           Road-wheel steer angle :math:`\\delta` [rad].
    handwheel_angle : float       Steering-wheel angle [rad].
    sideslip_angle : float        Body slip angle at the CG :math:`\\beta` [rad].
    roll_angle : float            Body roll :math:`\\phi` [rad] (positive = right side down).
    front_slip_angle : float      :math:`\\alpha_f` [rad].
    rear_slip_angle : float       :math:`\\alpha_r` [rad].
    front_ride_height : float     Dynamic front ride height [m].
    rear_ride_height : float      Dynamic rear ride height [m].
    aero : AeroForces             Aero forces at the dynamic ride heights.
    tire_vertical_loads : dict    :math:`F_z` per wheel [N].
    tire_lateral_forces : dict    :math:`F_y` per wheel [N].
    load_transfer : LateralLoadTransfer  Breakdown of lateral load transfer [N].
    is_feasible : bool            ``False`` if the tires cannot make the corner.
    limiting_axle : str or None   ``"front"`` (understeer limit), ``"rear"``
                                  (oversteer limit), ``"both"`` or ``None``.
    wheel_lift : bool             ``True`` if an inside tire would need negative load.
    """

    speed: float
    corner_radius: float
    lateral_accel: float
    yaw_rate: float
    steer_angle: float
    handwheel_angle: float
    sideslip_angle: float
    roll_angle: float
    front_slip_angle: float
    rear_slip_angle: float
    front_ride_height: float
    rear_ride_height: float
    aero: AeroForces
    tire_vertical_loads: dict[str, float]
    tire_lateral_forces: dict[str, float]
    load_transfer: LateralLoadTransfer
    is_feasible: bool
    limiting_axle: str | None
    wheel_lift: bool

    @property
    def lateral_accel_g(self) -> float:
        """Lateral acceleration in units of g [-]."""
        return self.lateral_accel / GRAVITY

    @property
    def understeer_angle(self) -> float:
        """Extra steer beyond Ackermann, :math:`(\\alpha_f - \\alpha_r)\\,\\operatorname{sign}(a_y)` [rad].

        Positive = understeer, negative = oversteer, zero = neutral steer, for
        both left and right turns (in a right turn both slip angles are
        negative, so the sign of :math:`a_y` is factored out).
        """
        direction = -1.0 if self.lateral_accel < 0 else 1.0
        return (self.front_slip_angle - self.rear_slip_angle) * direction

    @property
    def total_vertical_load(self) -> float:
        """Sum of tire vertical loads [N]. Should equal weight + downforce."""
        return sum(self.tire_vertical_loads.values())

    @property
    def total_lateral_force(self) -> float:
        """Sum of tire lateral forces [N]. Should equal :math:`m a_y` when feasible."""
        return sum(self.tire_lateral_forces.values())


@dataclass(frozen=True)
class StraightLineState:
    """Straight-line driving result (steady cruising or an instant of acceleration).

    Attributes
    ----------
    speed : float                  [m/s]
    front_ride_height : float      [m]
    rear_ride_height : float       [m]
    aero : AeroForces              Aero forces at the dynamic ride heights.
    tire_vertical_loads : dict     :math:`F_z` per wheel [N].
    rolling_resistance : float     :math:`C_{rr} \\sum F_z` [N].
    tractive_force_required : float  Drive force needed [N].
    power_required : float         :math:`P = F_x V` [W].
    longitudinal_accel : float     :math:`a_x` [m/s^2].
    longitudinal_load_transfer : float  Load moved front -> rear [N] (whole axle).
    """

    speed: float
    front_ride_height: float
    rear_ride_height: float
    aero: AeroForces
    tire_vertical_loads: dict[str, float]
    rolling_resistance: float
    tractive_force_required: float
    power_required: float
    longitudinal_accel: float = 0.0
    longitudinal_load_transfer: float = 0.0


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
def solve_aero_ride_heights(
    setup: VehicleSetup, speed: float, tol: float = 1e-7, max_iter: int = 200
) -> tuple[float, float, AeroForces]:
    """Find the ride heights where downforce and suspension are in balance.

    This is a coupled problem: downforce compresses the suspension, which
    lowers the car, which changes the aero coefficients, which changes the
    downforce... We solve it with **fixed-point iteration**:

    .. math::

        h_f^{(k+1)} = h_{f,0} - \\frac{F_{z,aero,f}(h_f^{(k)}, h_r^{(k)})}{K_{z,f}}
        \\qquad
        h_r^{(k+1)} = h_{r,0} - \\frac{F_{z,aero,r}(h_f^{(k)}, h_r^{(k)})}{K_{z,r}}

    and repeat until the ride heights stop changing (by less than ``tol`` m).
    Ride heights are not allowed below zero (the car would be scraping).

    Returns
    -------
    (front_ride_height, rear_ride_height, aero_forces)

    Raises
    ------
    RuntimeError
        If the iteration fails to converge in ``max_iter`` steps.
    """
    hf0 = setup.static_front_ride_height
    hr0 = setup.static_rear_ride_height
    kz_f = setup.front_suspension.heave_stiffness
    kz_r = setup.rear_suspension.heave_stiffness

    hf, hr = hf0, hr0
    for _ in range(max_iter):
        aero = setup.aero.forces(speed, hf, hr)
        hf_new = max(0.0, hf0 - aero.downforce_front / kz_f)
        hr_new = max(0.0, hr0 - aero.downforce_rear / kz_r)
        if abs(hf_new - hf) < tol and abs(hr_new - hr) < tol:
            hf, hr = hf_new, hr_new
            return hf, hr, setup.aero.forces(speed, hf, hr)
        hf, hr = hf_new, hr_new
    raise RuntimeError(f"Aero ride-height iteration did not converge at V={speed:.2f} m/s.")


def _wheel_vertical_loads(
    setup: VehicleSetup, lateral_accel: float, aero: AeroForces
) -> tuple[dict[str, float], LateralLoadTransfer]:
    """Per-wheel vertical load = static + half the axle downforce +/- lateral transfer.

    .. math::

        F_{z,FL} = \\tfrac12 m g w_f + \\tfrac12 F_{aero,f} - \\Delta F_{z,f}
        \\qquad
        F_{z,FR} = \\tfrac12 m g w_f + \\tfrac12 F_{aero,f} + \\Delta F_{z,f}

    (and similarly at the rear). For a left turn :math:`\\Delta F_z > 0`, so
    the right-hand (outside) tires gain load.
    """
    static = static_wheel_loads(setup.mass, setup.weight_distribution_front, setup.gravity)
    transfer = lateral_load_transfer(
        setup.mass,
        lateral_accel,
        setup.cg_height,
        setup.weight_distribution_front,
        setup.front_suspension,
        setup.rear_suspension,
    )
    loads = {
        "FL": static["FL"] + 0.5 * aero.downforce_front - transfer.front,
        "FR": static["FR"] + 0.5 * aero.downforce_front + transfer.front,
        "RL": static["RL"] + 0.5 * aero.downforce_rear - transfer.rear,
        "RR": static["RR"] + 0.5 * aero.downforce_rear + transfer.rear,
    }
    return loads, transfer


def _axle_force(tire: TireModel, alpha: float, fz_left: float, fz_right: float) -> float:
    """Combined lateral force of both tires on an axle at a shared slip angle [N]."""
    return tire.lateral_force(alpha, fz_left) + tire.lateral_force(alpha, fz_right)


def _axle_peak(tire: TireModel, fz_left: float, fz_right: float) -> tuple[float, float]:
    """Slip angle [rad] and force [N] at which the axle's combined force peaks.

    The two tires carry different loads, so they peak at different slip
    angles. We scan a grid of slip angles and then polish the best point.
    """
    grid = np.linspace(0.0, _MAX_SLIP_ANGLE, 601)
    forces = np.array([_axle_force(tire, a, fz_left, fz_right) for a in grid])
    i = int(np.argmax(forces))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
    if hi > lo:
        res = minimize_scalar(
            lambda a: -_axle_force(tire, a, fz_left, fz_right),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-9},
        )
        if -res.fun > forces[i]:
            return float(res.x), float(-res.fun)
    return float(grid[i]), float(forces[i])


def _solve_axle_slip_angle(
    tire: TireModel, fz_left: float, fz_right: float, fy_required: float
) -> float | None:
    """Slip angle [rad] at which an axle produces ``fy_required`` [N].

    Returns ``None`` if the axle cannot generate that much force. Only the
    stable branch (below the peak) is searched, and the *smallest* matching
    slip angle is returned.
    """
    if fy_required == 0.0:
        return 0.0
    target = abs(fy_required)
    alpha_peak, f_peak = _axle_peak(tire, fz_left, fz_right)
    if target > f_peak * (1.0 + 1e-9):
        return None
    if math.isclose(target, f_peak, rel_tol=1e-9):
        return math.copysign(alpha_peak, fy_required)

    grid = np.linspace(0.0, alpha_peak, 201)
    for lo, hi in zip(grid[:-1], grid[1:]):
        if _axle_force(tire, hi, fz_left, fz_right) >= target:
            alpha = brentq(
                lambda a: _axle_force(tire, a, fz_left, fz_right) - target, lo, hi, xtol=1e-12
            )
            return math.copysign(alpha, fy_required)
    return math.copysign(alpha_peak, fy_required)


# ----------------------------------------------------------------------------
# Solvers
# ----------------------------------------------------------------------------
def solve_steady_state_cornering(setup: VehicleSetup, speed: float, radius: float) -> VehicleState:
    """Solve steady-state cornering at a given speed and corner radius.

    Algorithm
    ---------
    1. :math:`a_y = V^2 / R` (signed: left turn positive).
    2. Iterate aero <-> ride height to get dynamic ride heights and downforce
       (see :func:`solve_aero_ride_heights`).
    3. Compute per-tire vertical loads: static + aero + lateral load transfer.
    4. Bicycle-model force split:
       :math:`F_{y,f} = m a_y b/L`, :math:`F_{y,r} = m a_y a/L`.
    5. For each axle, find the common slip angle :math:`\\alpha` where
       :math:`F_y(\\alpha, F_{z,L}) + F_y(\\alpha, F_{z,R})` equals the required force.
    6. Steer angle :math:`\\delta = L/R + \\alpha_f - \\alpha_r`,
       sideslip :math:`\\beta = b/R - \\alpha_r`,
       roll :math:`\\phi = m a_y (h - h_{ra}) / K_{\\phi,total}`.

    Simplifications (good extension projects!): small angles
    (:math:`\\cos\\delta \\approx 1`), no drive/brake forces, no camber, no
    Ackermann/toe, drag-induced pitch neglected.

    Parameters
    ----------
    setup : VehicleSetup
    speed : float   Vehicle speed :math:`V` [m/s], must be >= 0.
    radius : float  Path radius :math:`R` [m]; positive = left, negative = right.
                    ``math.inf`` gives straight-line driving.

    Returns
    -------
    VehicleState
        If the tires cannot provide the force, ``is_feasible`` is ``False``,
        ``limiting_axle`` says which axle ran out of grip, and the affected
        slip/steer angles are ``nan``.
    """
    if speed < 0:
        raise ValueError("speed must be >= 0 [m/s].")
    if radius == 0 or math.isnan(radius):
        raise ValueError("radius must be non-zero [m]. Use math.inf for a straight line.")

    curvature = 0.0 if math.isinf(radius) else 1.0 / radius  # 1/m
    ay = speed**2 * curvature

    hf, hr, aero = solve_aero_ride_heights(setup, speed)
    loads, transfer = _wheel_vertical_loads(setup, ay, aero)
    wheel_lift = any(fz < 0.0 for fz in loads.values())
    tire_loads = {k: max(0.0, v) for k, v in loads.items()}

    a = setup.cg_to_front_axle
    b = setup.cg_to_rear_axle
    length = setup.wheelbase
    fy_front_required = setup.mass * ay * b / length
    fy_rear_required = setup.mass * ay * a / length

    alpha_f = _solve_axle_slip_angle(setup.front_tire, tire_loads["FL"], tire_loads["FR"], fy_front_required)
    alpha_r = _solve_axle_slip_angle(setup.rear_tire, tire_loads["RL"], tire_loads["RR"], fy_rear_required)

    front_ok, rear_ok = alpha_f is not None, alpha_r is not None
    is_feasible = front_ok and rear_ok
    limiting_axle = {
        (True, True): None,
        (False, True): "front",
        (True, False): "rear",
        (False, False): "both",
    }[(front_ok, rear_ok)]

    alpha_f_val = alpha_f if front_ok else math.nan
    alpha_r_val = alpha_r if rear_ok else math.nan

    def _fy(tire: TireModel, alpha: float, fz: float) -> float:
        return tire.lateral_force(alpha, fz) if not math.isnan(alpha) else math.nan

    lateral_forces = {
        "FL": _fy(setup.front_tire, alpha_f_val, tire_loads["FL"]),
        "FR": _fy(setup.front_tire, alpha_f_val, tire_loads["FR"]),
        "RL": _fy(setup.rear_tire, alpha_r_val, tire_loads["RL"]),
        "RR": _fy(setup.rear_tire, alpha_r_val, tire_loads["RR"]),
    }

    steer = length * curvature + alpha_f_val - alpha_r_val
    sideslip = b * curvature - alpha_r_val

    k_roll_total = setup.front_suspension.roll_stiffness + setup.rear_suspension.roll_stiffness
    h_ra = roll_axis_height(
        setup.front_suspension.roll_center_height,
        setup.rear_suspension.roll_center_height,
        setup.weight_distribution_front,
    )
    phi = roll_angle(setup.mass, ay, setup.cg_height, h_ra, k_roll_total)

    return VehicleState(
        speed=speed,
        corner_radius=radius,
        lateral_accel=ay,
        yaw_rate=speed * curvature,
        steer_angle=steer,
        handwheel_angle=steer * setup.steering_ratio,
        sideslip_angle=sideslip,
        roll_angle=phi,
        front_slip_angle=alpha_f_val,
        rear_slip_angle=alpha_r_val,
        front_ride_height=hf,
        rear_ride_height=hr,
        aero=aero,
        tire_vertical_loads=tire_loads,
        tire_lateral_forces=lateral_forces,
        load_transfer=transfer,
        is_feasible=is_feasible,
        limiting_axle=limiting_axle,
        wheel_lift=wheel_lift,
    )


def solve_straight_line(
    setup: VehicleSetup,
    speed: float,
    rolling_resistance_coeff: float = 0.012,
    longitudinal_accel: float = 0.0,
) -> StraightLineState:
    """Solve straight-line driving at an instant of speed ``V`` and acceleration ``a_x``.

    Newton's 2nd law along the car: the drive force must overcome the
    resistances *and* accelerate the mass:

    .. math::

        F_{x,drive} = m a_x + F_{drag} + F_{rr}
        = m a_x + \\tfrac{1}{2}\\rho V^2 A C_D + C_{rr} \\sum F_z

    and the power needed is :math:`P = F_{x,drive} V`. With :math:`a_x = 0`
    this is steady cruising. Acceleration also moves load rearward by
    :math:`\\Delta F_z = m a_x h / L` (half per tire); braking (:math:`a_x < 0`)
    moves it forward.

    This is *quasi-static*: it describes one instant and does not check
    whether the engine or tires can actually deliver the force.

    Parameters
    ----------
    setup : VehicleSetup
    speed : float                     [m/s], must be >= 0.
    rolling_resistance_coeff : float  :math:`C_{rr}` [-] (~0.01-0.015 for car tires).
    longitudinal_accel : float        :math:`a_x` [m/s^2]; positive = speeding up.
    """
    if speed < 0:
        raise ValueError("speed must be >= 0 [m/s].")
    if rolling_resistance_coeff < 0:
        raise ValueError("rolling_resistance_coeff must be >= 0.")

    hf, hr, aero = solve_aero_ride_heights(setup, speed)
    loads, _ = _wheel_vertical_loads(setup, 0.0, aero)
    transfer = longitudinal_load_transfer(setup.mass, longitudinal_accel, setup.cg_height, setup.wheelbase)
    for wheel in ("FL", "FR"):
        loads[wheel] -= 0.5 * transfer
    for wheel in ("RL", "RR"):
        loads[wheel] += 0.5 * transfer

    rolling = rolling_resistance_coeff * sum(loads.values()) if speed > 0 else 0.0
    tractive = setup.mass * longitudinal_accel + aero.drag + rolling
    return StraightLineState(
        speed=speed,
        front_ride_height=hf,
        rear_ride_height=hr,
        aero=aero,
        tire_vertical_loads=loads,
        rolling_resistance=rolling,
        tractive_force_required=tractive,
        power_required=tractive * speed,
        longitudinal_accel=longitudinal_accel,
        longitudinal_load_transfer=transfer,
    )


def find_max_cornering_speed(
    setup: VehicleSetup, radius: float, v_max: float = 120.0, tol: float = 1e-3
) -> float:
    """Highest speed [m/s] the car can hold on a circle of radius ``radius``.

    Strategy: step up the speed in 1 m/s increments until the tires run out
    of grip, then **bisect** between the last good and first bad speed until
    the bracket is narrower than ``tol``.

    With strong aero, grip can grow faster than demand at high speed, so this
    returns the *first* grip limit encountered. Returns ``v_max`` if no limit
    is found below it.
    """
    if radius == 0 or math.isinf(radius):
        raise ValueError("radius must be finite and non-zero [m].")

    def feasible(v: float) -> bool:
        return solve_steady_state_cornering(setup, v, radius).is_feasible

    v_good = 0.0
    v_bad = None
    for v in np.arange(1.0, v_max + 1.0, 1.0):
        if feasible(float(v)):
            v_good = float(v)
        else:
            v_bad = float(v)
            break
    if v_bad is None:
        return float(v_max)

    while v_bad - v_good > tol:
        mid = 0.5 * (v_good + v_bad)
        if feasible(mid):
            v_good = mid
        else:
            v_bad = mid
    return v_good


def understeer_gradient(setup: VehicleSetup) -> float:
    """Linear-range understeer gradient at static loads [rad/(m/s^2)].

    .. math::

        K_{us} = \\frac{m\\, w_f}{C_{\\alpha,f}} - \\frac{m\\,(1-w_f)}{C_{\\alpha,r}}

    where :math:`C_{\\alpha,f}` and :math:`C_{\\alpha,r}` are the total cornering
    stiffness of each axle (both tires) at static load. Then, for small
    :math:`a_y`, :math:`\\delta \\approx L/R + K_{us}\\, a_y`.

    * :math:`K_{us} > 0` -- understeer (stable, most road cars)
    * :math:`K_{us} = 0` -- neutral steer
    * :math:`K_{us} < 0` -- oversteer (unstable above a critical speed)
    """
    static = static_wheel_loads(setup.mass, setup.weight_distribution_front, setup.gravity)
    c_front = 2.0 * setup.front_tire.cornering_stiffness(static["FL"])
    c_rear = 2.0 * setup.rear_tire.cornering_stiffness(static["RL"])
    wf = setup.weight_distribution_front
    return setup.mass * wf / c_front - setup.mass * (1.0 - wf) / c_rear
