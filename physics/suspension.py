"""
Suspension: springs, roll stiffness, and how load moves between the tires.

Background
----------
When a car corners, brakes, or accelerates, its tires do not share the weight
equally. Inertia acts at the centre of gravity (CG), which sits *above* the
ground, so it creates a moment that lifts load off some tires and pushes it
onto others. This is called **load transfer**.

Key ideas covered in this module
--------------------------------
1. **Wheel rate** -- how stiff the spring feels *at the wheel*, after the
   suspension linkage (motion ratio) is accounted for.
2. **Ride rate** -- wheel rate in series with the tire's own vertical stiffness.
3. **Heave stiffness** :math:`K_z` -- resistance to the whole axle moving up and
   down together (used for aero ride-height changes).
4. **Roll stiffness** :math:`K_\\phi` -- resistance to the body rolling, from the
   springs plus the anti-roll bar.
5. **Roll moment distribution** -- how the roll moment is shared front/rear.
   This is the main tuning knob for understeer vs. oversteer!
6. **Lateral and longitudinal load transfer**.

Coordinate system (ISO 8855)
----------------------------
* x forward, y to the **left**, z up.
* Positive lateral acceleration :math:`a_y` = turning **left**; load moves to
  the **right** (outside) tires.
* Positive longitudinal acceleration :math:`a_x` = speeding up; load moves to
  the **rear**.
* Positive roll angle :math:`\\phi` = right side down (the body leans to the
  outside of a left turn).

Units
-----
N, m, N/m (linear stiffness), N*m/rad (roll stiffness), rad, m/s^2.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "GRAVITY",
    "AxleSuspension",
    "LateralLoadTransfer",
    "wheel_rate",
    "ride_rate",
    "axle_heave_stiffness",
    "axle_roll_stiffness",
    "roll_moment_distribution",
    "roll_axis_height",
    "roll_angle",
    "static_wheel_loads",
    "lateral_load_transfer",
    "longitudinal_load_transfer",
    "ride_height_change",
]

#: Standard gravitational acceleration [m/s^2].
GRAVITY = 9.80665


# ----------------------------------------------------------------------------
# Basic stiffness relationships
# ----------------------------------------------------------------------------
def wheel_rate(spring_rate: float, motion_ratio: float) -> float:
    """Effective spring stiffness measured at the wheel contact patch [N/m].

    The **motion ratio** :math:`MR` is spring travel divided by wheel travel.
    A spring mounted halfway along a control arm moves half as far as the
    wheel (:math:`MR = 0.5`). The lever effect appears twice (once for
    displacement, once for force), so the ratio is *squared*:

    .. math:: k_w = k_s \\cdot MR^2

    Parameters
    ----------
    spring_rate : float   :math:`k_s`, coil spring stiffness [N/m].
    motion_ratio : float  :math:`MR`, spring travel / wheel travel [-].
    """
    if spring_rate < 0:
        raise ValueError("spring_rate must be non-negative [N/m].")
    if motion_ratio <= 0:
        raise ValueError("motion_ratio must be positive.")
    return spring_rate * motion_ratio**2


def ride_rate(wheel_rate_n_per_m: float, tire_vertical_stiffness: float) -> float:
    """Combined suspension + tire vertical stiffness at one corner [N/m].

    The suspension spring and the tire act as two springs **in series**
    (the same force goes through both), so their compliances add:

    .. math:: \\frac{1}{k_{ride}} = \\frac{1}{k_w} + \\frac{1}{k_t}
              \\quad\\Rightarrow\\quad
              k_{ride} = \\frac{k_w k_t}{k_w + k_t}

    Parameters
    ----------
    wheel_rate_n_per_m : float       :math:`k_w` [N/m].
    tire_vertical_stiffness : float  :math:`k_t` [N/m].
    """
    if wheel_rate_n_per_m <= 0 or tire_vertical_stiffness <= 0:
        return 0.0
    return wheel_rate_n_per_m * tire_vertical_stiffness / (
        wheel_rate_n_per_m + tire_vertical_stiffness
    )


def axle_heave_stiffness(corner_rate: float) -> float:
    """Heave stiffness of one axle (both wheels moving up together) [N/m].

    In pure heave the left and right corners compress by the same amount, so
    the two corner springs act **in parallel**. The anti-roll bar twists
    equally at both ends and therefore adds **nothing** in heave:

    .. math:: K_z = 2\\,k_{corner}

    Parameters
    ----------
    corner_rate : float
        Stiffness of a single corner [N/m] (usually the ride rate).
    """
    return 2.0 * corner_rate


def axle_roll_stiffness(
    wheel_rate_n_per_m: float, track_width: float, anti_roll_bar_rate: float = 0.0
) -> float:
    """Roll stiffness of one axle [N*m/rad].

    Derivation: roll the body by a small angle :math:`\\phi`. Each wheel moves
    :math:`\\pm \\tfrac{t}{2}\\phi` relative to the body, creating a spring force
    :math:`k_w \\tfrac{t}{2}\\phi` acting on a lever arm :math:`\\tfrac{t}{2}`.
    Two wheels give a restoring moment of :math:`2 k_w (t/2)^2 \\phi`, so:

    .. math:: K_{\\phi} = \\frac{k_w t^2}{2} + K_{\\phi,ARB}

    Parameters
    ----------
    wheel_rate_n_per_m : float  :math:`k_w`, wheel rate of one corner [N/m].
    track_width : float         :math:`t`, lateral distance between tire centres [m].
    anti_roll_bar_rate : float  :math:`K_{\\phi,ARB}`, anti-roll bar roll stiffness [N*m/rad].
    """
    if track_width <= 0:
        raise ValueError("track_width must be positive [m].")
    if anti_roll_bar_rate < 0:
        raise ValueError("anti_roll_bar_rate must be non-negative [N*m/rad].")
    return 0.5 * wheel_rate_n_per_m * track_width**2 + anti_roll_bar_rate


@dataclass(frozen=True)
class AxleSuspension:
    """Suspension properties of one axle (front or rear).

    Attributes
    ----------
    spring_rate : float
        Coil spring stiffness :math:`k_s` [N/m].
    motion_ratio : float
        Spring travel / wheel travel :math:`MR` [-].
    track_width : float
        Distance between left and right tire centres :math:`t` [m].
    anti_roll_bar_rate : float
        Anti-roll bar contribution to axle roll stiffness [N*m/rad].
    roll_center_height : float
        Height of the axle's roll centre above the ground :math:`h_{rc}` [m].
    tire_vertical_stiffness : float
        Vertical stiffness of one tire :math:`k_t` [N/m].
    """

    spring_rate: float = 60_000.0
    motion_ratio: float = 0.8
    track_width: float = 1.55
    anti_roll_bar_rate: float = 20_000.0
    roll_center_height: float = 0.05
    tire_vertical_stiffness: float = 250_000.0

    def __post_init__(self) -> None:
        if self.track_width <= 0:
            raise ValueError("track_width must be positive [m].")
        if self.tire_vertical_stiffness <= 0:
            raise ValueError("tire_vertical_stiffness must be positive [N/m].")
        if self.roll_center_height < 0:
            raise ValueError("roll_center_height must be >= 0 [m] in this model.")
        wheel_rate(self.spring_rate, self.motion_ratio)  # validates inputs

    @property
    def wheel_rate(self) -> float:
        """:math:`k_w = k_s MR^2` [N/m]."""
        return wheel_rate(self.spring_rate, self.motion_ratio)

    @property
    def ride_rate(self) -> float:
        """Wheel rate in series with tire stiffness [N/m]."""
        return ride_rate(self.wheel_rate, self.tire_vertical_stiffness)

    @property
    def heave_stiffness(self) -> float:
        """Axle heave stiffness :math:`K_z = 2 k_{ride}` [N/m] (springs + tires)."""
        return axle_heave_stiffness(self.ride_rate)

    @property
    def roll_stiffness(self) -> float:
        """Axle roll stiffness :math:`K_\\phi = k_w t^2 / 2 + K_{ARB}` [N*m/rad].

        Tire compliance is deliberately left out: we want the roll of the
        *body* relative to the wheels, which is what the springs and bar resist.
        """
        return axle_roll_stiffness(self.wheel_rate, self.track_width, self.anti_roll_bar_rate)


# ----------------------------------------------------------------------------
# Roll geometry
# ----------------------------------------------------------------------------
def roll_moment_distribution(front_roll_stiffness: float, rear_roll_stiffness: float) -> float:
    """Front share of the elastic roll moment (a.k.a. roll stiffness distribution) [-].

    .. math:: \\lambda_f = \\frac{K_{\\phi,f}}{K_{\\phi,f} + K_{\\phi,r}}

    A stiffer front axle takes a bigger share of the roll moment, which
    transfers more load across the front tires. Because of tire load
    sensitivity, that *reduces* front grip and makes the car **understeer**.

    Parameters
    ----------
    front_roll_stiffness : float  :math:`K_{\\phi,f}` [N*m/rad].
    rear_roll_stiffness : float   :math:`K_{\\phi,r}` [N*m/rad].
    """
    total = front_roll_stiffness + rear_roll_stiffness
    if total <= 0:
        raise ValueError("Total roll stiffness must be positive.")

    # ----------------------------------------------------------------------
    # TODO (Student Exercise): Lateral Load Transfer Distribution (LLTD)
    # ----------------------------------------------------------------------
    # The fraction of roll moment carried by the FRONT axle is simply the
    # front's share of the total roll stiffness (like two springs sharing
    # one load in parallel).
    #
    # Try it yourself:
    #   1. Delete the line below and write the formula from the docstring.
    #   2. Double the front anti-roll bar in VehicleSetup. Does the front
    #      share go up? Does the car understeer more in the solver?
    #   3. Racing engineers often aim for LLTD about 5 % higher than the
    #      static front weight distribution. Find a setup that achieves it.
    # ----------------------------------------------------------------------
    front_share = front_roll_stiffness / total
    # ----------------------------------------------------------------------
    return front_share


def roll_axis_height(
    front_roll_center_height: float, rear_roll_center_height: float, weight_distribution_front: float
) -> float:
    """Height of the roll axis directly beneath the CG [m].

    The roll axis is the line joining the front and rear roll centres. The CG
    is a distance :math:`a = L(1 - w_f)` behind the front axle, so linear
    interpolation gives:

    .. math:: h_{ra} = w_f \\, h_{rc,f} + (1 - w_f)\\, h_{rc,r}

    where :math:`w_f` is the static front weight fraction.
    """
    _check_fraction(weight_distribution_front, "weight_distribution_front")
    return (
        weight_distribution_front * front_roll_center_height
        + (1.0 - weight_distribution_front) * rear_roll_center_height
    )


def roll_angle(
    mass: float,
    lateral_accel: float,
    cg_height: float,
    roll_axis_height_m: float,
    total_roll_stiffness: float,
) -> float:
    """Steady-state body roll angle [rad] (positive = right side down).

    The lateral inertia force :math:`m a_y` acts at the CG, a lever arm
    :math:`d = h - h_{ra}` above the roll axis. The springs resist with
    :math:`K_{\\phi,total}\\,\\phi`. Setting the moments equal:

    .. math:: \\phi = \\frac{m\\, a_y\\, (h - h_{ra})}{K_{\\phi,f} + K_{\\phi,r}}

    (A more advanced model also adds the small extra moment
    :math:`m g d \\phi` caused by the CG shifting sideways as the body rolls.)
    """
    if total_roll_stiffness <= 0:
        raise ValueError("total_roll_stiffness must be positive [N*m/rad].")
    roll_arm = cg_height - roll_axis_height_m
    return mass * lateral_accel * roll_arm / total_roll_stiffness


# ----------------------------------------------------------------------------
# Load distribution
# ----------------------------------------------------------------------------
def static_wheel_loads(
    mass: float, weight_distribution_front: float, gravity: float = GRAVITY
) -> dict[str, float]:
    """Vertical load on each tire with the car parked on flat ground [N].

    .. math::

        F_{z,FL} = F_{z,FR} = \\tfrac{1}{2} m g\\, w_f
        \\qquad
        F_{z,RL} = F_{z,RR} = \\tfrac{1}{2} m g\\, (1 - w_f)

    Returns
    -------
    dict
        Keys ``"FL", "FR", "RL", "RR"`` (front-left, front-right, ...).
    """
    if mass <= 0:
        raise ValueError("mass must be positive [kg].")
    _check_fraction(weight_distribution_front, "weight_distribution_front")
    weight = mass * gravity
    front = 0.5 * weight * weight_distribution_front
    rear = 0.5 * weight * (1.0 - weight_distribution_front)
    return {"FL": front, "FR": front, "RL": rear, "RR": rear}


@dataclass(frozen=True)
class LateralLoadTransfer:
    """Lateral load transfer on each axle [N].

    Each value is the load **added to the outside tire** and **removed from
    the inside tire** of that axle. A positive value with positive :math:`a_y`
    (left turn) means load moves to the right-hand tires.

    Attributes
    ----------
    front, rear : float
        Total transfer on each axle [N].
    front_geometric, rear_geometric : float
        Part transmitted directly through the suspension links via the roll
        centre (acts instantly, no body roll needed) [N].
    front_elastic, rear_elastic : float
        Part transmitted through the springs and anti-roll bar [N].
    front_roll_moment_share : float
        :math:`\\lambda_f`, front share of the elastic roll moment [-].
    """

    front: float
    rear: float
    front_geometric: float
    rear_geometric: float
    front_elastic: float
    rear_elastic: float
    front_roll_moment_share: float

    @property
    def lltd_front(self) -> float:
        """Total Lateral Load Transfer Distribution, front share [-].

        .. math:: LLTD_f = \\frac{\\Delta F_{z,f}}{\\Delta F_{z,f} + \\Delta F_{z,r}}

        Includes both the geometric and elastic paths. Returns ``nan`` when
        there is no load transfer at all (e.g. driving straight).
        """
        total = self.front + self.rear
        return self.front / total if total != 0 else float("nan")


def lateral_load_transfer(
    mass: float,
    lateral_accel: float,
    cg_height: float,
    weight_distribution_front: float,
    front: AxleSuspension,
    rear: AxleSuspension,
) -> LateralLoadTransfer:
    """Steady-state lateral load transfer on each axle [N].

    Split the total overturning moment :math:`m a_y h` into two paths:

    **Geometric (through the roll centres).** Each axle carries the lateral
    force of its own share of the mass, acting at its roll-centre height:

    .. math::

        \\Delta F_{z,f}^{geo} = \\frac{m\\,a_y\\,w_f\\,h_{rc,f}}{t_f}
        \\qquad
        \\Delta F_{z,r}^{geo} = \\frac{m\\,a_y\\,(1-w_f)\\,h_{rc,r}}{t_r}

    **Elastic (through springs + bars).** The roll moment about the roll axis,
    :math:`M_\\phi = m a_y (h - h_{ra})`, is shared by roll stiffness:

    .. math::

        \\Delta F_{z,f}^{el} = \\frac{\\lambda_f M_\\phi}{t_f}
        \\qquad
        \\Delta F_{z,r}^{el} = \\frac{(1-\\lambda_f) M_\\phi}{t_r}

    Check: :math:`\\Delta F_{z,f} t_f + \\Delta F_{z,r} t_r = m a_y h`. The total
    is fixed by mass, CG height and :math:`a_y` -- suspension tuning only
    changes how it is **distributed** between the axles!
    """
    _check_fraction(weight_distribution_front, "weight_distribution_front")
    lambda_f = roll_moment_distribution(front.roll_stiffness, rear.roll_stiffness)
    h_ra = roll_axis_height(front.roll_center_height, rear.roll_center_height, weight_distribution_front)
    roll_moment = mass * lateral_accel * (cg_height - h_ra)

    front_geo = mass * lateral_accel * weight_distribution_front * front.roll_center_height / front.track_width
    rear_geo = mass * lateral_accel * (1.0 - weight_distribution_front) * rear.roll_center_height / rear.track_width
    front_el = lambda_f * roll_moment / front.track_width
    rear_el = (1.0 - lambda_f) * roll_moment / rear.track_width

    return LateralLoadTransfer(
        front=front_geo + front_el,
        rear=rear_geo + rear_el,
        front_geometric=front_geo,
        rear_geometric=rear_geo,
        front_elastic=front_el,
        rear_elastic=rear_el,
        front_roll_moment_share=lambda_f,
    )


def longitudinal_load_transfer(
    mass: float, longitudinal_accel: float, cg_height: float, wheelbase: float
) -> float:
    """Load moved from the front axle to the rear axle under acceleration [N].

    Taking moments about the rear contact patch, the inertia force
    :math:`m a_x` at height :math:`h` is balanced by a change in front axle
    load over the wheelbase :math:`L`:

    .. math:: \\Delta F_{z,long} = \\frac{m\\, a_x\\, h}{L}

    Positive :math:`a_x` (accelerating) moves load **to the rear**; negative
    (braking) moves it **to the front**. The value is for the whole axle;
    each tire gets half.
    """
    if wheelbase <= 0:
        raise ValueError("wheelbase must be positive [m].")
    return mass * longitudinal_accel * cg_height / wheelbase


def ride_height_change(axle_vertical_force: float, heave_stiffness: float) -> float:
    """Drop in ride height caused by an extra vertical force on an axle [m].

    Hooke's law for the axle in heave:

    .. math:: \\Delta h = \\frac{F_{z}}{K_z}

    A positive force (e.g. downforce) gives a positive value, meaning the car
    sits **lower** by that amount.
    """
    if heave_stiffness <= 0:
        raise ValueError("heave_stiffness must be positive [N/m].")
    return axle_vertical_force / heave_stiffness


def _check_fraction(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1.")
