"""
Aerodynamics: downforce and drag from a ride-height-dependent aero map.

Background
----------
Air flowing over and under a car creates two forces we care about:

* **Downforce** pushes the car into the road, adding vertical load to the
  tires (and therefore grip) without adding mass.
* **Drag** pushes backwards, opposing motion and costing engine power.

Both grow with the *square* of speed:

.. math::

    F_{z,aero} = \\tfrac{1}{2} \\rho V^2 A \\, C_L

    F_{x,drag} = \\tfrac{1}{2} \\rho V^2 A \\, C_D

where

* :math:`\\rho` -- air density [kg/m^3] (~1.225 at sea level, 15 C)
* :math:`V`     -- vehicle speed [m/s]
* :math:`A`     -- reference (frontal) area [m^2]
* :math:`C_L`   -- downforce coefficient [-]
* :math:`C_D`   -- drag coefficient [-]

The term :math:`\\tfrac{1}{2}\\rho V^2` is called the **dynamic pressure** [Pa].

Sign convention
---------------
In this module :math:`C_L` is the **downforce** coefficient: a *positive* value
means the car is pushed *down*. (Aircraft engineers would call this a negative
lift coefficient, :math:`-C_L`.)

Why ride height matters
-----------------------
A car with an underbody (floor/diffuser) makes more downforce the closer it is
to the ground ("ground effect"). The *rake* (rear higher than front) also shifts
the **aero balance**, the share of downforce acting on the front axle. Real
teams measure these effects in a wind tunnel and store them in an **aero map**:
a table of :math:`C_L`, :math:`C_D` and balance against front and rear ride
height. We do exactly that here and interpolate between the measured points.

Units
-----
* ride heights : m (distance from ground to a reference point on the chassis)
* speed        : m/s
* forces       : N
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RegularGridInterpolator

__all__ = [
    "AIR_DENSITY_SEA_LEVEL",
    "AeroMap",
    "AeroModel",
    "AeroForces",
    "dynamic_pressure",
    "default_aero_map",
]

#: Standard sea-level air density [kg/m^3].
AIR_DENSITY_SEA_LEVEL = 1.225


def dynamic_pressure(speed: float, air_density: float = AIR_DENSITY_SEA_LEVEL) -> float:
    """Dynamic pressure of the oncoming air [Pa].

    .. math:: q = \\tfrac{1}{2} \\rho V^2

    Parameters
    ----------
    speed : float
        Vehicle speed relative to the air [m/s]. Sign does not matter.
    air_density : float
        :math:`\\rho` [kg/m^3].
    """
    return 0.5 * air_density * speed**2


@dataclass
class AeroMap:
    """2D lookup table of aero coefficients vs. front and rear ride height.

    All tables have shape ``(len(front_ride_heights), len(rear_ride_heights))``,
    i.e. rows are front ride height and columns are rear ride height.

    Values between grid points are found by **bilinear interpolation**.
    Ride heights outside the table are **clamped** to the nearest edge, because
    extrapolating wind-tunnel data is dangerous (the floor might stall!).

    Attributes
    ----------
    front_ride_heights : numpy.ndarray
        Strictly increasing front ride heights [m].
    rear_ride_heights : numpy.ndarray
        Strictly increasing rear ride heights [m].
    cl_table : numpy.ndarray
        Downforce coefficient :math:`C_L` [-] at each grid point.
    cd_table : numpy.ndarray
        Drag coefficient :math:`C_D` [-] at each grid point.
    front_balance_table : numpy.ndarray
        Fraction of total downforce acting on the front axle [-] (0 to 1).
    """

    front_ride_heights: np.ndarray
    rear_ride_heights: np.ndarray
    cl_table: np.ndarray
    cd_table: np.ndarray
    front_balance_table: np.ndarray

    def __post_init__(self) -> None:
        self.front_ride_heights = np.asarray(self.front_ride_heights, dtype=float)
        self.rear_ride_heights = np.asarray(self.rear_ride_heights, dtype=float)
        self.cl_table = np.asarray(self.cl_table, dtype=float)
        self.cd_table = np.asarray(self.cd_table, dtype=float)
        self.front_balance_table = np.asarray(self.front_balance_table, dtype=float)

        expected = (self.front_ride_heights.size, self.rear_ride_heights.size)
        for name in ("cl_table", "cd_table", "front_balance_table"):
            if getattr(self, name).shape != expected:
                raise ValueError(f"{name} must have shape {expected}.")
        for name in ("front_ride_heights", "rear_ride_heights"):
            axis = getattr(self, name)
            if axis.size < 2 or np.any(np.diff(axis) <= 0):
                raise ValueError(f"{name} must have >= 2 strictly increasing values.")
        if np.any(self.cd_table < 0):
            raise ValueError("Drag coefficients must be non-negative.")
        if np.any((self.front_balance_table < 0) | (self.front_balance_table > 1)):
            raise ValueError("front_balance_table values must lie in [0, 1].")

        grid = (self.front_ride_heights, self.rear_ride_heights)
        self._cl = RegularGridInterpolator(grid, self.cl_table, method="linear")
        self._cd = RegularGridInterpolator(grid, self.cd_table, method="linear")
        self._bal = RegularGridInterpolator(grid, self.front_balance_table, method="linear")

    def _clamp(self, front_rh: float, rear_rh: float) -> np.ndarray:
        """Clamp ride heights to the edges of the measured table."""
        f = np.clip(front_rh, self.front_ride_heights[0], self.front_ride_heights[-1])
        r = np.clip(rear_rh, self.rear_ride_heights[0], self.rear_ride_heights[-1])
        return np.array([[f, r]])

    def cl(self, front_rh: float, rear_rh: float) -> float:
        """Interpolated downforce coefficient :math:`C_L` [-]."""
        return float(self._cl(self._clamp(front_rh, rear_rh))[0])

    def cd(self, front_rh: float, rear_rh: float) -> float:
        """Interpolated drag coefficient :math:`C_D` [-]."""
        return float(self._cd(self._clamp(front_rh, rear_rh))[0])

    def front_balance(self, front_rh: float, rear_rh: float) -> float:
        """Interpolated front aero balance (front share of downforce) [-]."""
        return float(self._bal(self._clamp(front_rh, rear_rh))[0])


def default_aero_map() -> AeroMap:
    """A plausible teaching aero map for a small sports/club racing car.

    The synthetic data follows real-world trends:

    * Lower ride heights -> more downforce (ground effect), up to ~+35 %.
    * More rake (rear higher than front) -> balance shifts forward.
    * Drag rises gently with downforce (induced drag).

    Returns
    -------
    AeroMap
        A 5 x 5 map spanning 30-90 mm front and 50-110 mm rear ride height.
    """
    front = np.linspace(0.030, 0.090, 5)  # m
    rear = np.linspace(0.050, 0.110, 5)  # m
    hf, hr = np.meshgrid(front, rear, indexing="ij")

    cl_base = 1.00
    cl = cl_base * (1.0 + 0.35 * (0.090 - hf) / 0.060 + 0.10 * (0.110 - hr) / 0.060)
    cd = 0.36 + 0.08 * (cl - cl_base)
    rake = hr - hf  # m; positive = nose down
    balance = np.clip(0.42 + 1.5 * (rake - 0.020), 0.0, 1.0)

    return AeroMap(
        front_ride_heights=front,
        rear_ride_heights=rear,
        cl_table=cl,
        cd_table=cd,
        front_balance_table=balance,
    )


@dataclass(frozen=True)
class AeroForces:
    """Result of an aero calculation at one operating point.

    Attributes
    ----------
    downforce_total : float  Total downforce [N] (positive = pushes car down).
    downforce_front : float  Downforce acting on the front axle [N].
    downforce_rear : float   Downforce acting on the rear axle [N].
    drag : float             Drag force [N] (positive = opposes motion).
    cl : float               Downforce coefficient used [-].
    cd : float               Drag coefficient used [-].
    front_balance : float    Front share of downforce [-].
    """

    downforce_total: float
    downforce_front: float
    downforce_rear: float
    drag: float
    cl: float
    cd: float
    front_balance: float


@dataclass
class AeroModel:
    """Vehicle aerodynamic model: an aero map plus reference area and air density.

    Attributes
    ----------
    aero_map : AeroMap
        Coefficient tables vs. ride height.
    frontal_area : float
        Reference area :math:`A` [m^2].
    air_density : float
        :math:`\\rho` [kg/m^3].
    """

    aero_map: AeroMap = field(default_factory=default_aero_map)
    frontal_area: float = 1.8
    air_density: float = AIR_DENSITY_SEA_LEVEL

    def __post_init__(self) -> None:
        if self.frontal_area <= 0:
            raise ValueError("frontal_area must be positive [m^2].")
        if self.air_density <= 0:
            raise ValueError("air_density must be positive [kg/m^3].")

    def downforce(self, speed: float, front_rh: float, rear_rh: float) -> float:
        """Total downforce [N].

        .. math:: F_{z,aero} = \\tfrac{1}{2} \\rho V^2 A \\, C_L(h_f, h_r)
        """
        q = dynamic_pressure(speed, self.air_density)
        return q * self.frontal_area * self.aero_map.cl(front_rh, rear_rh)

    def drag(self, speed: float, front_rh: float, rear_rh: float) -> float:
        """Drag force [N].

        .. math:: F_{x,drag} = \\tfrac{1}{2} \\rho V^2 A \\, C_D(h_f, h_r)
        """
        q = dynamic_pressure(speed, self.air_density)
        return q * self.frontal_area * self.aero_map.cd(front_rh, rear_rh)

    def split_downforce(
        self, downforce_total: float, front_rh: float, rear_rh: float
    ) -> tuple[float, float]:
        """Split total downforce between the front and rear axles.

        .. math::

            F_{z,aero,f} = B_f \\, F_{z,aero}
            \\qquad
            F_{z,aero,r} = (1 - B_f) \\, F_{z,aero}

        where :math:`B_f` is the front aero balance from the map.

        Returns
        -------
        (front, rear) : tuple of float
            Axle downforce [N].
        """
        # ------------------------------------------------------------------
        # TODO (Student Exercise): Aero Balance
        # ------------------------------------------------------------------
        # Aero balance tells us what fraction of the downforce lands on the
        # FRONT axle. A balance of 0.45 means 45 % front / 55 % rear.
        #
        # The default below reads the balance straight from the wind-tunnel
        # map. Try these modifications and see how cornering changes:
        #
        #   1. Hard-code a fixed balance, e.g. front_balance = 0.40, and
        #      compare understeer/oversteer at high speed.
        #   2. Add a "front wing angle" adjustment:
        #          front_balance = map_balance + 0.01 * wing_delta_deg
        #   3. Compute the balance from a centre of pressure position x_cp
        #      measured from the front axle and the wheelbase L:
        #          front_balance = 1 - x_cp / L
        #
        # Rule to remember: front + rear must ALWAYS add up to the total.
        # ------------------------------------------------------------------
        front_balance = self.aero_map.front_balance(front_rh, rear_rh)
        front = front_balance * downforce_total
        rear = downforce_total - front
        # ------------------------------------------------------------------
        return front, rear

    def forces(self, speed: float, front_rh: float, rear_rh: float) -> AeroForces:
        """Compute all aero forces at one operating point.

        Parameters
        ----------
        speed : float     Vehicle speed [m/s].
        front_rh : float  Front ride height [m].
        rear_rh : float   Rear ride height [m].
        """
        total = self.downforce(speed, front_rh, rear_rh)
        front, rear = self.split_downforce(total, front_rh, rear_rh)
        balance = front / total if total != 0 else self.aero_map.front_balance(front_rh, rear_rh)
        return AeroForces(
            downforce_total=total,
            downforce_front=front,
            downforce_rear=rear,
            drag=self.drag(speed, front_rh, rear_rh),
            cl=self.aero_map.cl(front_rh, rear_rh),
            cd=self.aero_map.cd(front_rh, rear_rh),
            front_balance=balance,
        )
