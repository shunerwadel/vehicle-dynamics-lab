"""
Tire models: how a tire turns a slip angle into a sideways (lateral) force.

Background
----------
A rolling tire that is pointed in a slightly different direction from the
direction it is actually travelling develops a *slip angle* :math:`\\alpha`
(radians). The rubber in the contact patch twists, and that twist produces a
lateral force :math:`F_y` (newtons) that pushes the car around the corner.

For small slip angles the relationship is close to a straight line:

.. math::

    F_y = C_\\alpha \\, \\alpha

where :math:`C_\\alpha` is the **cornering stiffness** in N/rad.

The force cannot grow forever. Friction limits it to

.. math::

    |F_y| \\le \\mu F_z

where :math:`\\mu` is the peak friction coefficient (dimensionless) and
:math:`F_z` is the vertical load on the tire (N).

Load sensitivity
----------------
Real tires get *less* grippy per newton as you load them up. Doubling the load
does **not** double the grip. We model this with a simple linear drop in the
friction coefficient:

.. math::

    \\mu(F_z) = \\mu_0 \\left[ 1 - k_{\\mu} \\frac{F_z - F_{z0}}{F_{z0}} \\right]

where :math:`F_{z0}` is the nominal (reference) load and :math:`k_{\\mu}` is the
load-sensitivity coefficient. This effect is the key reason lateral load
transfer reduces the total grip of an axle.

Sign convention
---------------
Positive slip angle produces positive lateral force (``F_y = +C * alpha``).
This keeps the algebra friendly for students; every model here is an odd
function, so ``F_y(-alpha) == -F_y(alpha)``.

Units
-----
* slip angle ``alpha``      : rad
* vertical load ``fz``      : N
* lateral force ``fy``      : N
* cornering stiffness       : N/rad
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, minimize_scalar

__all__ = [
    "TireParameters",
    "TireModel",
    "LinearTire",
    "PacejkaTire",
    "LookupTire",
]

#: Smallest load-sensitivity factor we allow. Prevents friction from going to
#: zero or negative at absurdly high loads, which would be unphysical.
_MIN_LOAD_FACTOR = 0.1


@dataclass(frozen=True)
class TireParameters:
    """Physical parameters shared by every tire model.

    Attributes
    ----------
    cornering_stiffness : float
        :math:`C_{\\alpha 0}`, cornering stiffness at the nominal load
        :math:`F_{z0}` [N/rad]. A typical road car tire is roughly
        40,000 - 90,000 N/rad.
    nominal_load : float
        :math:`F_{z0}`, the reference vertical load at which
        ``cornering_stiffness`` and ``peak_friction`` were measured [N].
    peak_friction : float
        :math:`\\mu_0`, peak lateral friction coefficient at the nominal load
        [-]. Street tires ~1.0, racing slicks ~1.6+.
    load_sensitivity : float
        :math:`k_{\\mu}`, fractional loss of friction per unit of normalised
        extra load [-]. ``0.0`` means no load sensitivity. Typical: 0.1-0.25.
    """

    cornering_stiffness: float = 60_000.0
    nominal_load: float = 4_000.0
    peak_friction: float = 1.2
    load_sensitivity: float = 0.15

    def __post_init__(self) -> None:
        if self.cornering_stiffness <= 0:
            raise ValueError("cornering_stiffness must be positive [N/rad].")
        if self.nominal_load <= 0:
            raise ValueError("nominal_load must be positive [N].")
        if self.peak_friction <= 0:
            raise ValueError("peak_friction must be positive.")
        if self.load_sensitivity < 0:
            raise ValueError("load_sensitivity must be >= 0.")


class TireModel:
    """Base class for every tire model.

    Sub-classes must implement :meth:`lateral_force`. Everything else
    (load sensitivity, peak force, inverse lookup) is shared.
    """

    def __init__(self, params: TireParameters | None = None) -> None:
        self.params = params if params is not None else TireParameters()

    # ------------------------------------------------------------------
    # Load sensitivity
    # ------------------------------------------------------------------
    def load_factor(self, fz: float) -> float:
        """Dimensionless grip-degradation factor caused by vertical load.

        .. math::

            f(F_z) = \\max\\left(f_{min},\\;
                     1 - k_{\\mu} \\frac{F_z - F_{z0}}{F_{z0}}\\right)

        Returns 1.0 at the nominal load, < 1.0 above it, > 1.0 below it.
        """
        p = self.params
        factor = 1.0 - p.load_sensitivity * (fz - p.nominal_load) / p.nominal_load
        return max(_MIN_LOAD_FACTOR, factor)

    def friction_coefficient(self, fz: float) -> float:
        """Peak friction coefficient at load ``fz``: :math:`\\mu(F_z) = \\mu_0 f(F_z)` [-]."""
        return self.params.peak_friction * self.load_factor(fz)

    def cornering_stiffness(self, fz: float) -> float:
        """Cornering stiffness at load ``fz`` [N/rad].

        Stiffness grows with load, but with the same diminishing return as
        friction:

        .. math::

            C_\\alpha(F_z) = C_{\\alpha 0} \\frac{F_z}{F_{z0}} f(F_z)

        A tire with zero (or negative) load has zero stiffness.
        """
        if fz <= 0.0:
            return 0.0
        p = self.params
        return p.cornering_stiffness * (fz / p.nominal_load) * self.load_factor(fz)

    def max_lateral_force(self, fz: float) -> float:
        """Friction-limited peak lateral force: :math:`F_{y,max} = \\mu(F_z) F_z` [N]."""
        if fz <= 0.0:
            return 0.0
        return self.friction_coefficient(fz) * fz

    # ------------------------------------------------------------------
    # Model-specific behaviour
    # ------------------------------------------------------------------
    def lateral_force(self, alpha: float, fz: float) -> float:
        """Lateral force [N] for slip angle ``alpha`` [rad] and load ``fz`` [N]."""
        raise NotImplementedError

    def peak_slip_angle(self, fz: float) -> float:
        """Slip angle [rad] at which the lateral force reaches its peak.

        The default implementation searches numerically between 0 and 30 deg.
        Sub-classes with a closed-form answer override this.
        """
        if fz <= 0.0:
            return 0.0
        result = minimize_scalar(
            lambda a: -self.lateral_force(a, fz),
            bounds=(0.0, np.radians(30.0)),
            method="bounded",
            options={"xatol": 1e-7},
        )
        return float(result.x)

    def slip_angle_for_force(self, fy: float, fz: float) -> float:
        """Inverse tire model: slip angle [rad] needed to produce force ``fy`` [N].

        Only the *stable* (pre-peak) branch of the curve is searched, which is
        the physically meaningful answer for steady-state driving.

        Raises
        ------
        ValueError
            If ``|fy|`` exceeds what the tire can generate at this load.
        """
        if fy == 0.0:
            return 0.0
        alpha_peak = self.peak_slip_angle(fz)
        f_peak = self.lateral_force(alpha_peak, fz)
        if abs(fy) > f_peak * (1.0 + 1e-9):
            raise ValueError(
                f"Requested |Fy|={abs(fy):.1f} N exceeds tire capacity "
                f"{f_peak:.1f} N at Fz={fz:.1f} N."
            )
        target = min(abs(fy), f_peak)
        if np.isclose(target, f_peak):
            return float(np.copysign(alpha_peak, fy))
        alpha = brentq(lambda a: self.lateral_force(a, fz) - target, 0.0, alpha_peak, xtol=1e-12)
        return float(np.copysign(alpha, fy))


class LinearTire(TireModel):
    """Linear tire with a hard friction ceiling (the simplest useful model).

    .. math::

        F_y = \\operatorname{clip}\\left(C_\\alpha(F_z)\\,\\alpha,\\;
              -\\mu(F_z) F_z,\\; +\\mu(F_z) F_z\\right)

    The force rises in a straight line until it hits the friction limit,
    then stays flat. Great for building intuition; real tires round off
    the corner more gradually (see :class:`PacejkaTire`).
    """

    def lateral_force(self, alpha: float, fz: float) -> float:
        if fz <= 0.0:
            return 0.0
        f_max = self.max_lateral_force(fz)
        return float(np.clip(self.cornering_stiffness(fz) * alpha, -f_max, f_max))

    def peak_slip_angle(self, fz: float) -> float:
        """Slip angle where the straight line meets the friction ceiling.

        .. math:: \\alpha_{peak} = \\frac{\\mu(F_z) F_z}{C_\\alpha(F_z)}
        """
        c = self.cornering_stiffness(fz)
        if c <= 0.0:
            return 0.0
        return self.max_lateral_force(fz) / c

    def slip_angle_for_force(self, fy: float, fz: float) -> float:
        """Closed-form inverse: :math:`\\alpha = F_y / C_\\alpha` (if within the friction limit)."""
        if fy == 0.0:
            return 0.0
        if abs(fy) > self.max_lateral_force(fz) * (1.0 + 1e-9):
            raise ValueError(
                f"Requested |Fy|={abs(fy):.1f} N exceeds tire capacity "
                f"{self.max_lateral_force(fz):.1f} N at Fz={fz:.1f} N."
            )
        return fy / self.cornering_stiffness(fz)


@dataclass(frozen=True)
class _PacejkaShape:
    """Shape factors for the simplified Magic Formula."""

    c: float = 1.3  # shape factor C [-]; ~1.3 for lateral force
    e: float = -1.0  # curvature factor E [-]; controls how "peaky" the curve is


class PacejkaTire(TireModel):
    """Simplified Pacejka "Magic Formula" tire.

    .. math::

        F_y = D \\sin\\Big( C \\arctan\\big( B\\alpha - E\\,[B\\alpha - \\arctan(B\\alpha)] \\big) \\Big)

    with

    * :math:`D = \\mu(F_z) F_z`                 -- peak force [N]
    * :math:`C`                                 -- shape factor [-]
    * :math:`B = C_\\alpha(F_z) / (C D)`        -- stiffness factor [1/rad]
    * :math:`E`                                 -- curvature factor [-]

    Choosing :math:`B` this way guarantees that the slope at the origin,
    :math:`B C D`, equals the cornering stiffness :math:`C_\\alpha(F_z)`, so the
    Magic Formula agrees with :class:`LinearTire` for small slip angles.
    """

    def __init__(
        self,
        params: TireParameters | None = None,
        shape_factor: float = 1.3,
        curvature_factor: float = -1.0,
    ) -> None:
        super().__init__(params)
        if not 1.0 <= shape_factor < 2.0:
            raise ValueError("shape_factor C should be in [1, 2) for a lateral-force curve.")
        if curvature_factor >= 1.0:
            raise ValueError("curvature_factor E must be < 1.")
        self.shape = _PacejkaShape(c=shape_factor, e=curvature_factor)

    def lateral_force(self, alpha: float, fz: float) -> float:
        if fz <= 0.0:
            return 0.0
        d = self.max_lateral_force(fz)
        c = self.shape.c
        e = self.shape.e
        b = self.cornering_stiffness(fz) / (c * d)
        ba = b * alpha
        return float(d * np.sin(c * np.arctan(ba - e * (ba - np.arctan(ba)))))


@dataclass
class LookupTire(TireModel):
    """Tire defined by a measured table of normalised force vs. slip angle.

    The table stores the *normalised* lateral force
    :math:`\\bar F_y = F_y / (\\mu_0 F_z)` (between -1 and 1) at the nominal load.
    At any other load the curve is scaled by the peak force:

    .. math::

        F_y(\\alpha, F_z) = \\bar F_y(\\alpha)\\; \\mu(F_z) F_z

    Values between table points are linearly interpolated; slip angles beyond
    the table are held at the last value (no extrapolation).

    Attributes
    ----------
    slip_angles_deg : numpy.ndarray
        Strictly increasing, non-negative slip angles [deg]. Must start at 0.
    normalised_force : numpy.ndarray
        :math:`\\bar F_y` at each slip angle [-]. Must start at 0.
    params : TireParameters
        Shared tire parameters (friction, load sensitivity).
    """

    slip_angles_deg: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0])
    )
    normalised_force: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.28, 0.54, 0.74, 0.88, 0.96, 1.0, 0.98, 0.95, 0.90])
    )
    params: TireParameters = field(default_factory=TireParameters)

    def __post_init__(self) -> None:
        self.slip_angles_deg = np.asarray(self.slip_angles_deg, dtype=float)
        self.normalised_force = np.asarray(self.normalised_force, dtype=float)
        if self.slip_angles_deg.shape != self.normalised_force.shape:
            raise ValueError("slip_angles_deg and normalised_force must be the same length.")
        if self.slip_angles_deg[0] != 0.0 or self.normalised_force[0] != 0.0:
            raise ValueError("Lookup table must start at (0 deg, 0 force).")
        if np.any(np.diff(self.slip_angles_deg) <= 0):
            raise ValueError("slip_angles_deg must be strictly increasing.")
        self._alpha_table = np.radians(self.slip_angles_deg)

    def lateral_force(self, alpha: float, fz: float) -> float:
        if fz <= 0.0:
            return 0.0
        fy_bar = np.interp(abs(alpha), self._alpha_table, self.normalised_force)
        return float(np.copysign(fy_bar * self.max_lateral_force(fz), alpha))

    def cornering_stiffness(self, fz: float) -> float:
        """Initial slope of the table, scaled by peak force [N/rad].

        .. math:: C_\\alpha \\approx \\frac{\\bar F_y(\\alpha_1)}{\\alpha_1} \\mu(F_z) F_z
        """
        if fz <= 0.0:
            return 0.0
        slope = self.normalised_force[1] / self._alpha_table[1]
        return float(slope * self.max_lateral_force(fz))

    def peak_slip_angle(self, fz: float) -> float:
        """The table point with the largest force (exact for piecewise-linear data)."""
        return float(self._alpha_table[int(np.argmax(self.normalised_force))])
