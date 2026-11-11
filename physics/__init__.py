"""
Vehicle Dynamics Lab -- core physics engine.

This package is deliberately free of any user-interface code. It depends only
on the Python standard library, NumPy and SciPy, so it can be imported from a
notebook, a script, a test suite, or a Streamlit app alike.

Modules
-------
tires       Lateral tire force models (linear, Pacejka, lookup table).
aero        Ride-height-dependent aero map: downforce, drag, aero balance.
suspension  Wheel/ride rates, roll & heave stiffness, load transfer.
vehicle     Steady-state cornering and straight-line solvers.

Quick start
-----------
>>> from physics import VehicleSetup, solve_steady_state_cornering
>>> state = solve_steady_state_cornering(VehicleSetup(), speed=20.0, radius=50.0)
>>> round(state.lateral_accel_g, 2)
0.82
"""

from .aero import AeroForces, AeroMap, AeroModel, default_aero_map, dynamic_pressure
from .suspension import (
    GRAVITY,
    AxleSuspension,
    LateralLoadTransfer,
    lateral_load_transfer,
    longitudinal_load_transfer,
    roll_moment_distribution,
    static_wheel_loads,
)
from .tires import LinearTire, LookupTire, PacejkaTire, TireModel, TireParameters
from .vehicle import (
    WHEELS,
    StraightLineState,
    VehicleSetup,
    VehicleState,
    find_max_cornering_speed,
    solve_aero_ride_heights,
    solve_steady_state_cornering,
    solve_straight_line,
    understeer_gradient,
)

__all__ = [
    "GRAVITY",
    "WHEELS",
    "AeroForces",
    "AeroMap",
    "AeroModel",
    "AxleSuspension",
    "LateralLoadTransfer",
    "LinearTire",
    "LookupTire",
    "PacejkaTire",
    "StraightLineState",
    "TireModel",
    "TireParameters",
    "VehicleSetup",
    "VehicleState",
    "default_aero_map",
    "dynamic_pressure",
    "find_max_cornering_speed",
    "lateral_load_transfer",
    "longitudinal_load_transfer",
    "roll_moment_distribution",
    "solve_aero_ride_heights",
    "solve_steady_state_cornering",
    "solve_straight_line",
    "static_wheel_loads",
    "understeer_gradient",
]
