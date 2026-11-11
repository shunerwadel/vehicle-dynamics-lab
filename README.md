# Vehicle Dynamics Lab

An educational, steady-state vehicle dynamics engine for high school students.
The `physics/` package is pure Python + NumPy/SciPy with **no UI code**, so it
can be imported unchanged into notebooks, scripts, or a Streamlit front end.

## Quick start

You need **Python 3.14** (from python.org; tick *"Add python.exe to PATH"*) and Git.

```powershell
git clone https://github.com/shunerwadel/vehicle-dynamics-lab.git
cd vehicle-dynamics-lab
powershell -ExecutionPolicy Bypass -File .\setup.ps1   # once: creates .venv, installs packages, runs the tests
powershell -ExecutionPolicy Bypass -File .\run.ps1     # starts the app and opens http://localhost:8501
```

* `setup.ps1 -Locked` installs the exact package versions from `requirements-lock.txt`
  instead of the minimum versions in `requirements.txt`. Use it if a newer package breaks something.
* `run.ps1 -Port 8502` serves on another port; `-NoBrowser` skips opening the browser. Stop the app with Ctrl+C.
* After `git pull`, run `setup.ps1` again to pick up any new packages and re-check the tests.
* Saved baselines live in `user_data/` (not in Git), so each computer keeps its own.

<details>
<summary>Manual setup (macOS/Linux, or without the scripts)</summary>

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt      # Windows: .\.venv\Scripts\python.exe
.venv/bin/python -m pytest
.venv/bin/python -m streamlit run app.py                 # then open http://localhost:8501
```

`.streamlit/config.toml` runs Streamlit headless (no first-launch email prompt, no browser
auto-open) and turns off usage statistics.
</details>

The web app (`app.py`) has a sidebar for the car setup, an **Experiment Runner** tab
(cornering or straight-line tests with Plotly charts and theory boxes), and a
**Setup Assistant** chat tab where students type changes such as
"stiffen front springs by 20%" (parsed by `utils/agent_parser.py`).

```python
import math
from physics import VehicleSetup, solve_steady_state_cornering, find_max_cornering_speed

car = VehicleSetup()                                   # ~1200 kg club racer
state = solve_steady_state_cornering(car, speed=22.0, radius=50.0)  # +R = left turn
print(state.lateral_accel_g, math.degrees(state.steer_angle), state.tire_vertical_loads)
print(find_max_cornering_speed(car, radius=50.0))      # m/s
```

## Physics architecture

```
tires.py ──┐
aero.py ───┼──► vehicle.py  (steady-state solvers)
suspension.py ┘
```

| Module | Responsibility | Key equations |
|---|---|---|
| `tires.py` | Lateral force from slip angle and load. `LinearTire`, `PacejkaTire`, `LookupTire`, all with load sensitivity and a friction limit. | $F_y = C_\alpha \alpha$, $\lvert F_y\rvert \le \mu(F_z) F_z$ |
| `aero.py` | 2D aero map (C_L, C_D, front balance vs. front/rear ride height) with bilinear interpolation. | $F = \tfrac12 \rho V^2 A C$ |
| `suspension.py` | Wheel/ride rate, heave and roll stiffness, roll moment distribution, lateral and longitudinal load transfer, roll angle. | $k_w = k_s MR^2$, $K_\phi = k_w t^2/2 + K_{ARB}$ |
| `vehicle.py` | `VehicleSetup` / `VehicleState` dataclasses; cornering, straight-line, and max-speed solvers. | $a_y = V^2/R$, $\delta = L/R + \alpha_f - \alpha_r$ |

### How the cornering solver works

1. Lateral acceleration $a_y = V^2/R$.
2. Fixed-point iteration couples aero and suspension: downforce compresses the
   springs, which changes the ride height, which changes the downforce.
3. Per-tire vertical loads = static weight + aero + lateral load transfer
   (geometric part through the roll centres, elastic part shared by roll stiffness).
4. Bicycle-model force and yaw balance give the force each axle must make.
5. For each axle, solve for the shared slip angle at which both tires (on
   different loads) produce that force. If they can't, the state is flagged
   `is_feasible = False` with the `limiting_axle`.
6. Steer, sideslip, and roll angles follow from geometry and roll stiffness.

**Conventions:** ISO 8855 (x forward, y left, z up). Positive radius = left turn.
SI units throughout; angles in radians.

## Student exercises

Search the code for `TODO (Student Exercise)`:

* `suspension.roll_moment_distribution` covers Lateral Load Transfer Distribution (LLTD).
* `aero.AeroModel.split_downforce` covers aero balance.

Each block has a working default implementation, so the code always runs. The
tests in `tests/test_physics.py` check the physical laws (force, moment, and
roll balance). If an edit breaks one of those laws, the tests will flag it.
