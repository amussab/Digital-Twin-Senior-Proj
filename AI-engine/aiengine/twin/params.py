"""All rotor / twin parameters in one place, with provenance.

Tags: [CITED], [TEAM DOC], [ASSUMED - needs ME confirmation].
The Bently Nevada RK4 datasheet (doc 141592 Rev K) is cited by the COE doc but its geometry
table was NOT available to this agent, so every geometric value below is an assumption to be
replaced by the ME member with measured/datasheet values.
"""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class RotorParams:
    # --- material ---
    E: float = 2.0e11            # Pa  [CITED: textbook value for carbon steel, ~200-210 GPa]
    rho: float = 7850.0          # kg/m3 [CITED: textbook value for steel]
    # --- shaft geometry (RK4: 10 mm class shaft) ---
    shaft_d: float = 0.010       # m  [ASSUMED - needs ME confirmation]
    length: float = 0.560        # m  total shaft length [ASSUMED - needs ME confirmation]
    n_elem: int = 28             # elements, 0.02 m each [model choice; mesh convergence not studied]
    # --- node indices (node i is at x = i * length / n_elem) ---
    bearing_nodes: tuple = (3, 25)   # x = 0.06 m, 0.50 m [ASSUMED - needs ME confirmation]
    probe_nodes: tuple = (5, 23)     # x = 0.10 m, 0.46 m [ASSUMED]; one radial probe per plane [TEAM DOC: COE doc]
    disk_node: int = 11              # x = 0.22 m, off-centre on purpose [ASSUMED]
    disk_mass: float = 0.8           # kg lumped, translational only, no gyroscopics [ASSUMED]
    # --- bearings ---
    k_nominal: tuple = (5.0e4, 5.0e4)   # N/m healthy bearing stiffness [ASSUMED - needs ME confirmation]
    c_bearing: tuple = (200.0, 200.0)   # N s/m viscous damping, held fixed [ASSUMED]
    # --- unbalance (nominal simulation value; the real one comes from commissioning) ---
    unbalance_nominal: float = 1.0e-5   # kg m (10 g mm) [ASSUMED, simulation only]
    unbalance_phase_nominal: float = 0.5  # rad [ASSUMED, simulation only]
    # --- failure criterion ---
    failure_drop: float = 0.25       # K drop >= 25 % [TEAM DOC: Project Overview, failure criterion]
    # --- identification noise model (confidence measure only) ---
    rel_noise: float = 0.05          # relative complex-measurement sigma [ASSUMED]
    # --- twin runtime ---
    n_commission: int = 20           # commissioning windows [design choice]
    min_rul_points: int = 8          # points needed before a physics RUL is issued [design choice]
    rul_fit_points: int = 200        # recent points used for the trend fit [design choice]
    max_rel_std: float = 0.15        # identifications with predicted K sigma above this are not used [design choice]


DEFAULT = RotorParams()


def as_dict(p: RotorParams = DEFAULT) -> dict:
    return asdict(p)
