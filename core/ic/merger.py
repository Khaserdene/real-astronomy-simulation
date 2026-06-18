"""Galaxy-merger initial conditions: two disk galaxies on a collision orbit.

Two :func:`make_disk_galaxy` systems are placed on an approaching orbit with an
impact parameter, and the second disk is inclined so the encounter is genuinely
three-dimensional.  Run with the standard gravity :class:`~core.engine.Engine`,
the close passage draws out tidal tails and bridges and the pair eventually
merges -- exactly the "collide galaxies" scenario.

All quantities are in code units (kpc, km/s, 1e10 Msun).
"""
from __future__ import annotations

import numpy as np

from core.state import State
from core.ic.disk import make_disk_galaxy


def make_merger(n_each: int = 20000,
                separation: float = 50.0,     # initial separation [kpc]
                impact_param: float = 16.0,   # perpendicular offset [kpc]
                v_approach: float = 130.0,     # relative approach speed [km/s]
                inclination: float = 60.0,     # tilt of galaxy 2 [deg]
                mass_ratio: float = 1.0,       # m2 / m1
                seed: int = 0) -> State:
    """Two disk galaxies set on a colliding orbit."""
    half = max(n_each // 2, 1)
    g1 = make_disk_galaxy(n_disk=half, n_halo=half, seed=seed)
    g2 = make_disk_galaxy(n_disk=half, n_halo=half, seed=seed + 1,
                          disk_mass=5.0 * mass_ratio,
                          halo_mass=12.0 * mass_ratio)

    # Incline galaxy 2 about the x-axis (positions + velocities).
    g2_pos = _rotate_x(g2.pos, inclination)
    g2_vel = _rotate_x(g2.vel, inclination)

    # Offset and give each galaxy half the approach velocity (+/- x), with the
    # impact parameter split along y.
    g1_pos = g1.pos + np.array([-separation / 2, -impact_param / 2, 0.0])
    g2_pos = g2_pos + np.array([+separation / 2, +impact_param / 2, 0.0])
    g1_vel = g1.vel + np.array([+v_approach / 2, 0.0, 0.0])
    g2_vel = g2_vel + np.array([-v_approach / 2, 0.0, 0.0])

    pos = np.vstack((g1_pos, g2_pos))
    vel = np.vstack((g1_vel, g2_vel))
    mass = np.concatenate((g1.mass, g2.mass))
    ptype = np.concatenate((g1.ptype, g2.ptype))
    ids = np.arange(pos.shape[0], dtype=np.int64)

    return State(
        pos=pos, vel=vel, mass=mass, ptype=ptype, ids=ids, seed=seed,
        meta={"ic": "merger", "n_each": n_each, "separation": separation,
              "impact_param": impact_param, "v_approach": v_approach,
              "inclination": inclination, "mass_ratio": mass_ratio},
    )


def _rotate_x(v: np.ndarray, deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return v @ R.T
