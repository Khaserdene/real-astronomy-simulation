"""Plummer-sphere initial conditions.

A Plummer sphere is the textbook self-gravitating system in virial equilibrium,
which makes it the ideal correctness test for the gravity solver and integrator:
total energy should be conserved and 2*KE + PE ~= 0 (virial) for the lifetime of
the run.  Sampling follows the standard inverse-transform + rejection method
(Aarseth, Henon & Wielen 1974).

All quantities are returned in code units (kpc, km/s, 1e10 Msun).
"""
from __future__ import annotations

import numpy as np

from core.state import State, PTYPE_DM
from core.units import G


def make_plummer(n: int = 10000,
                 total_mass: float = 10.0,   # 1e10 Msun -> 1e11 Msun
                 scale_radius: float = 1.0,  # kpc
                 seed: int = 0) -> State:
    """Sample ``n`` equal-mass particles from a Plummer model.

    Parameters
    ----------
    n : number of particles.
    total_mass : total system mass [1e10 Msun].
    scale_radius : Plummer scale radius ``a`` [kpc].
    seed : RNG seed (makes the IC deterministic and reproducible).
    """
    rng = np.random.default_rng(seed)
    a = scale_radius
    M = total_mass

    # --- radii from the inverse CDF  M(<r)/M = r^3 / (r^2 + a^2)^{3/2} ---
    x1 = rng.random(n)
    r = a / np.sqrt(x1 ** (-2.0 / 3.0) - 1.0)

    # --- isotropic positions ---
    pos = _random_directions(rng, n) * r[:, None]

    # --- speeds via rejection on g(q) = q^2 (1 - q^2)^{7/2} ---
    q = np.empty(n)
    remaining = np.arange(n)
    while remaining.size:
        x4 = rng.random(remaining.size)
        x5 = rng.random(remaining.size)
        accept = 0.1 * x5 < x4 ** 2 * (1.0 - x4 ** 2) ** 3.5
        idx = remaining[accept]
        q[idx] = x4[accept]
        remaining = remaining[~accept]

    v_esc = np.sqrt(2.0 * G * M / a) * (1.0 + (r / a) ** 2) ** -0.25
    speed = q * v_esc
    vel = _random_directions(rng, n) * speed[:, None]

    mass = np.full(n, M / n)
    ptype = np.full(n, PTYPE_DM, dtype=np.int32)
    ids = np.arange(n, dtype=np.int64)

    return State(
        pos=pos, vel=vel, mass=mass, ptype=ptype, ids=ids,
        time=0.0, step=0, seed=seed,
        meta={"ic": "plummer", "total_mass": M, "scale_radius": a, "n": n},
    )


def _random_directions(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return ``n`` uniformly-distributed unit vectors on the sphere."""
    cos_theta = rng.uniform(-1.0, 1.0, n)
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.column_stack(
        (sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta)
    )
