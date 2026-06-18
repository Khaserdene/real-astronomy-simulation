"""Cosmological initial conditions: a perturbed particle field that collapses
into a cosmic web.

We use the Zel'dovich approximation: start from a regular grid of particles and
displace each by a smooth, curl-free random field built from a handful of Fourier
modes.  Where the displacements converge the density rises; under gravity those
overdensities collapse first and grow hierarchically into clumps (proto-galaxies)
strung along filaments with voids between -- the cosmic web.  A gentle outward
Hubble flow sets the initial expansion the collapse competes against.

Run with the standard gravity :class:`~core.engine.Engine`.  (This isolated box
has open boundaries, not periodic -- the central region shows the web cleanly;
the GIZMO/GADGET research arm is the route to fully periodic cosmology.)

All quantities in code units (kpc, km/s, 1e10 Msun).
"""
from __future__ import annotations

import numpy as np

from core.state import State, PTYPE_DM


def make_cosmo_box(n_side: int = 22,
                   box_size: float = 24.0,     # kpc
                   total_mass: float = 120.0,  # 1e10 Msun
                   amplitude: float = 1.6,      # Zel'dovich displacement [kpc]
                   n_modes: int = 14,
                   hubble: float = 6.0,         # initial expansion [km/s/kpc]
                   vel_factor: float = 8.0,     # growing-mode peculiar velocity
                   seed: int = 0) -> State:
    """Build a Zel'dovich-perturbed particle cube that collapses into a web."""
    rng = np.random.default_rng(seed)

    # Regular grid of Lagrangian positions q, centred on the origin.
    lin = (np.arange(n_side) + 0.5) / n_side * box_size - box_size / 2
    qx, qy, qz = np.meshgrid(lin, lin, lin, indexing="ij")
    q = np.column_stack((qx.ravel(), qy.ravel(), qz.ravel()))
    n = q.shape[0]

    # Smooth, curl-free displacement field psi(q) = sum_m (A_m/k) k_hat sin(k.q+phi).
    psi = np.zeros_like(q)
    kfun = 2.0 * np.pi / box_size
    for _ in range(n_modes):
        kvec = kfun * rng.integers(1, 4, size=3) * rng.choice([-1, 1], size=3)
        kmag = np.linalg.norm(kvec)
        khat = kvec / kmag
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.normal(0.0, 1.0) / kmag
        wave = np.sin(q @ kvec + phase)
        psi += amp * wave[:, None] * khat
    # Normalise to the requested rms displacement amplitude.
    psi *= amplitude / (np.sqrt((psi ** 2).sum(1)).mean() + 1e-9)

    pos = q + psi
    # Hubble expansion outward + growing-mode peculiar velocity along psi.
    vel = hubble * q + vel_factor * psi

    mass = np.full(n, total_mass / n)
    ptype = np.full(n, PTYPE_DM, dtype=np.int32)
    ids = np.arange(n, dtype=np.int64)
    return State(
        pos=pos, vel=vel, mass=mass, ptype=ptype, ids=ids, seed=seed,
        meta={"ic": "cosmo", "n_side": n_side, "box_size": box_size,
              "total_mass": total_mass},
    )
