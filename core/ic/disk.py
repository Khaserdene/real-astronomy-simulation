"""Exponential-disk galaxy initial conditions (the visual demo IC).

A stellar exponential disk is embedded in a live Plummer dark-matter halo so the
disk has a realistic rotation curve and is partially stabilised.  This is a
pragmatic "looks like a galaxy" IC, *not* a perfectly relaxed equilibrium -- bar
and spiral structure forming over time is expected and desirable.  Use
:func:`core.ic.plummer.make_plummer` for the strict equilibrium correctness test.

Determinism: identical ``seed`` reproduces the same galaxy at any particle count,
which is what lets a low-N preview and a high-N production run be "the same
galaxy, sampled differently".

All quantities are in code units (kpc, km/s, 1e10 Msun).
"""
from __future__ import annotations

import numpy as np

from core.state import State, PTYPE_STAR, PTYPE_DM
from core.units import G
from core.ic.plummer import _random_directions


def make_disk_galaxy(n_disk: int = 20000,
                     n_halo: int = 20000,
                     disk_mass: float = 5.0,      # 1e10 Msun
                     disk_scale: float = 3.0,     # Rd [kpc]
                     disk_height: float = 0.25,   # z0 [kpc]
                     halo_mass: float = 12.0,     # 1e10 Msun (disk-dominated)
                     halo_scale: float = 8.0,     # a  [kpc]
                     sigma_frac: float = 0.1,     # radial dispersion / v_circ
                     seed: int = 0) -> State:
    """Build a stellar exponential disk inside a live Plummer DM halo."""
    rng = np.random.default_rng(seed)

    disk_pos = _sample_exponential_disk(rng, n_disk, disk_scale, disk_height)
    halo_pos = _sample_plummer_positions(rng, n_halo, halo_scale)

    m_disk = np.full(n_disk, disk_mass / n_disk)
    m_halo = np.full(n_halo, halo_mass / n_halo)

    pos = np.vstack((disk_pos, halo_pos))
    mass = np.concatenate((m_disk, m_halo))

    # --- spherically-enclosed mass profile from the actual particles ---
    r_sph = np.linalg.norm(pos, axis=1)
    order = np.argsort(r_sph)
    m_cumulative = np.empty_like(mass)
    m_cumulative[order] = np.cumsum(mass[order])
    # softened circular speed (avoid divide-by-zero at the centre)
    eps = 0.05 * disk_scale
    v_circ = np.sqrt(G * m_cumulative / np.sqrt(r_sph ** 2 + eps ** 2))

    vel = np.zeros_like(pos)

    # Disk stars: circular rotation + anisotropic dispersion in cylindrical
    # coordinates (small vertical dispersion keeps the disk thin; radial/
    # azimuthal dispersion sets the Toomre temperature for spiral/bar growth).
    R_disk = np.linalg.norm(disk_pos[:, :2], axis=1) + 1e-6
    e_R = np.column_stack((disk_pos[:, 0] / R_disk, disk_pos[:, 1] / R_disk,
                           np.zeros(n_disk)))
    e_phi = np.column_stack((-disk_pos[:, 1] / R_disk, disk_pos[:, 0] / R_disk,
                             np.zeros(n_disk)))
    vc_disk = v_circ[:n_disk]
    sigma_R = sigma_frac * vc_disk
    d_R = rng.normal(0.0, sigma_R)
    d_phi = rng.normal(0.0, 0.5 * sigma_R)
    d_z = rng.normal(0.0, 0.35 * sigma_R)
    vel[:n_disk] = (e_R * d_R[:, None] + e_phi * (vc_disk + d_phi)[:, None])
    vel[:n_disk, 2] = d_z

    # Halo: cool isotropic dispersion (~0.3 v_circ). Kept deliberately on the
    # cold side so the crude (non-self-consistent) halo does not eject particles.
    vc_halo = v_circ[n_disk:]
    vel[n_disk:] = _random_directions(rng, n_halo) \
        * np.abs(rng.normal(0.0, 0.3 * vc_halo[:, None]))

    # Remove any net drift so the galaxy sits still in the box.
    vel -= np.average(vel, axis=0, weights=mass)

    ptype = np.concatenate((np.full(n_disk, PTYPE_STAR, dtype=np.int32),
                            np.full(n_halo, PTYPE_DM, dtype=np.int32)))
    ids = np.arange(n_disk + n_halo, dtype=np.int64)

    return State(
        pos=pos, vel=vel, mass=mass, ptype=ptype, ids=ids,
        time=0.0, step=0, seed=seed,
        meta={"ic": "disk_galaxy", "n_disk": n_disk, "n_halo": n_halo,
              "disk_mass": disk_mass, "disk_scale": disk_scale,
              "halo_mass": halo_mass, "halo_scale": halo_scale},
    )


def _sample_exponential_disk(rng, n, Rd, z0):
    """Sample positions from Sigma(R) ~ exp(-R/Rd) with exponential vertical."""
    # Radial: invert the cumulative  M(<R)/M = 1 - (1 + R/Rd) exp(-R/Rd).
    u = rng.random(n)
    x = np.linspace(0.0, 20.0, 100000)               # x = R / Rd
    cdf = 1.0 - (1.0 + x) * np.exp(-x)
    R = np.interp(u, cdf, x) * Rd
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    # Vertical: exponential with scale height z0 (symmetric).
    z = rng.exponential(z0, n) * rng.choice((-1.0, 1.0), n)
    return np.column_stack((R * np.cos(phi), R * np.sin(phi), z))


def _sample_plummer_positions(rng, n, a):
    """Sample isotropic Plummer positions with scale radius ``a``."""
    x1 = rng.uniform(0.0, 0.999, n)
    r = a / np.sqrt(x1 ** (-2.0 / 3.0) - 1.0)
    return _random_directions(rng, n) * r[:, None]
