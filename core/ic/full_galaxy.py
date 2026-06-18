"""Full multi-component galaxy IC: stars + live dark-matter halo + SPH gas.

Unlike the split scenarios (``disk`` = stars only, ``gas`` = gas only, ``living``
= gas+stars in a *fixed analytic* halo), this builds the whole galaxy as live
particles that all self-gravitate together:

  * **stars**  -- an exponential stellar disk (collisionless);
  * **dark matter** -- a live Plummer halo (collisionless, dominant mass) -- not
    an analytic potential, so it responds to the disk and to mergers;
  * **gas**    -- an exponential SPH gas disk (pressure, shocks, and -- in the
    living engine -- star formation + supernova feedback).

Returned in the *engine's* species convention (0=gas, 1=star, 2=dark matter),
ready for :meth:`core.living_galaxy_engine.LivingGalaxyEngine.setup`.  Dust is a
render-time attribute derived from gas density, not a particle species.

All quantities are code units (kpc, km/s, 1e10 Msun); ``u`` is (km/s)^2.
"""
from __future__ import annotations

import numpy as np

from core.state import PTYPE_STAR


def make_full_galaxy(n: int = 30000, seed: int = 0,
                     gas_fraction: float = 0.18,
                     star_fraction: float = 0.32,
                     disk_mass: float = 5.0, disk_scale: float = 3.0,
                     disk_height: float = 0.25, gas_mass: float = 1.2,
                     halo_mass: float = 14.0, halo_scale: float = 8.0,
                     sound_speed: float = 12.0):
    """Build a stars + live-DM + gas galaxy.

    Returns ``(pos, vel, mass, u, species)`` with ``species`` in engine
    convention (0 gas, 1 star, 2 dark matter).
    """
    from core.ic.disk import make_disk_galaxy
    from core.ic.gas_disk import make_gas_disk

    n_gas = max(int(n * gas_fraction), 1)
    n_star = max(int(n * star_fraction), 1)
    n_dm = max(n - n_gas - n_star, 1)

    # Stars (disk) + live DM (halo): velocities set from the live enclosed mass.
    sd = make_disk_galaxy(n_disk=n_star, n_halo=n_dm, seed=seed,
                          disk_mass=disk_mass, disk_scale=disk_scale,
                          disk_height=disk_height, halo_mass=halo_mass,
                          halo_scale=halo_scale)
    sd_species = np.where(sd.ptype == PTYPE_STAR, 1, 2).astype(np.int32)

    # Gas disk: rotates in an analytic potential chosen to match the live
    # stars+halo mass, so it starts on sensible orbits; live gravity takes over.
    pot = dict(M_d=disk_mass, a=disk_scale, b=disk_height,
               M_h=halo_mass, a_h=halo_scale)
    gpos, gvel, gmass, gu = make_gas_disk(
        n=n_gas, gas_mass=gas_mass, disk_scale=disk_scale,
        disk_height=disk_height, sound_speed=sound_speed, pot=pot, seed=seed + 1)

    pos = np.vstack((gpos, sd.pos))
    vel = np.vstack((gvel, sd.vel))
    mass = np.concatenate((gmass, sd.mass))
    species = np.concatenate((np.zeros(n_gas, np.int32), sd_species))
    # u is per-particle but only gas uses it; non-gas gets a harmless placeholder.
    u = np.concatenate((gu, np.full(len(sd.mass), 200.0)))

    # Remove any residual net drift so the galaxy sits still.
    vel -= np.average(vel, axis=0, weights=mass)
    return pos, vel, mass, u, species
