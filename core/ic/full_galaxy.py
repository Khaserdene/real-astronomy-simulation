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


def make_cosmo_baryon(n: int = 30000, seed: int = 0, gas_fraction: float = 0.16,
                      sound_speed: float = 22.0, box_size: float = 24.0,
                      amplitude: float = 1.4, hubble: float = 42.0):
    """Cosmological box with baryons: DM + gas that cools and forms stars.

    The "living universe" -- a Zel'dovich-perturbed box (cosmic web) where a
    fraction of the matter is gas; run with cooling + star formation, the gas
    drains into the filament knots, cools and lights up as stars.
    Returns ``(pos, vel, mass, u, species)`` (0 gas, 1 star, 2 DM).
    """
    from core.ic.cosmo import make_cosmo_box

    n_side = max(int(round(n ** (1.0 / 3.0))), 4)
    st = make_cosmo_box(n_side=n_side, box_size=box_size, total_mass=120.0,
                        amplitude=amplitude, n_modes=14, hubble=hubble,
                        vel_factor=18.0, seed=seed)
    m = st.n
    rng = np.random.default_rng(seed + 7)
    is_gas = rng.random(m) < gas_fraction
    species = np.where(is_gas, 0, 2).astype(np.int32)
    gamma = 5.0 / 3.0
    u = np.full(m, sound_speed ** 2 / (gamma * (gamma - 1.0)))
    return st.pos, st.vel, st.mass, u, species


def make_protogalaxy(n: int = 30000, seed: int = 0, gas_fraction: float = 0.4,
                     halo_mass: float = 14.0, halo_scale: float = 8.0,
                     gas_mass: float = 4.0, gas_radius: float = 12.0,
                     spin: float = 0.18, sound_speed: float = 32.0):
    """A forming galaxy: a live DM halo + a warm, slowly-rotating gas cloud.

    With cooling on, the gas radiates energy, collapses along its spin axis into a
    rotationally-supported disk, and converts to stars -- so the composition
    evolves from gas-dominated to star-dominated (galaxy formation/evolution).
    Returns ``(pos, vel, mass, u, species)`` (0 gas, 1 star, 2 DM).
    """
    from core.ic.plummer import make_plummer, _random_directions

    n_gas = max(int(n * gas_fraction), 1)
    n_dm = max(n - n_gas, 1)

    dm = make_plummer(n=n_dm, total_mass=halo_mass, scale_radius=halo_scale,
                      seed=seed)

    # Warm gas cloud: Plummer-sampled sphere, slow solid-body rotation about z.
    rng = np.random.default_rng(seed + 3)
    a = gas_radius / 2.0
    x1 = rng.random(n_gas)
    r = np.clip(a / np.sqrt(x1 ** (-2.0 / 3.0) - 1.0), 0.0, gas_radius * 2.0)
    gpos = _random_directions(rng, n_gas) * r[:, None]
    gvel = spin * np.column_stack((-gpos[:, 1], gpos[:, 0], np.zeros(n_gas)))
    gmass = np.full(n_gas, gas_mass / n_gas)
    gamma = 5.0 / 3.0
    gu = np.full(n_gas, sound_speed ** 2 / (gamma * (gamma - 1.0)))

    pos = np.vstack((gpos, dm.pos))
    vel = np.vstack((gvel, dm.vel))
    mass = np.concatenate((gmass, dm.mass))
    u = np.concatenate((gu, np.full(n_dm, 200.0)))
    species = np.concatenate((np.zeros(n_gas, np.int32),
                              np.full(n_dm, 2, np.int32)))
    vel -= np.average(vel, axis=0, weights=mass)
    return pos, vel, mass, u, species


def make_galaxy_merger(n: int = 40000, seed: int = 0,
                       separation: float = 60.0, v_approach: float = 120.0,
                       impact_param: float = 15.0, inclination: float = 30.0):
    """Two full galaxies (stars + live DM + gas) on a collision course.

    The richest merger: both galaxies are live multi-component systems, so the
    encounter produces tidal tails *and* gas shocks / triggered star formation.
    Returns ``(pos, vel, mass, u, species)`` (species 0 gas, 1 star, 2 DM).
    """
    half = max(n // 2, 1)
    p1, v1, m1, u1, s1 = make_full_galaxy(n=half, seed=seed)
    p2, v2, m2, u2, s2 = make_full_galaxy(n=n - half, seed=seed + 101)

    # Incline the second galaxy's disk about the x-axis (the approach axis).
    th = np.radians(inclination)
    c, s = np.cos(th), np.sin(th)
    rot = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    p2 = p2 @ rot.T
    v2 = v2 @ rot.T

    # Place on an approaching orbit with an impact parameter (offset in y).
    p1 = p1 + np.array([-separation / 2, -impact_param / 2, 0.0])
    p2 = p2 + np.array([+separation / 2, +impact_param / 2, 0.0])
    v1 = v1 + np.array([+v_approach / 2, 0.0, 0.0])
    v2 = v2 + np.array([-v_approach / 2, 0.0, 0.0])

    pos = np.vstack((p1, p2))
    vel = np.vstack((v1, v2))
    mass = np.concatenate((m1, m2))
    u = np.concatenate((u1, u2))
    species = np.concatenate((s1, s2))
    vel -= np.average(vel, axis=0, weights=mass)
    return pos, vel, mass, u, species
