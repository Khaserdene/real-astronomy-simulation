"""Gas-disk initial conditions for the SPH gas engine.

An exponential gas disk on near-circular orbits in a fixed analytic galaxy
potential (Miyamoto-Nagai disk + Plummer halo), with a small thermal energy
(warm ISM).  Cold, rotating gas in a disk potential develops spiral structure --
exactly the feature that makes a rendered galaxy read as real.

Returned arrays are in code units (kpc, km/s, 1e10 Msun); internal energy ``u`` is
in (km/s)^2.
"""
import numpy as np

from core.units import G


def make_gas_disk(n=12000, gas_mass=1.0, disk_scale=3.5, disk_height=0.2,
                  sound_speed=12.0, pot=None, seed=0):
    """Sample a rotating exponential gas disk.

    pot : dict of analytic-potential params (disk_mass M_d, disk_a, disk_b,
          halo_mass M_h, halo_a) the gas orbits in.  Defaults to a Milky-Way-ish
          disk + halo.
    sound_speed : isothermal-ish sound speed [km/s] -> sets the thermal energy.
    """
    if pot is None:
        pot = dict(M_d=5.0, a=3.0, b=0.3, M_h=12.0, a_h=8.0)
    rng = np.random.default_rng(seed)

    # Radial exponential, thin vertical, random azimuth.
    u_ = rng.random(n)
    x = np.linspace(0.0, 20.0, 100000)
    cdf = 1.0 - (1.0 + x) * np.exp(-x)
    R = np.interp(u_, cdf, x) * disk_scale
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    z = rng.exponential(disk_height, n) * rng.choice((-1.0, 1.0), n)
    pos = np.column_stack((R * np.cos(phi), R * np.sin(phi), z))

    # Circular speed from the analytic in-plane radial acceleration.
    aR = _radial_accel(R, pot)
    v_c = np.sqrt(np.maximum(R * aR, 0.0))
    e_phi = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros(n)))
    vel = e_phi * v_c[:, None]
    vel += rng.normal(0.0, 0.04 * v_c[:, None], size=(n, 3))  # small dispersion

    mass = np.full(n, gas_mass / n)
    gamma = 5.0 / 3.0
    u = np.full(n, sound_speed ** 2 / (gamma * (gamma - 1.0)))
    return pos, vel, mass, u


def _radial_accel(R, pot):
    """Inward in-plane radial acceleration of the analytic potential at z=0."""
    ad = pot["a"] + pot["b"]
    a_disk = G * pot["M_d"] * R / (R ** 2 + ad ** 2) ** 1.5
    a_halo = G * pot["M_h"] * R / (R ** 2 + pot["a_h"] ** 2) ** 1.5
    return a_disk + a_halo
