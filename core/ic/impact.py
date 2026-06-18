"""Giant-impact initial conditions: two self-gravitating bodies that collide.

Two uniform spheres of SPH particles are put on a colliding orbit with an impact
parameter (an off-centre hit, like the Moon-forming impact).  Run with SPH +
self-gravity (no external potential) the bodies deform on contact, shock-heat,
throw off a debris disc, and partially re-merge.

These are dimensionally generic self-gravitating bodies in the code unit system
(kpc, km/s, 1e10 Msun) -- scaled for numerically convenient sizes rather than
literal planet values, but the collision physics is the same.
"""
import numpy as np

from core.units import G


def make_impact(n_each=6000, body_mass=0.02, body_radius=1.2,
                separation=6.0, impact_param=1.0, v_approach=45.0,
                cs_frac=0.7, seed=0):
    """Two uniform self-gravitating spheres on a colliding orbit.

    Returns (pos, vel, mass, u) in code units; ``u`` is internal energy [(km/s)^2].
    """
    rng = np.random.default_rng(seed)
    p1 = _uniform_sphere(rng, n_each, body_radius)
    p2 = _uniform_sphere(rng, n_each, body_radius)

    p1 = p1 + np.array([-separation / 2, -impact_param / 2, 0.0])
    p2 = p2 + np.array([+separation / 2, +impact_param / 2, 0.0])
    pos = np.vstack((p1, p2))

    v1 = np.tile([+v_approach / 2, 0.0, 0.0], (n_each, 1))
    v2 = np.tile([-v_approach / 2, 0.0, 0.0], (n_each, 1))
    vel = np.vstack((v1, v2))

    mass = np.full(2 * n_each, body_mass / n_each)

    # Internal energy for rough pressure support: c_s ~ cs_frac * sqrt(G M / R).
    gamma = 5.0 / 3.0
    v_vir = np.sqrt(G * body_mass / body_radius)
    cs = cs_frac * v_vir
    u = np.full(2 * n_each, cs ** 2 / (gamma * (gamma - 1.0)))
    return pos, vel, mass, u


def make_body(n=6000, body_mass=0.02, body_radius=1.2, cs_frac=0.7, seed=0):
    """One self-gravitating uniform SPH sphere at the origin, at rest.

    Returns (pos, vel, mass, u) in code units -- the single-object building block
    the scene composer places (e.g. two of them = a giant impact).
    """
    rng = np.random.default_rng(seed)
    pos = _uniform_sphere(rng, n, body_radius)
    vel = np.zeros_like(pos)
    mass = np.full(n, body_mass / n)
    gamma = 5.0 / 3.0
    cs = cs_frac * np.sqrt(G * body_mass / body_radius)
    u = np.full(n, cs ** 2 / (gamma * (gamma - 1.0)))
    return pos, vel, mass, u


def _uniform_sphere(rng, n, radius):
    r = radius * rng.random(n) ** (1.0 / 3.0)
    cos_t = rng.uniform(-1.0, 1.0, n)
    sin_t = np.sqrt(1.0 - cos_t ** 2)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.column_stack((r * sin_t * np.cos(phi),
                            r * sin_t * np.sin(phi), r * cos_t))
