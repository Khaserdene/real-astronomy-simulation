"""Gas-disk engine: SPH hydrodynamics + gas self-gravity + analytic potential.

The gas evolves under three forces: SPH pressure/viscosity (from :mod:`core.solvers.sph`),
its own self-gravity (direct N^2), and a fixed analytic galaxy potential
(Miyamoto-Nagai disk + Plummer halo) standing in for the stars and dark halo.
This is the tractable route to a gas disk that develops spiral structure for a
realistic render; full live star+gas coupling is a later refinement.

No ``from __future__ import annotations`` here: Taichi kernel annotations must be
live objects.
"""
import numpy as np
import taichi as ti

from core.units import G
from core.solvers import sph
from core.sph_engine import _kick, _drift


@ti.kernel
def add_grav_ext(pos: ti.template(), mass: ti.template(), acc: ti.template(),
                 n: ti.i32, eps2: ti.f64, M_d: ti.f64, a: ti.f64, b: ti.f64,
                 M_h: ti.f64, a_h: ti.f64, is_hernquist: ti.i32, smbh_mass: ti.f64):
    """Add gas self-gravity (softened N^2) + analytic disk+halo accel to acc."""
    for i in range(n):
        av = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
        pi = pos[i]
        for j in range(n):
            d = pos[j] - pi
            r2 = d.dot(d) + eps2
            inv = 1.0 / ti.sqrt(r2)
            av += (G * mass[j] * inv * inv * inv) * d
        # Miyamoto-Nagai disk
        R2 = pi.x * pi.x + pi.y * pi.y
        S = ti.sqrt(pi.z * pi.z + b * b)
        aS = a + S
        invD3 = (R2 + aS * aS) ** (-1.5)
        av += ti.Vector([-G * M_d * pi.x * invD3,
                         -G * M_d * pi.y * invD3,
                         -G * M_d * pi.z * aS * invD3 / S], dt=ti.f64)
        if is_hernquist == 1:
            r = pi.norm()
            r_safe = ti.max(r, 1e-6)
            av += -(G * M_h / ((r_safe + a_h) ** 2 * r_safe)) * pi
        else:
            rh = R2 + pi.z * pi.z + a_h * a_h
            av += -(G * M_h * rh ** (-1.5)) * pi
        if smbh_mass > 0.0:
            r_safe = pi.norm() + 1e-6
            av += -(G * smbh_mass / (r_safe * r_safe * r_safe)) * pi
        acc[i] += av


@ti.kernel
def add_analytic(pos: ti.template(), acc: ti.template(), n: ti.i32,
                 M_d: ti.f64, a: ti.f64, b: ti.f64, M_h: ti.f64, a_h: ti.f64,
                 is_hernquist: ti.i32, smbh_mass: ti.f64):
    """Add ONLY the analytic disk+halo potential (no self-gravity).

    Used on the Barnes-Hut path, where self-gravity comes from the tree; a no-op
    when M_d==M_h==0 (the giant-impact case)."""
    for i in range(n):
        pi = pos[i]
        R2 = pi.x * pi.x + pi.y * pi.y
        S = ti.sqrt(pi.z * pi.z + b * b)
        aS = a + S
        invD3 = (R2 + aS * aS) ** (-1.5)
        av = ti.Vector([-G * M_d * pi.x * invD3,
                        -G * M_d * pi.y * invD3,
                        -G * M_d * pi.z * aS * invD3 / S], dt=ti.f64)
        if is_hernquist == 1:
            r = pi.norm()
            r_safe = ti.max(r, 1e-6)
            av += -(G * M_h / ((r_safe + a_h) ** 2 * r_safe)) * pi
        else:
            rh = R2 + pi.z * pi.z + a_h * a_h
            av += -(G * M_h * rh ** (-1.5)) * pi
        if smbh_mass > 0.0:
            r_safe = pi.norm() + 1e-6
            av += -(G * smbh_mass / (r_safe * r_safe * r_safe)) * pi
        acc[i] += av


@ti.kernel
def add_field(acc: ti.template(), src: ti.template(), n: ti.i32):
    """acc += src (fold a separately-computed gravity field in)."""
    for i in range(n):
        acc[i] += src[i]


@ti.kernel
def cool_to_floor(u: ti.template(), n: ti.i32, dt: ti.f64,
                  u_floor: ti.f64, inv_tcool: ti.f64):
    """Radiative cooling: relax internal energy toward a floor.

    Excess thermal energy above ``u_floor`` decays on timescale ``1/inv_tcool``
    via exponential relaxation -- unconditionally stable for any ``dt``, never
    overshoots, and never cools below the floor.  This stands in for a full
    cooling function and is what lets shock/feedback heat radiate away so the gas
    settles into a thin cold disk instead of puffing up and dispersing.
    """
    decay = ti.exp(-dt * inv_tcool)
    for i in range(n):
        excess = u[i] - u_floor
        if excess > 0.0:
            u[i] = u_floor + excess * decay


class GasDiskEngine:
    def __init__(self, pot=None, gamma=5.0 / 3.0, alpha=1.0, beta=2.0,
                 eta=1.3, softening=0.2,
                 cooling=True, u_floor=60.0, t_cool=0.02,
                 gravity_mode="direct", theta=0.6):
        self.pot = pot or dict(M_d=5.0, a=3.0, b=0.3, M_h=12.0, a_h=8.0)
        self.gamma, self.alpha, self.beta, self.eta = gamma, alpha, beta, eta
        self.softening = float(softening)
        self.eps2 = softening ** 2
        self.gravity_mode = gravity_mode      # "direct" (N^2) | "bh" (treecode)
        self.theta = float(theta)
        self._bh = None
        self._grid = None
        # Radiative cooling: relax u toward u_floor on timescale t_cool.
        self.cooling = bool(cooling)
        self.u_floor = float(u_floor)
        self.inv_tcool = 1.0 / float(t_cool) if t_cool > 0 else 0.0
        self.n = 0
        self.time = 0.0
        self.step_count = 0
        self._f = {}

    def setup(self, pos, vel, mass, u):
        n = pos.shape[0]
        self.n = n
        f = {name: ti.Vector.field(3, ti.f64, shape=n)
             for name in ("pos", "vel", "acc", "acc_g")}
        for name in ("mass", "u", "du", "rho", "pressure", "cs", "h", "frozen"):
            f[name] = ti.field(ti.f64, shape=n) if name != "frozen" \
                else ti.field(ti.i32, shape=n)
        self._f = f
        f["pos"].from_numpy(np.ascontiguousarray(pos, np.float64))
        f["vel"].from_numpy(np.ascontiguousarray(vel, np.float64))
        f["mass"].from_numpy(np.ascontiguousarray(mass, np.float64))
        f["u"].from_numpy(np.ascontiguousarray(u, np.float64))
        f["frozen"].from_numpy(np.zeros(n, np.int32))
        span = pos.max(0) - pos.min(0)
        h0 = self.eta * (mass.mean() / (mass.sum() / max(np.prod(span), 1e-6)))
        f["h"].from_numpy(np.full(n, max(h0, 0.05)))
        if self.gravity_mode == "bh" and n >= 256:
            from core.solvers.barnes_hut import BarnesHut
            self._bh = BarnesHut(n, theta=self.theta, softening=self.softening)
        else:
            self._bh = None
        if n >= 2000:
            from core.solvers.neighbor_grid import NeighborGrid
            self._grid = NeighborGrid(n)
        else:
            self._grid = None
        self._density_h(iters=15)
        self._forces()

    def _gas_density(self):
        f = self._f
        ng = self._grid.rebuild_all(f["pos"], f["h"]) if self._grid is not None else 0
        if ng > 0:
            g = self._grid
            sph.compute_density_grid(f["pos"], f["mass"], f["rho"], f["h"], self.n,
                                     g.gsort, g.cell_start, g.nx, g.ny, g.nz,
                                     g.lo[0], g.lo[1], g.lo[2], g.inv_cell)
        else:
            sph.compute_density(f["pos"], f["mass"], f["rho"], f["h"], self.n,
                                0.0, 0.0)

    def _density_h(self, iters=3):
        f = self._f
        for _ in range(iters):
            self._gas_density()
            sph.update_h(f["rho"], f["mass"], f["h"], self.n, self.eta)
        self._gas_density()

    def _forces(self):
        f = self._f
        sph.compute_pressure(f["rho"], f["u"], f["pressure"], f["cs"],
                             self.n, self.gamma)
        if self._grid is not None and self._grid.ngas > 0:
            g = self._grid
            sph.compute_hydro_forces_grid(f["pos"], f["vel"], f["mass"], f["rho"],
                                          f["pressure"], f["cs"], f["h"], f["acc"],
                                          f["du"], self.n, self.alpha, self.beta,
                                          g.gsort, g.cell_start, g.nx, g.ny, g.nz,
                                          g.lo[0], g.lo[1], g.lo[2], g.inv_cell)
        else:
            sph.compute_hydro_forces(f["pos"], f["vel"], f["mass"], f["rho"],
                                     f["pressure"], f["cs"], f["h"], f["acc"],
                                     f["du"], self.n, self.alpha, self.beta, 0.0, 0.0)
        if self._bh is not None:
            # Self-gravity via the treecode; analytic potential added separately.
            self._bh.compute(f["pos"], f["mass"], f["acc_g"])
            add_field(f["acc"], f["acc_g"], self.n)
            add_analytic(f["pos"], f["acc"], self.n,
                         self.pot["M_d"], self.pot["a"], self.pot["b"],
                         self.pot["M_h"], self.pot["a_h"],
                         int(self.pot.get("is_hernquist", 0)),
                         float(self.pot.get("smbh_mass", 0.0)))
        else:
            add_grav_ext(f["pos"], f["mass"], f["acc"], self.n, self.eps2,
                         self.pot["M_d"], self.pot["a"], self.pot["b"],
                         self.pot["M_h"], self.pot["a_h"],
                         int(self.pot.get("is_hernquist", 0)),
                         float(self.pot.get("smbh_mass", 0.0)))

    def step(self, dt):
        f = self._f
        half = 0.5 * dt
        _kick(f["vel"], f["acc"], f["u"], f["du"], f["frozen"], self.n, half)
        _drift(f["pos"], f["vel"], f["frozen"], self.n, dt)
        self._density_h()
        self._forces()
        _kick(f["vel"], f["acc"], f["u"], f["du"], f["frozen"], self.n, half)
        if self.cooling and self.inv_tcool > 0.0:
            cool_to_floor(f["u"], self.n, dt, self.u_floor, self.inv_tcool)
        self.time += dt
        self.step_count += 1

    def get(self, name):
        return self._f[name].to_numpy()

    def to_state(self):
        """Export the current gas state as a core.State (ptype = gas)."""
        from core.state import State, PTYPE_GAS
        n = self.n
        return State(
            pos=self.get("pos"), vel=self.get("vel"), mass=self.get("mass"),
            ptype=np.full(n, PTYPE_GAS, np.int32), ids=np.arange(n, dtype=np.int64),
            time=self.time, step=self.step_count,
            u=self.get("u"), rho=self.get("rho"),
            meta={"engine": "gas_disk"},
        )
