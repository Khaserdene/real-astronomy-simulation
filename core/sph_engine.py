"""Self-contained SPH gas engine (hydro only; gravity coupling comes later).

Drives the SPH kernels with an adaptive smoothing length and a leapfrog (KDK)
integrator that also advances the internal energy.  Particles flagged ``frozen``
are held fixed -- used as reservoir/wall boundaries for the shock-tube test.

NOTE: deliberately *no* ``from __future__ import annotations`` here -- Taichi reads
kernel argument annotations as live objects (``ti.template()``); PEP-563 would turn
them into strings and break kernel compilation.
"""
import numpy as np
import taichi as ti

from core.solvers import sph


@ti.kernel
def _kick(vel: ti.template(), acc: ti.template(), u: ti.template(),
          du: ti.template(), frozen: ti.template(), n: ti.i32, dt_half: ti.f64):
    for i in range(n):
        if frozen[i] == 0:
            vel[i] += acc[i] * dt_half
            u[i] += du[i] * dt_half
            if u[i] < 1e-8:        # keep internal energy positive
                u[i] = 1e-8


@ti.kernel
def _drift(pos: ti.template(), vel: ti.template(), frozen: ti.template(),
           n: ti.i32, dt: ti.f64):
    for i in range(n):
        if frozen[i] == 0:
            pos[i] += vel[i] * dt


@ti.kernel
def _thermal_kinetic(vel: ti.template(), mass: ti.template(),
                     u: ti.template(), n: ti.i32) -> ti.f64:
    tot = 0.0
    for i in range(n):
        tot += 0.5 * mass[i] * vel[i].dot(vel[i]) + mass[i] * u[i]
    return tot


class SPHEngine:
    """Gas-only SPH engine with adaptive smoothing length."""

    def __init__(self, gamma=1.4, alpha=1.0, beta=2.0, eta=1.3,
                 periodic_y=0.0, periodic_z=0.0):
        self.gamma, self.alpha, self.beta, self.eta = gamma, alpha, beta, eta
        self.ly, self.lz = periodic_y, periodic_z   # 0 = non-periodic
        self.n = 0
        self.time = 0.0
        self.step_count = 0
        self._fields = {}

    def setup(self, pos, vel, mass, u, frozen=None):
        n = pos.shape[0]
        self.n = n
        f = {name: ti.Vector.field(3, ti.f64, shape=n)
             for name in ("pos", "vel", "acc")}
        for name in ("mass", "u", "du", "rho", "pressure", "cs", "h"):
            f[name] = ti.field(ti.f64, shape=n)
        f["frozen"] = ti.field(ti.i32, shape=n)
        self._fields = f

        f["pos"].from_numpy(np.ascontiguousarray(pos, np.float64))
        f["vel"].from_numpy(np.ascontiguousarray(vel, np.float64))
        f["mass"].from_numpy(np.ascontiguousarray(mass, np.float64))
        f["u"].from_numpy(np.ascontiguousarray(u, np.float64))
        f["frozen"].from_numpy(np.zeros(n, np.int32) if frozen is None
                               else np.ascontiguousarray(frozen, np.int32))
        # Initial smoothing length from a rough mean spacing, then converge it.
        mean_h = self.eta * (mass.mean() / np.median(_estimate_rho0(pos, mass)))
        f["h"].from_numpy(np.full(n, max(mean_h, 1e-3)))
        self._update_density_h(iters=15)   # converge adaptive h from scratch
        self._update_forces()

    # ------------------------------------------------------------- internals
    def _update_density_h(self, iters: int = 3):
        f = self._fields
        for _ in range(iters):
            sph.compute_density(f["pos"], f["mass"], f["rho"], f["h"], self.n,
                                self.ly, self.lz)
            sph.update_h(f["rho"], f["mass"], f["h"], self.n, self.eta)
        sph.compute_density(f["pos"], f["mass"], f["rho"], f["h"], self.n,
                            self.ly, self.lz)

    def _update_forces(self):
        f = self._fields
        sph.compute_pressure(f["rho"], f["u"], f["pressure"], f["cs"],
                             self.n, self.gamma)
        sph.compute_hydro_forces(f["pos"], f["vel"], f["mass"], f["rho"],
                                 f["pressure"], f["cs"], f["h"], f["acc"],
                                 f["du"], self.n, self.alpha, self.beta,
                                 self.ly, self.lz)

    # ------------------------------------------------------------------ step
    def step(self, dt: float):
        f = self._fields
        half = 0.5 * dt
        _kick(f["vel"], f["acc"], f["u"], f["du"], f["frozen"], self.n, half)
        _drift(f["pos"], f["vel"], f["frozen"], self.n, dt)
        self._update_density_h()
        self._update_forces()
        _kick(f["vel"], f["acc"], f["u"], f["du"], f["frozen"], self.n, half)
        self.time += dt
        self.step_count += 1

    # ------------------------------------------------------------ accessors
    def total_energy(self) -> float:
        f = self._fields
        return _thermal_kinetic(f["vel"], f["mass"], f["u"], self.n)

    def get(self, name: str) -> np.ndarray:
        return self._fields[name].to_numpy()


def _estimate_rho0(pos, mass):
    """Crude nearest-neighbour density estimate for the initial h guess."""
    # Use the global mean number density as a stand-in (cheap, only for h0).
    span = pos.max(axis=0) - pos.min(axis=0)
    vol = max(np.prod(np.where(span > 0, span, 1.0)), 1e-9)
    return np.full(pos.shape[0], mass.sum() / vol)
