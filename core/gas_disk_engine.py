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
                 M_h: ti.f64, a_h: ti.f64):
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
        # Plummer halo
        rh = R2 + pi.z * pi.z + a_h * a_h
        av += -(G * M_h * rh ** (-1.5)) * pi
        acc[i] += av


class GasDiskEngine:
    def __init__(self, pot=None, gamma=5.0 / 3.0, alpha=1.0, beta=2.0,
                 eta=1.3, softening=0.2):
        self.pot = pot or dict(M_d=5.0, a=3.0, b=0.3, M_h=12.0, a_h=8.0)
        self.gamma, self.alpha, self.beta, self.eta = gamma, alpha, beta, eta
        self.eps2 = softening ** 2
        self.n = 0
        self.time = 0.0
        self.step_count = 0
        self._f = {}

    def setup(self, pos, vel, mass, u):
        n = pos.shape[0]
        self.n = n
        f = {name: ti.Vector.field(3, ti.f64, shape=n)
             for name in ("pos", "vel", "acc")}
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
        self._density_h(iters=15)
        self._forces()

    def _density_h(self, iters=3):
        f = self._f
        for _ in range(iters):
            sph.compute_density(f["pos"], f["mass"], f["rho"], f["h"], self.n,
                                0.0, 0.0)
            sph.update_h(f["rho"], f["mass"], f["h"], self.n, self.eta)
        sph.compute_density(f["pos"], f["mass"], f["rho"], f["h"], self.n,
                            0.0, 0.0)

    def _forces(self):
        f = self._f
        sph.compute_pressure(f["rho"], f["u"], f["pressure"], f["cs"],
                             self.n, self.gamma)
        sph.compute_hydro_forces(f["pos"], f["vel"], f["mass"], f["rho"],
                                 f["pressure"], f["cs"], f["h"], f["acc"],
                                 f["du"], self.n, self.alpha, self.beta, 0.0, 0.0)
        add_grav_ext(f["pos"], f["mass"], f["acc"], self.n, self.eps2,
                     self.pot["M_d"], self.pot["a"], self.pot["b"],
                     self.pot["M_h"], self.pot["a_h"])

    def step(self, dt):
        f = self._f
        half = 0.5 * dt
        _kick(f["vel"], f["acc"], f["u"], f["du"], f["frozen"], self.n, half)
        _drift(f["pos"], f["vel"], f["frozen"], self.n, dt)
        self._density_h()
        self._forces()
        _kick(f["vel"], f["acc"], f["u"], f["du"], f["frozen"], self.n, half)
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
