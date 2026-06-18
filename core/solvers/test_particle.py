"""Test-particle (restricted) dynamics in a fixed analytic potential.

This is the fastest "Preview" tier (Toomre & Toomre 1972 style): massless tracer
particles move in a frozen analytic galaxy potential (Miyamoto-Nagai disk +
Plummer halo).  There is no self-gravity, so it is *not* physically converged --
but it captures the gross morphology (disk shape, tidal tails in a merger) almost
instantly, which is exactly what a preview is for.

Acceleration of a Miyamoto-Nagai disk (mass M, scale a, b):
    S = sqrt(z^2 + b^2),  D = sqrt(R^2 + (a + S)^2)
    a_xy = -G M r_xy / D^3
    a_z  = -G M z (a + S) / (D^3 S)
plus a Plummer halo  a = -G Mh r / (r^2 + ah^2)^{3/2}.
"""
from __future__ import annotations

import numpy as np
import taichi as ti

from core.state import State
from core.units import G
from core.integrators import leapfrog


@ti.data_oriented
class TestParticleEngine:
    """Integrate massless tracers in a fixed analytic potential."""

    def __init__(self, disk_mass=5.0, disk_a=3.0, disk_b=0.3,
                 halo_mass=50.0, halo_a=20.0):
        # Stored as plain floats; Taichi captures them as compile-time constants
        # inside the kernel (they are fixed for the life of the run).
        self.M_d = float(disk_mass)
        self.a = float(disk_a)
        self.b = float(disk_b)
        self.M_h = float(halo_mass)
        self.a_h = float(halo_a)
        self.n = 0
        self.time = 0.0
        self.step_count = 0
        self.seed = 0
        self.meta = {}
        self.pos = self.vel = self.acc = None
        self._ptype = self._ids = None

    def load_state(self, state: State) -> None:
        self.n = state.n
        self.time = state.time
        self.step_count = state.step
        self.seed = state.seed
        self.meta = dict(state.meta)
        self._ptype = state.ptype.copy()
        self._ids = state.ids.copy()
        self.pos = ti.Vector.field(3, ti.f64, shape=self.n)
        self.vel = ti.Vector.field(3, ti.f64, shape=self.n)
        self.acc = ti.Vector.field(3, ti.f64, shape=self.n)
        self.pos.from_numpy(state.pos)
        self.vel.from_numpy(state.vel)
        self._compute_acc()

    def to_state(self) -> State:
        n = self.n
        return State(
            pos=self.pos.to_numpy(), vel=self.vel.to_numpy(),
            mass=np.zeros(n), ptype=self._ptype, ids=self._ids,
            time=self.time, step=self.step_count, seed=self.seed,
            meta=dict(self.meta),
        )

    @ti.kernel
    def _accel(self):
        M_d, a, b = self.M_d, self.a, self.b
        M_h, a_h = self.M_h, self.a_h
        for i in range(self.n):
            p = self.pos[i]
            R2 = p.x * p.x + p.y * p.y
            S = ti.sqrt(p.z * p.z + b * b)
            aS = a + S
            D2 = R2 + aS * aS
            inv_D3 = D2 ** (-1.5)
            ax = -G * M_d * p.x * inv_D3
            ay = -G * M_d * p.y * inv_D3
            az = -G * M_d * p.z * aS * inv_D3 / S
            # Plummer halo (spherical)
            r2 = R2 + p.z * p.z + a_h * a_h
            inv_h3 = r2 ** (-1.5)
            self.acc[i] = ti.Vector([ax, ay, az], dt=ti.f64) \
                - (G * M_h * inv_h3) * p

    def _compute_acc(self) -> None:
        self._accel()

    def step(self, dt: float) -> None:
        # Reuse the shared leapfrog kernels (free functions accept scalar args;
        # data_oriented *method* kernels in Taichi 1.7 do not).
        leapfrog.kick(self.vel, self.acc, self.n, 0.5 * dt)
        leapfrog.drift(self.pos, self.vel, self.n, dt)
        self._compute_acc()
        leapfrog.kick(self.vel, self.acc, self.n, 0.5 * dt)
        self.time += dt
        self.step_count += 1
