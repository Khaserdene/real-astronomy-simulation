"""The headless N-body engine: owns GPU fields and drives the integrator.

The Engine is deliberately GUI-agnostic.  It loads a :class:`~core.state.State`,
steps it forward in time with a leapfrog integrator + direct-N^2 gravity, and can
hand the state back out (for snapshots, checkpoints, or a live viewport).

Taichi must be initialised once before constructing an Engine; use
:func:`init_taichi`.
"""
from __future__ import annotations

import numpy as np
import taichi as ti

from core.state import State
from core.units import G
from core.solvers import gravity
from core.integrators import leapfrog


def init_taichi(arch: str = "cuda") -> None:
    """Initialise the Taichi runtime in double precision.

    ``arch`` is one of ``"cuda"``, ``"vulkan"``, ``"cpu"``.  Double precision
    matters for long-term energy conservation in gravitational dynamics.
    """
    arch_map = {"cuda": ti.cuda, "vulkan": ti.vulkan, "cpu": ti.cpu}
    ti.init(arch=arch_map.get(arch, ti.cuda), default_fp=ti.f64,
            random_seed=0)


class Engine:
    """Direct-N^2 gravitational N-body engine."""

    def __init__(self, softening: float = 0.05):
        self.softening = float(softening)
        self.n = 0
        self.time = 0.0
        self.step_count = 0
        self.seed = 0
        self.meta: dict = {}
        # GPU fields (allocated lazily in load_state).
        self.pos = self.vel = self.acc = self.mass = None
        # Non-dynamical attributes kept on the host (carried through I/O).
        self._ptype = self._ids = None
        self._age = self._temperature = self._metallicity = None

    # ------------------------------------------------------------------ setup
    def load_state(self, state: State) -> None:
        """Upload a State onto the GPU and prime the accelerations."""
        self.n = state.n
        self.time = state.time
        self.step_count = state.step
        self.seed = state.seed
        self.meta = dict(state.meta)
        self._ptype = state.ptype.copy()
        self._ids = state.ids.copy()
        self._age = state.age
        self._temperature = state.temperature
        self._metallicity = state.metallicity

        self.pos = ti.Vector.field(3, ti.f64, shape=self.n)
        self.vel = ti.Vector.field(3, ti.f64, shape=self.n)
        self.acc = ti.Vector.field(3, ti.f64, shape=self.n)
        self.mass = ti.field(ti.f64, shape=self.n)

        self.pos.from_numpy(state.pos)
        self.vel.from_numpy(state.vel)
        self.mass.from_numpy(state.mass)
        self._compute_acc()

    def to_state(self) -> State:
        """Download the current GPU state back into a host State."""
        return State(
            pos=self.pos.to_numpy(),
            vel=self.vel.to_numpy(),
            mass=self.mass.to_numpy(),
            ptype=self._ptype,
            ids=self._ids,
            time=self.time,
            step=self.step_count,
            seed=self.seed,
            age=self._age,
            temperature=self._temperature,
            metallicity=self._metallicity,
            meta=dict(self.meta),
        )

    # --------------------------------------------------------------- stepping
    def _compute_acc(self) -> None:
        gravity.compute_acc_direct(self.pos, self.mass, self.acc, self.n,
                                   G, self.softening ** 2)

    def step(self, dt: float) -> None:
        """Advance the system by one leapfrog (KDK) step of size ``dt``."""
        half = 0.5 * dt
        leapfrog.kick(self.vel, self.acc, self.n, half)
        leapfrog.drift(self.pos, self.vel, self.n, dt)
        self._compute_acc()
        leapfrog.kick(self.vel, self.acc, self.n, half)
        self.time += dt
        self.step_count += 1

    # ------------------------------------------------------------ diagnostics
    def energies(self) -> tuple[float, float, float]:
        """Return (kinetic, potential, total) energy in code units."""
        ke = gravity.kinetic_energy(self.vel, self.mass, self.n)
        pe = gravity.potential_energy(self.pos, self.mass, self.n,
                                      G, self.softening ** 2)
        return ke, pe, ke + pe
