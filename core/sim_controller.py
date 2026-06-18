"""Unified backend controller for the GUI -- runs any scenario, no GUI code.

This is the single "brain" the PyQt app talks to.  It builds any scenario through
:mod:`core.scenarios`, advances it, exposes type-aware display points/colours
(shared with snapshot playback), and handles snapshots/checkpoints.  Keeping it
GUI-free means the whole workflow stays unit-testable headlessly.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

from core.scenarios import SCENARIOS, build_scenario, resume_scenario
from core.state import State
from core.sim_display import state_to_display


@dataclass
class Settings:
    scenario: str = "disk"
    seed: int = 7
    n: int = 20000
    running: bool = False
    auto_snapshot: bool = False
    snap_every: int = 25
    out_dir: str = "output/gui_run"


class SimController:
    def __init__(self):
        self.s = Settings()
        self.engine = None
        self.dt = 1e-4
        self._frame = 0
        self._since_snap = 0
        self.status = "no simulation -- press Build"

    # --------------------------------------------------------------- building
    def build(self, overrides: dict | None = None) -> str:
        self.engine, self.dt = build_scenario(
            self.s.scenario, n=self.s.n, seed=self.s.seed, overrides=overrides)
        self.s.running = False
        self._frame = 0
        self._since_snap = 0
        self.status = (f"built '{self.s.scenario}' "
                       f"(N={self.engine.n:,}, dt={self.dt:g})")
        return self.status

    def reset(self):
        self.engine = None
        self.s.running = False
        self.status = "reset -- press Build"

    # -------------------------------------------------------------- stepping
    def toggle_run(self):
        if self.engine is not None:
            self.s.running = not self.s.running

    def advance(self, max_steps: int = 1):
        if self.engine is None or not self.s.running:
            return
        for _ in range(max_steps):
            self.engine.step(self.dt)
            self._since_snap += 1
            if self.s.auto_snapshot and self._since_snap >= self.s.snap_every:
                self._since_snap = 0
                self.save_snapshot()

    def step_once(self, n: int = 1):
        if self.engine is not None:
            for _ in range(n):
                self.engine.step(self.dt)

    # ----------------------------------------------------------- checkpoints
    def save_snapshot(self) -> str:
        if self.engine is None:
            return "nothing to save"
        os.makedirs(self.s.out_dir, exist_ok=True)
        state = self.engine.to_state()
        state.meta.update({"scenario": self.s.scenario, "dt": self.dt,
                           "seed": self.s.seed})
        path = os.path.join(self.s.out_dir, f"snap_{self._frame:04d}.h5")
        state.save(path)
        self._frame += 1
        self.status = f"saved {os.path.basename(path)}"
        return path

    def load_checkpoint(self, path: str) -> str:
        state = State.load(path)
        scenario = str(state.meta.get("scenario", self.s.scenario))
        self.engine, self.dt = resume_scenario(scenario, state)
        self.s.scenario = scenario
        self.s.running = False
        self.status = (f"resumed {os.path.basename(path)} "
                       f"(step {self.engine.step_count})")
        return self.status

    def list_snapshots(self, folder: str | None = None):
        folder = folder or self.s.out_dir
        return sorted(glob.glob(os.path.join(folder, "snap_*.h5")))

    # ----------------------------------------------------------- display data
    def display_arrays(self, clip_kpc: float = 120.0):
        if self.engine is None:
            return np.zeros((0, 3), np.float32), np.zeros((0, 4), np.float32)
        return state_to_display(self.engine.to_state(), clip_kpc)

    @staticmethod
    def display_snapshot(path: str, clip_kpc: float = 120.0):
        """Load a saved snapshot and return its display arrays (for playback)."""
        return state_to_display(State.load(path), clip_kpc)

    def stats(self) -> dict:
        if self.engine is None:
            return {"step": 0, "time": 0.0, "n": 0, "energy": None}
        e = None
        if hasattr(self.engine, "energies"):
            e = self.engine.energies()[2]
        extra = {}
        if hasattr(self.engine, "star_count"):
            extra["stars"] = self.engine.star_count()
        return {"step": self.engine.step_count, "time": self.engine.time,
                "n": self.engine.n, "energy": e,
                "scenario": self.s.scenario, **extra}
