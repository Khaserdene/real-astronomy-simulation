"""Editor state machine -- all GUI logic, no GUI dependency.

The controller owns the current :class:`~core.simulation.Simulation` and the
editor's settings (scene, level, run/pause state, snapshot cadence, output
folder).  Keeping it free of any Taichi-UI code means the entire editor workflow
-- build, run, pause, checkpoint, resume, promote-to-production, export -- can be
unit-tested headlessly; the GGUI app is just a thin view on top.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.scene import Scene
from core.simulation import Simulation
from core.state import State, PTYPE_STAR
from export.to_pointcloud import snapshot_to_ply, synth_temperature
from export.blackbody import blackbody_rgb

LEVELS = ["preview", "low", "medium", "high", "ultra"]


@dataclass
class EditorSettings:
    ic: str = "disk"
    seed: int = 7
    level: str = "low"
    running: bool = False
    auto_snapshot: bool = False
    snap_every: int = 20
    out_dir: str = "output/editor_run"
    point_scale: float = 1.0          # display zoom (kpc -> view units)


class EditorController:
    """Drives a Simulation in response to editor commands."""

    def __init__(self):
        self.s = EditorSettings()
        self.sim: Optional[Simulation] = None
        self._frame = 0          # next snapshot index
        self._steps_since_snap = 0
        self.status = "no simulation -- press Build"

    # --------------------------------------------------------------- building
    def build(self) -> str:
        """Create a fresh simulation from the current scene + level settings."""
        scene = Scene(ic=self.s.ic, seed=int(self.s.seed))
        self.sim = Simulation.from_scene(scene, self.s.level)
        self.s.running = False
        self._frame = 0
        self._steps_since_snap = 0
        self.status = (f"built {self.s.ic} @ {self.sim.level.name} "
                       f"(N={self.sim.engine.n:,}, {self.sim.level.solver})")
        return self.status

    def promote(self, level: str = "high") -> str:
        """Re-build the *same* scene (same seed) at a higher level."""
        self.s.level = level
        return self.build()

    def reset(self) -> str:
        self.sim = None
        self.s.running = False
        self.status = "reset -- press Build"
        return self.status

    # -------------------------------------------------------------- stepping
    def toggle_run(self) -> None:
        if self.sim is not None:
            self.s.running = not self.s.running

    def advance(self, max_steps: int = 1) -> None:
        """Advance the sim if running (called once per UI frame)."""
        if self.sim is None or not self.s.running:
            return
        for _ in range(max_steps):
            self.sim.engine.step(self.sim.dt)
            self._steps_since_snap += 1
            if self.s.auto_snapshot and self._steps_since_snap >= self.s.snap_every:
                self._steps_since_snap = 0
                self.save_snapshot()

    def step_once(self, n: int = 1) -> None:
        if self.sim is None:
            return
        for _ in range(n):
            self.sim.engine.step(self.sim.dt)

    # ----------------------------------------------------------- checkpoints
    def save_snapshot(self) -> str:
        if self.sim is None:
            return "nothing to save"
        os.makedirs(self.s.out_dir, exist_ok=True)
        state = self.sim.engine.to_state()
        state.meta.update(self.sim.meta)
        path = os.path.join(self.s.out_dir, f"snap_{self._frame:04d}.h5")
        state.save(path)
        self._frame += 1
        self.status = f"saved {os.path.basename(path)}"
        return path

    def load_checkpoint(self, path: str) -> str:
        self.sim = Simulation.from_checkpoint(path)
        self.s.level = self.sim.level.name
        self.s.running = False
        self.status = (f"resumed {os.path.basename(path)} "
                       f"(step {self.sim.engine.step_count})")
        return self.status

    def export_plys(self) -> str:
        """Convert every snapshot in the output folder to a coloured PLY."""
        from export.to_pointcloud import batch
        if not os.path.isdir(self.s.out_dir):
            return "no snapshots to export"
        batch(self.s.out_dir)
        self.status = f"exported PLYs in {self.s.out_dir}/ply"
        return self.status

    # ----------------------------------------------------------- display data
    def display_arrays(self, clip_kpc: float = 120.0):
        """Return (positions[N,3] float32, colors[N,3] float32) for the viewport.

        Renders star particles coloured by blackbody temperature.  A few halo
        particles can scatter to large radii in coarse runs; positions beyond
        ``clip_kpc`` are dropped so they don't wreck the camera framing.
        """
        if self.sim is None:
            empty = np.zeros((0, 3), np.float32)
            return empty, empty
        state = self.sim.engine.to_state()
        stars = state.ptype == PTYPE_STAR
        if not stars.any():
            stars = np.ones(state.n, bool)
        keep = stars & (np.linalg.norm(state.pos, axis=1) < clip_kpc)
        pos = (state.pos[keep] * self.s.point_scale).astype(np.float32)
        temp = (state.temperature[keep] if state.temperature is not None
                else synth_temperature(state.pos[keep], seed=state.seed))
        col = blackbody_rgb(temp).astype(np.float32)
        return pos, col

    def stats(self) -> dict:
        if self.sim is None:
            return {"step": 0, "time": 0.0, "n": 0, "energy": None}
        e = None
        if hasattr(self.sim.engine, "energies"):
            e = self.sim.engine.energies()[2]
        return {"step": self.sim.engine.step_count, "time": self.sim.engine.time,
                "n": self.sim.engine.n, "energy": e,
                "level": self.sim.level.name, "solver": self.sim.level.solver}
