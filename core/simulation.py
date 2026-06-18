"""High-level Simulation driver: ties Scene + level + engine + checkpoints.

This is the single orchestration point used by both the CLI and (later) the GUI.
It builds the right solver for a level, runs the integration loop while writing
frame-by-frame snapshots, and can resume from any snapshot -- on the same or a
different machine -- because every snapshot is a full, self-describing checkpoint.
"""
from __future__ import annotations

import os
import time
from typing import Optional

from core.config import LevelConfig, get_level, detect_hardware, check_fit
from core.scene import Scene, build_state
from core.state import State
from core.engine import Engine


def _make_engine(solver: str, softening: float, params: dict):
    """Construct the solver engine matching a level's solver choice."""
    if solver == "direct":
        return Engine(softening=softening)
    if solver == "test_particle":
        from core.solvers.test_particle import TestParticleEngine
        return TestParticleEngine(
            disk_mass=params.get("disk_mass", 5.0),
            disk_a=params.get("disk_scale", 3.0),
            disk_b=params.get("disk_height", 0.3),
            halo_mass=params.get("halo_mass", 50.0),
            halo_a=params.get("halo_scale", 20.0),
        )
    raise ValueError(f"solver '{solver}' not implemented yet")


class Simulation:
    """Owns an engine + its run/checkpoint metadata."""

    def __init__(self, engine, level: LevelConfig, dt: float, meta: dict):
        self.engine = engine
        self.level = level
        self.dt = dt
        self.meta = meta

    # ----------------------------------------------------------- constructors
    @classmethod
    def from_scene(cls, scene: Scene, level_name: str = "medium",
                   overrides: Optional[dict] = None) -> "Simulation":
        level = get_level(level_name, overrides)
        hw = detect_hardware()
        check_fit(level, hw)

        state = build_state(scene, level.n, level.solver)
        params = scene.resolved_params()
        # Stamp everything needed to resume this exact run later.
        state.meta.update({
            "level": level.name, "solver": level.solver,
            "softening": level.softening, "dt": level.dt,
            "scene_ic": scene.ic, "scene_seed": scene.seed,
            "disk_mass": params.get("disk_mass", 0.0),
            "disk_scale": params.get("disk_scale", 0.0),
            "disk_height": params.get("disk_height", 0.0),
            "halo_mass": params.get("halo_mass", 0.0),
            "halo_scale": params.get("halo_scale", 0.0),
        })
        engine = _make_engine(level.solver, level.softening, params)
        engine.load_state(state)
        return cls(engine, level, level.dt, dict(state.meta))

    @classmethod
    def from_checkpoint(cls, path: str,
                        overrides: Optional[dict] = None) -> "Simulation":
        state = State.load(path)
        m = state.meta
        solver = str(m.get("solver", "direct"))
        softening = float(m.get("softening", 0.1))
        dt = float(m.get("dt", 1e-4))
        params = {k: float(m[k]) for k in
                  ("disk_mass", "disk_scale", "disk_height",
                   "halo_mass", "halo_scale") if k in m}
        if overrides:
            dt = overrides.get("dt", dt) or dt
            softening = overrides.get("softening", softening) or softening
        level = LevelConfig(name=str(m.get("level", "resumed")), solver=solver,
                            n=state.n, dt=dt, softening=softening)
        engine = _make_engine(solver, softening, params)
        engine.load_state(state)
        return cls(engine, level, dt, dict(state.meta))

    # ------------------------------------------------------------------- run
    def run(self, steps: int, out: str, snap_every: int = 20,
            verbose: bool = True) -> int:
        """Advance ``steps`` steps, writing a snapshot every ``snap_every``.

        Returns the index of the last frame written.  Each snapshot doubles as a
        resumable checkpoint.
        """
        os.makedirs(out, exist_ok=True)
        eng = self.engine
        frame = eng.step_count // max(snap_every, 1)
        self._write(out, frame)

        t0 = time.time()
        for s in range(1, steps + 1):
            eng.step(self.dt)
            if s % snap_every == 0:
                frame += 1
                self._write(out, frame)
                if verbose:
                    rate = s / (time.time() - t0)
                    extra = ""
                    if hasattr(eng, "energies"):
                        _, _, e = eng.energies()
                        extra = f" E={e:.4g}"
                    print(f"  step {eng.step_count:6d} t={eng.time:.4f}"
                          f"{extra}  ({rate:.1f} steps/s)")
        return frame

    def _write(self, out: str, frame: int) -> None:
        state = self.engine.to_state()
        state.meta.update(self.meta)
        state.save(os.path.join(out, f"snap_{frame:04d}.h5"))
