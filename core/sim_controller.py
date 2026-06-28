"""Unified backend controller for the GUI -- runs any scenario, no GUI code.

This is the single "brain" the PyQt app talks to.  It builds any scenario through
:mod:`core.scenarios`, advances it, exposes type-aware display points/colours
(shared with snapshot playback), and handles snapshots/checkpoints.  Keeping it
GUI-free means the whole workflow stays unit-testable headlessly.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np

from core.scenarios import SCENARIOS, build_scenario, resume_scenario
from core.scene_spec import SceneSpec
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
    gravity_mode: str = "direct"   # "direct" (N^2) | "bh" (Barnes-Hut treecode)
    theta: float = 0.6
    # Physics (engine) parameters -- edited in the GUI's Physics panel.
    cooling: bool = True           # radiative cooling of gas
    u_floor: float = 60.0          # cooling temperature floor [(km/s)^2]
    t_cool: float = 0.02           # cooling timescale [code units]
    sf_prob: float = 0.03          # star-formation probability / step (eligible gas)
    du_sn: float = 400.0           # supernova feedback energy per event
    v_sn: float = 50.0             # supernova kinetic kick velocity
    live_halo: bool = True         # Use N-body DM instead of analytic potential

    def engine_params(self) -> dict:
        return dict(cooling=self.cooling, u_floor=self.u_floor,
                    t_cool=self.t_cool, sf_prob=self.sf_prob, du_sn=self.du_sn, v_sn=self.v_sn)


class SimController:
    def __init__(self):
        self.s = Settings()
        self.engine = None
        self.dt = 1e-4
        self.spec: SceneSpec | None = None    # what was built (for resume/§7)
        self.continuing = False               # resumed from an existing folder?
        self._frame = 0
        self._since_snap = 0
        self.status = "no simulation -- press Build"

    # --------------------------------------------------------------- building
    def build(self, overrides: dict | None = None) -> str:
        ov = overrides.copy() if overrides else {}
        ov["live_halo"] = getattr(self.s, "live_halo", True)
        self.engine, self.dt = build_scenario(
            self.s.scenario, n=self.s.n, seed=self.s.seed, overrides=ov,
            gravity_mode=self.s.gravity_mode, theta=self.s.theta,
            engine_params=self.s.engine_params())
        self.spec = SceneSpec.single(
            self.s.scenario, self.s.n, self.s.seed, overrides)
        self.spec.gravity_mode = self.s.gravity_mode
        self.spec.theta = self.s.theta
        self.continuing = False
        self.s.running = False
        self._frame = 0
        self._since_snap = 0
        self.status = (f"built '{self.s.scenario}' "
                       f"(N={self.engine.n:,}, dt={self.dt:g})")
        return self.status

    def build_scene(self, spec) -> str:
        """Build a composed multi-object scene (from the editor) and load it."""
        from core.objects import build_scene as _build_scene

        self.engine, self.dt = _build_scene(spec)
        self.spec = spec
        self.continuing = False
        self.s.scenario = spec.objects[0].template if spec.objects else "scene"
        self.s.n = self.engine.n
        self.s.running = False
        self._frame = 0
        self._since_snap = 0
        self.status = (f"built scene ({len(spec.objects)} objects, "
                       f"N={self.engine.n:,}, dt={self.dt:g})")
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
        if self.spec is not None:
            state.meta["scene"] = self.spec.to_json()
        path = os.path.join(self.s.out_dir, f"snap_{self._frame:04d}.h5")
        state.save(path)
        self._frame += 1
        self.status = f"saved {os.path.basename(path)}"
        return path

    def load_checkpoint(self, path: str) -> str:
        state = State.load(path)
        scene = state.meta.get("scene")
        self.spec = SceneSpec.from_json(scene) if scene else None
        scenario = (self.spec.primary.template if self.spec
                    else str(state.meta.get("scenario", self.s.scenario)))
        self.engine, self.dt = resume_scenario(scenario, state, self.spec)
        self.s.scenario = scenario
        if self.spec is not None:
            self.s.n = self.spec.primary.n
            self.s.seed = self.spec.primary.seed
        # Continue writing alongside the loaded file, after its last frame, so
        # the existing run's snapshots are preserved rather than overwritten.
        self.s.out_dir = os.path.dirname(path) or self.s.out_dir
        self._frame = self._next_frame_index(self.s.out_dir)
        self.continuing = True
        self.s.running = False
        self.status = (f"resumed {os.path.basename(path)} "
                       f"(step {self.engine.step_count}, "
                       f"next frame {self._frame:04d})")
        return self.status

    def open_folder(self, folder: str) -> str:
        """Resume the latest snapshot in *folder* and continue writing there."""
        snaps = self.list_snapshots(folder)
        if not snaps:
            self.status = f"no snapshots in {folder}"
            return self.status
        return self.load_checkpoint(snaps[-1])

    def list_snapshots(self, folder: str | None = None):
        folder = folder or self.s.out_dir
        return sorted(glob.glob(os.path.join(folder, "snap_*.h5")))

    def _next_frame_index(self, folder: str) -> int:
        """One past the highest snap_NNNN index already in *folder* (else 0)."""
        idx = -1
        for p in self.list_snapshots(folder):
            m = re.search(r"snap_(\d+)\.h5$", os.path.basename(p))
            if m:
                idx = max(idx, int(m.group(1)))
        return idx + 1

    # ----------------------------------------------------------- display data
    def display_arrays(self, clip_kpc: float = 120.0):
        if self.engine is None:
            return np.zeros((0, 3), np.float32), np.zeros((0, 4), np.float32)
        return state_to_display(self.engine.to_state(), clip_kpc)

    @staticmethod
    def display_snapshot(path: str, clip_kpc: float = 120.0):
        """Load a saved snapshot and return its display arrays (for playback)."""
        return state_to_display(State.load(path), clip_kpc)

    def edit_particles(self, remove_idx=None, add_pos=None,
                       add_ptype: int = 1) -> str:
        """Brush edit: remove particles by index and/or add a cluster, then
        rebuild the engine from the edited arrays (preserving kind/time/step)."""
        if self.engine is None:
            return "nothing to edit"
        st = self.engine.to_state()
        pos, vel, mass, ptype = st.pos, st.vel, st.mass, st.ptype
        u = st.u

        if remove_idx is not None and len(remove_idx):
            keep = np.ones(len(pos), bool)
            keep[np.asarray(remove_idx, int)] = False
            pos, vel, mass, ptype = pos[keep], vel[keep], mass[keep], ptype[keep]
            if u is not None:
                u = u[keep]

        if add_pos is not None and len(add_pos):
            add_pos = np.asarray(add_pos, np.float64).reshape(-1, 3)
            m = len(add_pos)
            avel = np.zeros((m, 3))
            amass = np.full(m, float(np.median(mass)) if len(mass) else 1e-5)
            aptype = np.full(m, int(add_ptype), np.int32)
            pos = np.vstack([pos, add_pos]); vel = np.vstack([vel, avel])
            mass = np.concatenate([mass, amass])
            ptype = np.concatenate([ptype, aptype])
            if u is not None:
                au = np.full(m, float(np.median(u)) if len(u) else 200.0)
                u = np.concatenate([u, au])

        self._rebuild_from_arrays(pos, vel, mass, ptype, u,
                                  st.time, st.step)
        self.status = f"edited -> N={self.engine.n:,}"
        return self.status

    def _rebuild_from_arrays(self, pos, vel, mass, ptype, u, time, step):
        ids = np.arange(len(pos), dtype=np.int64)
        state = State(pos=pos, vel=vel, mass=mass, ptype=ptype, ids=ids,
                      u=u, time=time, step=step)
        scenario = self.spec.primary.template if self.spec else self.s.scenario
        self.engine, self.dt = resume_scenario(scenario, state, self.spec)
        self.s.n = self.engine.n

    def frame_progress(self) -> float:
        """Progress in [0,1] toward the next snapshot frame (cadence indicator).

        Tied to ``snap_every`` so it sweeps 0->100% once per produced frame, with
        or without auto-snapshot enabled.
        """
        if self.engine is None or self.s.snap_every <= 0:
            return 0.0
        return (self._since_snap % self.s.snap_every) / self.s.snap_every

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
