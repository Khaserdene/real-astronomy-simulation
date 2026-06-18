"""Phase 2 verification: levels, hardware detection, driver, resume.

Run:
    .venv/Scripts/python.exe verify_phase2.py

Checks:
  1. Hardware detection returns sane values.
  2. Level presets load and overrides apply.
  3. A Scene is deterministic: same seed+level -> identical initial state.
  4. The Simulation driver runs preview (test_particle) and direct solvers.
  5. Resume recovers solver/level from snapshot metadata and continues.
"""
import os
import shutil

import numpy as np

from core.engine import init_taichi
from core.config import (detect_hardware, get_level, suggest_level,
                         max_particles)
from core.scene import Scene
from core.simulation import Simulation
from core.state import State

OUT = os.path.join(os.path.dirname(__file__), "output", "_verify2")


def main():
    init_taichi("cuda")
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)

    # ---------------------------------------------------------- 1: hardware
    print("=== hardware detection ===")
    hw = detect_hardware()
    print(f"  {hw.gpu_name}  VRAM={hw.vram_mb}MiB  RAM={hw.ram_gb}GB  "
          f"CUDA={hw.has_cuda}")
    print(f"  fits ~{max_particles(hw):,} particles; suggests '{suggest_level(hw)}'")
    assert hw.vram_mb > 0, "expected a GPU"

    # ---------------------------------------------------------- 2: levels
    print("=== level presets ===")
    high = get_level("high")
    assert high.solver == "direct" and high.n == 120000
    ov = get_level("high", {"n": 5000})
    assert ov.n == 5000, "override failed"
    print(f"  high={high.n:,} (override -> {ov.n:,})  OK")

    # ---------------------------------------------------------- 3: determinism
    print("=== scene determinism (same seed -> identical IC) ===")
    sc = Scene(ic="disk", seed=123)
    a = Simulation.from_scene(sc, "low").engine.to_state()
    b = Simulation.from_scene(sc, "low").engine.to_state()
    assert np.array_equal(a.pos, b.pos) and np.array_equal(a.vel, b.vel)
    print(f"  N={a.n} reproduced identically  OK")

    # ---------------------------------------------------------- 4: solvers run
    print("=== driver runs both solvers ===")
    prev = Simulation.from_scene(Scene("disk", seed=7), "preview",
                                 overrides={"n": 5000})
    prev.run(20, out=os.path.join(OUT, "preview"), snap_every=10, verbose=False)
    assert prev.level.solver == "test_particle"

    prod = Simulation.from_scene(Scene("disk", seed=7), "low")
    prod.run(20, out=os.path.join(OUT, "prod"), snap_every=10, verbose=False)
    e_before = prod.engine.energies()[2]
    print(f"  preview + direct ran; direct E={e_before:.4g}  OK")

    # ---------------------------------------------------------- 5: resume
    print("=== resume recovers solver/level from metadata ===")
    snap = os.path.join(OUT, "prod", "snap_0002.h5")
    meta = State.load(snap).meta
    assert meta.get("solver") == "direct" and "softening" in meta
    res = Simulation.from_checkpoint(snap)
    assert res.level.solver == "direct"
    assert res.engine.step_count == 20
    res.run(10, out=os.path.join(OUT, "prod"), snap_every=10, verbose=False)
    assert res.engine.step_count == 30
    print(f"  resumed step 20 -> 30, solver={res.level.solver}  OK")

    print("\nALL PHASE 2 CHECKS PASSED")


if __name__ == "__main__":
    main()
