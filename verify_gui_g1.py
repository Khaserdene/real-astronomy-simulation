"""G1 verification: the unified SimController across ALL scenarios (headless).

    .venv/Scripts/python.exe verify_gui_g1.py

Builds, steps, displays, snapshots and resumes every registered scenario --
gravity (disk/merger/cosmo/plummer) and SPH (gas/living/impact) -- through one
interface, with no GUI.
"""
import os
import shutil

import numpy as np

from core.engine import init_taichi
from core.scenarios import SCENARIOS
from core.sim_controller import SimController

OUT = os.path.join("output", "_verify_g1")


def main():
    init_taichi("cuda")
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)

    for name in SCENARIOS:
        print(f"=== {name} ===")
        c = SimController()
        c.s.scenario = name
        c.s.seed = 3
        c.s.n = 3000
        c.s.out_dir = os.path.join(OUT, name)
        c.build()
        assert c.engine is not None and c.engine.n > 0
        n0 = c.engine.n

        # step
        c.s.running = True
        c.advance(5)
        assert c.engine.step_count == 5

        # display arrays
        pos, rgba = c.display_arrays()
        assert 0 < pos.shape[0] <= n0
        assert rgba.shape == (pos.shape[0], 4)
        assert np.isfinite(pos).all() and np.isfinite(rgba).all()
        assert 0.0 <= rgba.min() and rgba.max() <= 1.0

        # snapshot + resume
        snap = c.save_snapshot()
        c2 = SimController()
        c2.load_checkpoint(snap)
        assert c2.engine.step_count == 5
        c2.s.running = True
        c2.advance(3)
        assert c2.engine.step_count == 8

        # playback path
        pos2, rgba2 = SimController.display_snapshot(snap)
        assert pos2.shape[0] > 0

        print(f"  OK  N={n0:,}  display={pos.shape[0]:,}  "
              f"resume {c2.engine.step_count}  {c.s.scenario}")

    print("\nALL G1 (unified controller) CHECKS PASSED")


if __name__ == "__main__":
    main()
