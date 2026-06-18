"""Phase 3 verification: the editor controller, headless (no GGUI window).

Run:
    .venv/Scripts/python.exe verify_phase3.py

Exercises the full editor workflow without opening a window: build, run/step,
auto-snapshot, resume, promote-to-production, display arrays, export.
"""
import os
import shutil

import numpy as np

from core.engine import init_taichi
from editor.controller import EditorController

OUT = os.path.join("output", "_verify3")


def main():
    init_taichi("cuda")
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)

    c = EditorController()
    c.s.ic = "disk"
    c.s.seed = 7
    c.s.level = "low"
    c.s.out_dir = OUT

    print("=== build ===")
    c.build()
    assert c.sim is not None and c.sim.engine.n > 0
    print(" ", c.status)

    print("=== run / advance ===")
    c.s.running = True
    c.advance(10)
    assert c.sim.engine.step_count == 10
    print(f"  stepped to {c.sim.engine.step_count}")

    print("=== auto-snapshot ===")
    c.s.auto_snapshot = True
    c.s.snap_every = 5
    c.advance(10)           # should write 2 snapshots
    snaps = sorted(f for f in os.listdir(OUT) if f.endswith(".h5"))
    assert len(snaps) >= 2, snaps
    print(f"  wrote {snaps}")

    print("=== resume from latest ===")
    latest = os.path.join(OUT, snaps[-1])
    step_before = c.sim.engine.step_count
    c.load_checkpoint(latest)
    assert c.sim.level.solver == "direct"
    print(f"  {c.status} (was at {step_before})")

    print("=== display arrays ===")
    pos, col = c.display_arrays()
    # display returns the (clipped) star subset, so 0 < N_display <= N_total
    assert 0 < pos.shape[0] <= c.sim.engine.n and col.shape == pos.shape
    assert np.isfinite(pos).all() and np.isfinite(col).all()
    assert col.max() <= 1.0 and col.min() >= 0.0
    print(f"  pos{pos.shape} col in [{col.min():.2f},{col.max():.2f}]")

    print("=== promote to medium (same seed) ===")
    c.promote("medium")
    assert c.sim.level.name == "medium" and c.s.seed == 7
    print(f"  {c.status}")

    print("=== export PLYs ===")
    c.s.running = False
    c.save_snapshot()
    c.export_plys()
    plys = [f for f in os.listdir(os.path.join(OUT, "ply")) if f.endswith(".ply")]
    assert plys, "no PLYs exported"
    print(f"  exported {len(plys)} PLY files")

    print("\nALL PHASE 3 (controller) CHECKS PASSED")


if __name__ == "__main__":
    main()
