"""Headless command-line runner: simulate and write frame-by-frame snapshots.

GUI-independent entry point. Runs on any machine from a Scene + level, writing
one HDF5 snapshot per frame (each also a resumable checkpoint) into a folder --
the artifacts Blender will later render.

Examples
--------
    # Show detected hardware and the suggested level, then exit
    .venv/Scripts/python.exe cli.py --suggest

    # Instant low-fidelity preview (massless tracers in analytic potential)
    .venv/Scripts/python.exe cli.py --ic disk --seed 7 --level preview \
        --steps 1500 --snap-every 30 --out output/preview

    # Same galaxy (same seed!) at production fidelity
    .venv/Scripts/python.exe cli.py --ic disk --seed 7 --level high \
        --steps 4000 --snap-every 25 --out output/run_high

    # Resume an interrupted run from its last snapshot (any machine)
    .venv/Scripts/python.exe cli.py --resume output/run_high/snap_0040.h5 \
        --steps 4000 --snap-every 25 --out output/run_high
"""
from __future__ import annotations

import argparse

from core.engine import init_taichi
from core.config import detect_hardware, suggest_level, get_level, max_particles
from core.scene import Scene
from core.simulation import Simulation


def cmd_suggest():
    hw = detect_hardware()
    print(f"GPU : {hw.gpu_name}  ({hw.vram_mb} MiB VRAM, CUDA={hw.has_cuda})")
    print(f"RAM : {hw.ram_gb} GB")
    print(f"Fits ~{max_particles(hw):,} particles in the VRAM budget")
    lvl = suggest_level(hw)
    print(f"Suggested level: {lvl}  ->  {get_level(lvl).note}")


def main():
    ap = argparse.ArgumentParser(description="Headless N-body runner")
    ap.add_argument("--suggest", action="store_true",
                    help="print hardware + suggested level and exit")
    ap.add_argument("--ic", choices=["disk", "plummer", "merger", "cosmo"],
                    default="disk")
    ap.add_argument("--level", default="medium",
                    choices=["preview", "low", "medium", "high", "ultra"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", help="snapshot .h5 to continue from")
    # Optional overrides of the chosen level.
    ap.add_argument("--n", type=int, help="override particle count")
    ap.add_argument("--dt", type=float, help="override timestep")
    ap.add_argument("--softening", type=float, help="override softening [kpc]")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--snap-every", type=int, default=20)
    ap.add_argument("--out", default="output/run")
    ap.add_argument("--arch", default="cuda", choices=["cuda", "vulkan", "cpu"])
    args = ap.parse_args()

    if args.suggest:
        init_taichi(args.arch)
        cmd_suggest()
        return

    init_taichi(args.arch)
    overrides = {"n": args.n, "dt": args.dt, "softening": args.softening}

    if args.resume:
        sim = Simulation.from_checkpoint(args.resume, overrides=overrides)
        print(f"resuming {args.resume}: N={sim.engine.n} "
              f"step={sim.engine.step_count} level={sim.level.name} "
              f"solver={sim.level.solver}")
    else:
        scene = Scene(ic=args.ic, seed=args.seed)
        sim = Simulation.from_scene(scene, args.level, overrides=overrides)
        print(f"new run: ic={args.ic} seed={args.seed} level={sim.level.name} "
              f"solver={sim.level.solver} N={sim.engine.n} dt={sim.dt}")

    last = sim.run(args.steps, out=args.out, snap_every=args.snap_every)
    print(f"done: {last + 1} snapshots in {args.out}/")


if __name__ == "__main__":
    main()
