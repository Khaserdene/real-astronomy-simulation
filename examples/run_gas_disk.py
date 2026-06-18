"""Example: simulate an SPH gas disk and export it for a Blender volume render.

    .venv/Scripts/python.exe examples/run_gas_disk.py --steps 400 --n 10000

Then render the volume:
    "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" --background \
        --python blender/render_gas.py -- \
        --ply output/gas_demo/gas.ply --out output/gas_demo/render.png \
        --voxel 0.22 --radius 0.35 --density 0.12 --emission 0.13 \
        --cam-dist 85 --cam-elev 32
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.engine import init_taichi
from core.ic.gas_disk import make_gas_disk
from core.gas_disk_engine import GasDiskEngine
from export.to_pointcloud import _write_ply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--dt", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="output/gas_demo")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    init_taichi("cuda")

    pos, vel, mass, u = make_gas_disk(n=args.n, seed=args.seed)
    eng = GasDiskEngine(softening=0.2)
    eng.setup(pos, vel, mass, u)
    for s in range(args.steps):
        eng.step(args.dt)
        if s % 50 == 0:
            print(f"  step {s}: rho_max={eng.get('rho').max():.3f}")

    P = eng.get("pos").astype(np.float32)
    P = P[np.linalg.norm(P, axis=1) < 40]          # drop far stragglers
    rgb = np.full((P.shape[0], 3), 255, np.uint8)
    _write_ply(os.path.join(args.out, "gas.ply"), P, rgb)
    print(f"wrote {P.shape[0]} points -> {args.out}/gas.ply")


if __name__ == "__main__":
    main()
