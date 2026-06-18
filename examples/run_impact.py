"""Example: a giant impact -- two self-gravitating bodies collide (SPH).

    .venv/Scripts/python.exe examples/run_impact.py --steps 520

Then render (colour = internal energy, so shock-heated material glows):
    "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" --background \
        --python blender/render_frames.py -- \
        --ply output/impact_demo --out output/impact_demo/render \
        --radius 0.06 --emission 1.6 --glare 4 --cam-dist 16 --cam-elev 35
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.engine import init_taichi
from core.ic.impact import make_impact
from core.gas_disk_engine import GasDiskEngine
from export.blackbody import blackbody_rgb
from export.to_pointcloud import _write_ply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-each", type=int, default=7000)
    ap.add_argument("--steps", type=int, default=520)
    ap.add_argument("--dt", type=float, default=5e-4)
    ap.add_argument("--v", type=float, default=30.0, help="approach speed [km/s]")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="output/impact_demo")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    init_taichi("cuda")

    pos, vel, mass, u = make_impact(n_each=args.n_each, separation=6.0,
                                    impact_param=1.6, v_approach=args.v,
                                    seed=args.seed)
    # Pure SPH + self-gravity (zero external potential).
    eng = GasDiskEngine(pot=dict(M_d=0.0, a=1.0, b=1.0, M_h=0.0, a_h=1.0),
                        softening=0.1)
    eng.setup(pos, vel, mass, u)
    for s in range(args.steps):
        eng.step(args.dt)
    U = eng.get("u")
    print(f"shock heating: u {U.min():.0f} -> {U.max():.0f}")

    P = eng.get("pos").astype(np.float32)
    lo, hi = np.percentile(U, 5), np.percentile(U, 98)
    norm = np.clip((U - lo) / (hi - lo + 1e-9), 0, 1)
    rgb = (blackbody_rgb(2600.0 + 6500.0 * norm) * 255).astype(np.uint8)
    _write_ply(os.path.join(args.out, "impact.ply"), P, rgb)
    print(f"wrote {P.shape[0]} points -> {args.out}/impact.ply")


if __name__ == "__main__":
    main()
