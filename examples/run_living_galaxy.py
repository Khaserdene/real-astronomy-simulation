"""Example: a "living" galaxy -- gas forms stars, stars age and explode (SN).

    .venv/Scripts/python.exe examples/run_living_galaxy.py --steps 750

Then render the stars (coloured by age) + gas:
    "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" --background \
        --python blender/render_frames.py -- \
        --ply output/living_demo --out output/living_demo/render \
        --radius 0.08 --emission 1.6 --glare 5 --cam-elev 62
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.engine import init_taichi
from core.ic.gas_disk import make_gas_disk
from core.living_galaxy_engine import LivingGalaxyEngine
from export.blackbody import blackbody_rgb
from export.to_pointcloud import _write_ply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=11000)
    ap.add_argument("--steps", type=int, default=750)
    ap.add_argument("--dt", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--out", default="output/living_demo")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    init_taichi("cuda")

    pos, vel, mass, u = make_gas_disk(n=args.n, seed=args.seed)
    eng = LivingGalaxyEngine(softening=0.2, sf_density_factor=5.0, sf_prob=0.03,
                             t_sn=0.025, r_fb=0.7, du_sn=500.0)
    eng.setup(pos, vel, mass, u)
    for s in range(args.steps):
        eng.step(args.dt)
        if s % 150 == 0:
            print(f"  step {s}: stars={eng.star_count()} u_max={eng.get('u').max():.0f}")
    print(f"final: {eng.star_count()} stars of {eng.n}")

    P = eng.get("pos").astype(np.float32)
    is_star = eng.get("is_star")
    ages = eng.ages()
    keep = np.linalg.norm(P, axis=1) < 30
    gas = (is_star == 0) & keep
    star = (is_star == 1) & keep

    col = np.zeros((P.shape[0], 3))
    t_star = np.clip(9500.0 - (ages / 0.3) * 6000.0, 3200.0, 9500.0)  # young=hot
    col[star] = blackbody_rgb(t_star[star])
    col[gas] = np.array([0.06, 0.025, 0.015])      # faint dust glow
    sel = gas | star
    rgb = (np.clip(col[sel], 0, 1) * 255).astype(np.uint8)
    _write_ply(os.path.join(args.out, "living.ply"), P[sel], rgb)
    print(f"wrote {int(sel.sum())} points -> {args.out}/living.ply")


if __name__ == "__main__":
    main()
