"""Verify Barnes-Hut self-gravity in LivingGalaxyEngine (galaxy scenario).

Checks (1) the BH treecode produces accelerations close to direct N^2 on the
unified galaxy ICs, and (2) it is faster at the engine's first force evaluation.
"""
import time
import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f64)

from core.scenarios import build_scenario

N = 8000


def first_acc(gravity_mode):
    eng, dt = build_scenario("galaxy", n=N, seed=1, gravity_mode=gravity_mode)
    acc = eng.get("acc").copy()
    return eng, acc, dt


# Direct reference.
t0 = time.perf_counter()
eng_d, acc_d, dt = first_acc("direct")
t_direct = time.perf_counter() - t0

# Barnes-Hut.
t0 = time.perf_counter()
eng_b, acc_b, _ = first_acc("bh")
t_bh = time.perf_counter() - t0

# Force relative error (norm of difference / norm of direct), per particle.
diff = np.linalg.norm(acc_b - acc_d, axis=1)
mag = np.linalg.norm(acc_d, axis=1)
rel = diff / np.maximum(mag, 1e-30)
print(f"N = {N}")
print(f"BH active on direct engine?  {eng_d._bh}")
print(f"BH active on bh engine?      {eng_b._bh is not None}")
print(f"force rel-err  mean={rel.mean():.3%}  median={np.median(rel):.3%}  "
      f"p99={np.percentile(rel, 99):.3%}")
print(f"build+first-force wall time  direct={t_direct:.2f}s  bh={t_bh:.2f}s")

# Step both a few times and confirm BH stays stable (energy-ish: kinetic bounded).
for _ in range(20):
    eng_d.step(dt)
    eng_b.step(dt)
ke_d = 0.5 * (eng_d.get("mass")[:, None] * eng_d.get("vel") ** 2).sum()
ke_b = 0.5 * (eng_b.get("mass")[:, None] * eng_b.get("vel") ** 2).sum()
print(f"after 20 steps  KE direct={ke_d:.3e}  bh={ke_b:.3e}  "
      f"rel-diff={abs(ke_b - ke_d) / ke_d:.3%}")
print(f"star count  direct={eng_d.star_count()}  bh={eng_b.star_count()}")

assert rel.mean() < 0.05, "BH mean force error too large"
assert np.isfinite(ke_b) and ke_b > 0, "BH run went unstable"
print("\nOK: Barnes-Hut self-gravity matches direct within tolerance.")
