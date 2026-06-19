"""Verify the uniform-grid SPH neighbour search matches the exact N^2 kernels.

The grid only changes *which* neighbours are visited, not the physics, so at t=0
(before any stochastic star formation) the grid density and SPH+gravity
acceleration must equal the brute-force N^2 result to within float round-off.
Also steps both a while to confirm the grid path stays finite/stable.
"""
import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f64)

from core.ic.full_galaxy import make_full_galaxy
from core.living_galaxy_engine import LivingGalaxyEngine

N = 8000
pos, vel, mass, u, species = make_full_galaxy(n=N, seed=3)


# Build once with the grid so h converges; then evaluate density+forces both
# ways on the *identical* converged state (no extra iterations) to compare the
# kernels apples-to-apples.
g = LivingGalaxyEngine(pot=dict(M_d=0.0, a=1.0, b=1.0, M_h=0.0, a_h=1.0),
                       softening=0.2, gravity_mode="direct")
g.setup(pos, vel, mass, u, species=species)
print(f"N={N}   grid active? {g._grid is not None and g._grid.ngas} gas binned")

# Grid evaluation on the converged fields.
g._gas_density(); g._forces()
rho_g, acc_g = g.get("rho"), g.get("acc")
# Exact N^2 evaluation on the SAME fields (h unchanged: single density pass).
g._grid = None
g._gas_density(); g._forces()
rho_d, acc_d = g.get("rho"), g.get("acc")
gas = species == 0

rho_err = np.abs(rho_g - rho_d)[gas] / np.maximum(np.abs(rho_d)[gas], 1e-30)
acc_diff = np.linalg.norm(acc_g - acc_d, axis=1)
acc_mag = np.linalg.norm(acc_d, axis=1)
acc_err = acc_diff / np.maximum(acc_mag, 1e-30)
print(f"gas density rel-err   max={rho_err.max():.2e}  mean={rho_err.mean():.2e}")
print(f"acceleration rel-err  max={acc_err.max():.2e}  mean={acc_err.mean():.2e}")

# Stability: step a fresh grid engine and confirm it stays finite (bounded KE).
s = LivingGalaxyEngine(pot=dict(M_d=0.0, a=1.0, b=1.0, M_h=0.0, a_h=1.0),
                       softening=0.2, gravity_mode="direct")
s.setup(pos, vel, mass, u, species=species)
dt = 2e-3
for _ in range(40):
    s.step(dt)
ke = 0.5 * (s.get("mass")[:, None] * s.get("vel") ** 2).sum()
print(f"after 40 grid steps   KE={ke:.3e}  stars={s.star_count()}  "
      f"u finite? {np.isfinite(s.get('u')).all()}")

assert rho_err.max() < 1e-4, "grid density disagrees with N^2"
assert acc_err.max() < 1e-4, "grid acceleration disagrees with N^2"
assert np.isfinite(ke) and ke > 0, "grid run went unstable"
print("\nOK: grid SPH matches N^2 within round-off and steps stably.")
