import sys
import time
import numpy as np
from core.engine import init_taichi
from core.scenarios import build_scenario

def main():
    init_taichi("cuda")
    print("Testing GasDiskEngine with N=5000: Grid vs Direct O(N^2)")
    
    n_steps = 30
    n_particles = 5000
    
    # 1. Run with Grid (default since N >= 2000)
    print("Running with Grid...")
    t0 = time.time()
    engine_grid, dt = build_scenario('gas', n=n_particles, seed=42, overrides=None, gravity_mode="direct", theta=0.6, engine_params={})
    for _ in range(n_steps):
        engine_grid.step(dt)
    t_grid = time.time() - t0
    pos_grid = engine_grid.get("pos")
    rho_grid = engine_grid.get("rho")
    
    # 2. Run with Direct (force grid = None)
    print("Running Direct O(N^2)...")
    t0 = time.time()
    engine_direct, _ = build_scenario('gas', n=n_particles, seed=42, overrides=None, gravity_mode="direct", theta=0.6, engine_params={})
    engine_direct._grid = None  # disable grid
    for _ in range(n_steps):
        engine_direct.step(dt)
    t_direct = time.time() - t0
    pos_direct = engine_direct.get("pos")
    rho_direct = engine_direct.get("rho")
    
    # Compare
    pos_err = np.linalg.norm(pos_grid - pos_direct, axis=1).max()
    rho_err = np.abs(rho_grid - rho_direct).max()
    
    print(f"\nResults:")
    print(f"Grid time:   {t_grid:.3f} s")
    print(f"Direct time: {t_direct:.3f} s")
    print(f"Speedup:     {t_direct / t_grid:.2f}x")
    print(f"Max pos error: {pos_err:.2e}")
    print(f"Max rho error: {rho_err:.2e}")
    
    # Tolerance is loose because floating-point summation order differs
    if pos_err < 1e-4 and rho_err < 1e-2:
        print("\nPASS: Grid matches Direct")
    else:
        print("\nFAIL: Grid results differ from Direct")
        sys.exit(1)

if __name__ == '__main__':
    main()
