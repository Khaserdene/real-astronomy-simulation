"""Phase 1 verification: energy conservation, virial balance, I/O round-trip.

Run with the project venv:
    .venv/Scripts/python.exe verify_phase1.py

Checks:
  1. Plummer sphere integrates with bounded energy drift (symplectic leapfrog).
  2. The system stays near virial equilibrium  (-2*KE/PE ~ 1).
  3. A State saved to HDF5 reloads identically.
  4. An exponential disk produces a rotating, disk-shaped distribution (plot).
"""
import os

import numpy as np

from core.engine import Engine, init_taichi
from core.state import State
from core.ic.plummer import make_plummer
from core.ic.disk import make_disk_galaxy

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


def main():
    init_taichi("cuda")

    # ---------------------------------------------------------- 1 & 2: Plummer
    print("=== Plummer equilibrium test ===")
    state = make_plummer(n=3000, total_mass=10.0, scale_radius=1.0, seed=42)
    eng = Engine(softening=0.05)
    eng.load_state(state)

    ke0, pe0, e0 = eng.energies()
    virial0 = -2.0 * ke0 / pe0
    print(f"  initial: KE={ke0:.3f} PE={pe0:.3f} E={e0:.3f} "
          f"virial(-2KE/PE)={virial0:.3f}")

    dt = 3e-5
    nsteps = 1000
    for s in range(nsteps):
        eng.step(dt)

    ke1, pe1, e1 = eng.energies()
    virial1 = -2.0 * ke1 / pe1
    drift = abs((e1 - e0) / e0)
    print(f"  final  : KE={ke1:.3f} PE={pe1:.3f} E={e1:.3f} "
          f"virial(-2KE/PE)={virial1:.3f}")
    print(f"  steps={nsteps} dt={dt}  |dE/E|={drift:.2e}")
    assert drift < 1e-3, f"energy drift too large: {drift:.2e}"
    assert 0.8 < virial1 < 1.2, f"virial ratio off: {virial1:.3f}"
    print("  PASS: energy conserved & near virial equilibrium")

    # ---------------------------------------------------------- 3: I/O round-trip
    print("=== HDF5 round-trip test ===")
    path = os.path.join(OUT, "roundtrip.h5")
    s_out = eng.to_state()
    s_out.save(path)
    s_in = State.load(path)
    assert np.allclose(s_out.pos, s_in.pos)
    assert np.allclose(s_out.vel, s_in.vel)
    assert np.array_equal(s_out.ids, s_in.ids)
    assert s_out.step == s_in.step and s_out.seed == s_in.seed
    print(f"  PASS: state with N={s_in.n} step={s_in.step} reloaded identically")

    # ---------------------------------------------------------- 4: disk visual
    print("=== Disk galaxy visual ===")
    disk = make_disk_galaxy(n_disk=8000, n_halo=8000, seed=7)
    eng2 = Engine(softening=0.1)
    eng2.load_state(disk)
    for s in range(300):
        eng2.step(1e-4)
    _plot_disk(eng2.to_state(), os.path.join(OUT, "disk_faceon.png"))
    print(f"  wrote {os.path.join(OUT, 'disk_faceon.png')}")
    print("\nALL PHASE 1 CHECKS PASSED")


def _plot_disk(state: State, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from core.state import PTYPE_STAR

    stars = state.ptype == PTYPE_STAR
    p = state.pos[stars]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.scatter(p[:, 0], p[:, 1], s=0.5, c="gold", alpha=0.4)
    ax1.set(title="Face-on (x-y)", xlabel="kpc", ylabel="kpc",
            xlim=(-25, 25), ylim=(-25, 25), aspect="equal")
    ax2.scatter(p[:, 0], p[:, 2], s=0.5, c="gold", alpha=0.4)
    ax2.set(title="Edge-on (x-z)", xlabel="kpc", ylabel="kpc",
            xlim=(-25, 25), ylim=(-25, 25), aspect="equal")
    for ax in (ax1, ax2):
        ax.set_facecolor("black")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
