"""Phase 5 verification: SPH on the Sod shock tube vs the exact Riemann solution.

Run:
    .venv/Scripts/python.exe verify_phase5.py

Sets up the classic Sod problem (rho/P jump 1/1 -> 0.125/0.1, gamma=1.4) as a 3D
isotropic-lattice tube, evolves it with the SPH engine to t=0.2, and overlays the
binned SPH density on the exact analytic solution.  A correct solver reproduces
the rarefaction fan, contact discontinuity, and shock at the right places.
"""
import os

import numpy as np

from core.engine import init_taichi
from core.sph_engine import SPHEngine

OUT = "output"
GAMMA = 1.4


# --------------------------------------------------------------- exact solver
def sod_exact(x, t, x0=0.5,
              rhoL=1.0, PL=1.0, rhoR=0.125, PR=0.1, gamma=GAMMA):
    """Exact Sod density profile at positions ``x`` and time ``t``."""
    cL = np.sqrt(gamma * PL / rhoL)
    cR = np.sqrt(gamma * PR / rhoR)

    def fK(p, rhoK, PK, cK):
        if p > PK:                                   # shock
            A = 2.0 / ((gamma + 1) * rhoK)
            B = (gamma - 1) / (gamma + 1) * PK
            return (p - PK) * np.sqrt(A / (p + B))
        return 2 * cK / (gamma - 1) * ((p / PK) ** ((gamma - 1) / (2 * gamma)) - 1)

    # Newton solve for the star-region pressure p*.
    p = 0.5 * (PL + PR)
    for _ in range(80):
        f = fK(p, rhoL, PL, cL) + fK(p, rhoR, PR, cR)
        df = 1e-6
        fp = (fK(p + df, rhoL, PL, cL) + fK(p + df, rhoR, PR, cR) - f) / df
        p_new = max(p - f / fp, 1e-8)
        if abs(p_new - p) < 1e-10:
            break
        p = p_new
    pstar = p
    ustar = 0.5 * (fK(pstar, rhoR, PR, cR) - fK(pstar, rhoL, PL, cL))

    # Left star density (rarefaction) and right star density (shock).
    rhoLstar = rhoL * (pstar / PL) ** (1.0 / gamma)
    rhoRstar = rhoR * ((pstar / PR + (gamma - 1) / (gamma + 1)) /
                       ((gamma - 1) / (gamma + 1) * pstar / PR + 1))
    cLstar = np.sqrt(gamma * pstar / rhoLstar)
    # Wave speeds.
    sh_head = -cL
    sh_tail = ustar - cLstar
    shock = (ustar + cR * np.sqrt((gamma + 1) / (2 * gamma) * pstar / PR
                                  + (gamma - 1) / (2 * gamma)))

    rho = np.empty_like(x)
    xi = (x - x0) / t
    for i, s in enumerate(xi):
        if s < sh_head:
            rho[i] = rhoL
        elif s < sh_tail:                            # rarefaction fan
            c = 2 / (gamma + 1) * (cL - (gamma - 1) / 2 * s)
            rho[i] = rhoL * (c / cL) ** (2 / (gamma - 1))
        elif s < ustar:
            rho[i] = rhoLstar
        elif s < shock:
            rho[i] = rhoRstar
        else:
            rho[i] = rhoR
    return rho


# --------------------------------------------------------------- tube builder
def build_tube(sL=0.0125, n_trans=8):
    """3D isotropic-lattice Sod tube, periodic in y,z.

    Density ratio 8 -> right spacing sR = 2 sL (since rho ~ 1/s^3).  The periodic
    transverse box Ly = n_trans * sR is an integer multiple of *both* spacings, so
    the lattice tiles seamlessly under the periodic wrap.
    """
    sR = 2.0 * sL
    Ly = n_trans * sR
    rhoL, PL, rhoR, PR = 1.0, 1.0, 0.125, 0.1
    m = rhoL * sL ** 3                                # equal particle mass

    def lattice(x0, x1, s):
        xs = np.arange(x0 + 0.5 * s, x1, s)
        ys = np.arange(0.5 * s, Ly - 1e-9, s)
        gx, gy, gz = np.meshgrid(xs, ys, ys, indexing="ij")
        return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    left = lattice(0.0, 0.5, sL)
    right = lattice(0.5, 1.0, sR)
    pos = np.vstack([left, right])
    n = pos.shape[0]
    mass = np.full(n, m)
    u = np.empty(n)
    u[:left.shape[0]] = PL / ((GAMMA - 1) * rhoL)
    u[left.shape[0]:] = PR / ((GAMMA - 1) * rhoR)
    vel = np.zeros((n, 3))
    frozen = ((pos[:, 0] < 0.06) | (pos[:, 0] > 0.94)).astype(np.int32)
    return pos, vel, mass, u, frozen, Ly


def main():
    init_taichi("cuda")
    os.makedirs(OUT, exist_ok=True)
    pos, vel, mass, u, frozen, Ly = build_tube()
    print(f"tube: N={pos.shape[0]} ({int(frozen.sum())} frozen), Ly={Ly:.3f}")

    eng = SPHEngine(gamma=GAMMA, periodic_y=Ly, periodic_z=Ly)
    eng.setup(pos, vel, mass, u, frozen)
    rho0 = eng.get("rho")
    print(f"initial density: left~{np.median(rho0[pos[:, 0] < 0.45]):.3f} "
          f"right~{np.median(rho0[pos[:, 0] > 0.55]):.3f} (expect 1.0 / 0.125)")
    e0 = eng.total_energy()

    t_end, dt = 0.2, 5e-4
    nsteps = int(t_end / dt)
    for s in range(nsteps):
        eng.step(dt)
    e1 = eng.total_energy()
    print(f"steps={nsteps} t={eng.time:.3f}  energy {e0:.4f} -> {e1:.4f} "
          f"(drift {abs(e1 - e0) / e0:.1e})")

    # Bin density along x (periodic transverse -> every particle is interior).
    P = eng.get("pos")
    rho = eng.get("rho")
    live = frozen == 0
    xb = np.linspace(0.1, 0.9, 40)
    idx = np.digitize(P[live, 0], xb)
    rho_live = rho[live]
    binned = np.array([rho_live[idx == k].mean() if (idx == k).any() else np.nan
                       for k in range(1, len(xb))])
    xc = 0.5 * (xb[1:] + xb[:-1])

    # Compare against the exact solution at the bin centres.
    exact = sod_exact(xc, eng.time)
    valid = ~np.isnan(binned)
    rms = np.sqrt(np.nanmean((binned[valid] - exact[valid]) ** 2))
    print(f"density RMS error vs exact: {rms:.3f}")

    _plot(xc, binned, eng.time, os.path.join(OUT, "sod_shock.png"))
    print(f"wrote {OUT}/sod_shock.png")
    assert abs(e1 - e0) / e0 < 0.05, "energy not conserved"
    assert rms < 0.08, f"SPH density too far from exact Sod ({rms:.3f})"
    print("\nPHASE 5 SPH (Sod shock tube) PASSED")


def _plot(xc, binned, t, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xf = np.linspace(0.1, 0.9, 400)
    plt.figure(figsize=(8, 5))
    plt.plot(xf, sod_exact(xf, t), "k-", lw=2, label="exact")
    plt.plot(xc, binned, "o", ms=5, color="crimson", label="SPH")
    plt.xlabel("x"); plt.ylabel("density"); plt.title(f"Sod shock tube, t={t:.2f}")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


if __name__ == "__main__":
    main()
