"""Smoothed-Particle Hydrodynamics (SPH) kernels -- the gas physics.

Standard Monaghan SPH with a cubic-spline kernel, ideal-gas equation of state,
Monaghan-Balsara artificial viscosity for shocks, and a per-particle adaptive
smoothing length so the resolution follows the density.  This is what turns the
simulator from "points under gravity" into something with gas: shocks, pressure
support, and (later) the dense star-forming regions and outflows that make a
galaxy look real.

Kernels operate on Taichi fields supplied by the engine.  Neighbour sums are
O(N^2) -- exact and simple, fine for the gas counts we run; a neighbour grid
will replace it for large N.

Conventions (3D cubic spline, Monaghan 1992):
    W(r,h) = (1/(pi h^3)) f(q),   q = r/h
    f(q)   = 1 - 1.5 q^2 + 0.75 q^3   (0<=q<1),   0.25 (2-q)^3   (1<=q<2)
"""
import taichi as ti

PI = 3.141592653589793


@ti.func
def cubic_w(r: ti.f64, h: ti.f64) -> ti.f64:
    """Cubic-spline kernel value W(r, h) in 3D."""
    q = r / h
    sigma = 1.0 / (PI * h * h * h)
    w = 0.0
    if q < 1.0:
        w = sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
    elif q < 2.0:
        t = 2.0 - q
        w = sigma * 0.25 * t * t * t
    return w


@ti.func
def min_image(dr, ly: ti.f64, lz: ti.f64):
    """Wrap a separation to the nearest periodic image in y, z (x is open).

    ``ly``/``lz`` <= 0 mean that direction is non-periodic.
    """
    if ly > 0.0:
        dr[1] -= ly * ti.round(dr[1] / ly)
    if lz > 0.0:
        dr[2] -= lz * ti.round(dr[2] / lz)
    return dr


@ti.func
def cubic_dwdr(r: ti.f64, h: ti.f64) -> ti.f64:
    """Radial derivative dW/dr of the cubic spline (<= 0)."""
    q = r / h
    sigma = 1.0 / (PI * h * h * h)
    dw = 0.0
    if q < 1.0:
        dw = sigma * (-3.0 * q + 2.25 * q * q) / h
    elif q < 2.0:
        t = 2.0 - q
        dw = sigma * (-0.75 * t * t) / h
    return dw


@ti.kernel
def compute_density(pos: ti.template(), mass: ti.template(),
                    rho: ti.template(), h: ti.template(), n: ti.i32,
                    ly: ti.f64, lz: ti.f64):
    """rho_i = sum_j m_j W(|r_ij|, h_i)  (includes the self term)."""
    for i in range(n):
        acc = 0.0
        pi = pos[i]
        hi = h[i]
        for j in range(n):
            dr = min_image(pos[j] - pi, ly, lz)
            r = dr.norm()
            if r < 2.0 * hi:
                acc += mass[j] * cubic_w(r, hi)
        rho[i] = acc


@ti.kernel
def update_h(rho: ti.template(), mass: ti.template(), h: ti.template(),
             n: ti.i32, eta: ti.f64):
    """Adaptive smoothing length h_i = eta (m_i / rho_i)^(1/3)."""
    for i in range(n):
        h[i] = eta * (mass[i] / rho[i]) ** (1.0 / 3.0)


@ti.kernel
def compute_pressure(rho: ti.template(), u: ti.template(),
                     pressure: ti.template(), cs: ti.template(),
                     n: ti.i32, gamma: ti.f64):
    """Ideal gas: P = (gamma-1) rho u,  sound speed c = sqrt(gamma P / rho)."""
    for i in range(n):
        p = (gamma - 1.0) * rho[i] * u[i]
        pressure[i] = p
        cs[i] = ti.sqrt(gamma * p / rho[i])


@ti.kernel
def compute_hydro_forces(pos: ti.template(), vel: ti.template(),
                         mass: ti.template(), rho: ti.template(),
                         pressure: ti.template(), cs: ti.template(),
                         h: ti.template(), acc: ti.template(),
                         du: ti.template(), n: ti.i32,
                         alpha: ti.f64, beta: ti.f64,
                         ly: ti.f64, lz: ti.f64):
    """Pressure-gradient acceleration + artificial viscosity, and du/dt.

    Uses the symmetric smoothing length h_ij = 1/2 (h_i + h_j).
    dv_i/dt = -sum_j m_j (P_i/rho_i^2 + P_j/rho_j^2 + Pi_ij) grad W_ij
    du_i/dt =  1/2 sum_j m_j (...) (v_i - v_j).grad W_ij
    """
    for i in range(n):
        a = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
        dudt = 0.0
        pi = pos[i]
        vi = vel[i]
        pr_i = pressure[i] / (rho[i] * rho[i])
        for j in range(n):
            if j == i:
                continue
            dr = min_image(pi - pos[j], ly, lz)
            r = dr.norm()
            hij = 0.5 * (h[i] + h[j])
            if r < 1e-12 or r >= 2.0 * hij:
                continue
            dv = vi - vel[j]
            pr_j = pressure[j] / (rho[j] * rho[j])

            # Monaghan artificial viscosity (only for approaching particles).
            visc = 0.0
            dvdr = dv.dot(dr)
            if dvdr < 0.0:
                mu = hij * dvdr / (r * r + 0.01 * hij * hij)
                c_bar = 0.5 * (cs[i] + cs[j])
                rho_bar = 0.5 * (rho[i] + rho[j])
                visc = (-alpha * c_bar * mu + beta * mu * mu) / rho_bar

            grad = cubic_dwdr(r, hij) * (dr / r)   # grad_i W_ij
            coeff = pr_i + pr_j + visc
            a += -mass[j] * coeff * grad
            dudt += 0.5 * mass[j] * coeff * dv.dot(grad)
        acc[i] = a
        du[i] = dudt
