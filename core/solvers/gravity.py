"""Direct N-body (O(N^2)) gravity on the GPU via Taichi.

This is the exact, brute-force pairwise solver -- simple and accurate, suitable
for the learning/verification stage and for small-to-medium particle counts.
A Barnes-Hut tree solver (same interface) will replace it for 1e6+ particles.

All kernels operate on Taichi fields supplied by the caller (the Engine), using
Plummer softening ``eps`` to avoid singular close encounters.
"""
import taichi as ti


@ti.kernel
def compute_acc_direct(pos: ti.template(), mass: ti.template(),
                       acc: ti.template(), n: ti.i32,
                       grav: ti.f64, eps2: ti.f64):
    """Accelerations a_i = G * sum_j m_j (r_j - r_i) / (|r_ij|^2 + eps^2)^{3/2}."""
    for i in range(n):
        a = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
        pi = pos[i]
        for j in range(n):
            d = pos[j] - pi
            r2 = d.dot(d) + eps2
            inv_r = 1.0 / ti.sqrt(r2)
            a += (grav * mass[j] * inv_r * inv_r * inv_r) * d
        acc[i] = a


@ti.kernel
def kinetic_energy(vel: ti.template(), mass: ti.template(),
                   n: ti.i32) -> ti.f64:
    """Total kinetic energy  KE = 1/2 sum_i m_i |v_i|^2."""
    ke = 0.0
    for i in range(n):
        ke += 0.5 * mass[i] * vel[i].dot(vel[i])
    return ke


@ti.kernel
def potential_energy(pos: ti.template(), mass: ti.template(), n: ti.i32,
                     grav: ti.f64, eps2: ti.f64) -> ti.f64:
    """Total (softened) potential energy  PE = -G sum_{i<j} m_i m_j / sqrt(r^2+eps^2).

    Computed as -1/2 of the full double sum over i != j.
    """
    pe = 0.0
    for i in range(n):
        pi = pos[i]
        mi = mass[i]
        acc_i = 0.0
        for j in range(n):
            if j != i:
                d = pos[j] - pi
                r2 = d.dot(d) + eps2
                acc_i += mass[j] / ti.sqrt(r2)
        pe += -0.5 * grav * mi * acc_i
    return pe
