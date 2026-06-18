"""Leapfrog (kick-drift-kick) integrator kernels.

Leapfrog is symplectic: it conserves energy to within a bounded oscillation over
long integrations, which is exactly what we verify on the Plummer test.  A full
step is  kick(dt/2) -> drift(dt) -> recompute acc -> kick(dt/2),  reusing the
acceleration computed at the end of the previous step.
"""
import taichi as ti


@ti.kernel
def kick(vel: ti.template(), acc: ti.template(), n: ti.i32, dt_half: ti.f64):
    """Half-step velocity update: v += a * dt/2."""
    for i in range(n):
        vel[i] += acc[i] * dt_half


@ti.kernel
def drift(pos: ti.template(), vel: ti.template(), n: ti.i32, dt: ti.f64):
    """Full-step position update: x += v * dt."""
    for i in range(n):
        pos[i] += vel[i] * dt
