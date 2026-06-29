"""Living-galaxy engine: gas + stars with star formation and SN feedback.

Gas and star particles share one field set.  All particles feel gravity and the
analytic galaxy potential; only gas particles carry SPH pressure.  Over time:

  * star formation  -- dense gas particles convert in place into star particles
    (type flips, formation time recorded);
  * stellar ageing  -- a star's age = current time - birth time drives its colour
    at render (young = hot/blue, old = cool/red);
  * supernova feedback -- a star, a short time after birth, injects thermal energy
    into the surrounding gas, heating and pushing it (bubbles / outflows).
  * mass return -- exploding stars return a fraction of their mass to surrounding
    gas, enriching the ISM.

Physics is N-independent: star formation uses a Schmidt-law (rate ~ rho/t_ff),
SN energy is distributed as a fixed budget per event (not per neighbor), and
stellar lifetimes are stochastic (only massive stars go SN).
"""
import numpy as np
import taichi as ti

from core.units import G
from core.solvers.sph import cubic_w, cubic_dwdr

# Physical constant for free-fall time: sqrt(3*pi/32) ~ 0.5427
_TFF_COEFF = 0.5427


@ti.kernel
def gas_density(pos: ti.template(), mass: ti.template(), rho: ti.template(),
                h: ti.template(), is_star: ti.template(), n: ti.i32):
    for i in range(n):
        if is_star[i] == 0:
            acc = 0.0
            pi = pos[i]
            hi = h[i]
            for j in range(n):
                if is_star[j] == 0:
                    r = (pos[j] - pi).norm()
                    if r < 2.0 * hi:
                        acc += mass[j] * cubic_w(r, hi)
            rho[i] = ti.max(acc, 1e-12)


@ti.kernel
def update_h_gas(mass: ti.template(), rho: ti.template(), h: ti.template(),
                 is_star: ti.template(), n: ti.i32, eta: ti.f64, min_h: ti.f64):
    """Update smoothing length h for gas from density with a minimum floor."""
    for i in range(n):
        if is_star[i] == 0:
            new_h = eta * (mass[i] / ti.max(rho[i], 1e-12)) ** (1.0 / 3.0)
            h[i] = ti.max(new_h, min_h)


@ti.kernel
def gas_pressure(rho: ti.template(), u: ti.template(), pressure: ti.template(),
                 cs: ti.template(), is_star: ti.template(), n: ti.i32,
                 gamma: ti.f64):
    for i in range(n):
        if is_star[i] == 0:
            p = (gamma - 1.0) * rho[i] * u[i]
            pressure[i] = p
            cs[i] = ti.sqrt(gamma * p / rho[i])


@ti.kernel
def gas_forces(pos: ti.template(), vel: ti.template(), mass: ti.template(),
               rho: ti.template(), pressure: ti.template(), cs: ti.template(),
               h: ti.template(), acc: ti.template(), du: ti.template(),
               is_star: ti.template(), n: ti.i32, alpha: ti.f64, beta: ti.f64):
    for i in range(n):
        a = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
        dudt = 0.0
        if is_star[i] == 0:
            pi = pos[i]
            vi = vel[i]
            pr_i = pressure[i] / (rho[i] * rho[i])
            for j in range(n):
                if is_star[j] == 0 and j != i:
                    dr = pi - pos[j]
                    r = dr.norm()
                    hij = 0.5 * (h[i] + h[j])
                    if 1e-12 < r < 2.0 * hij:
                        dv = vi - vel[j]
                        pr_j = pressure[j] / (rho[j] * rho[j])
                        visc = 0.0
                        dvdr = dv.dot(dr)
                        if dvdr < 0.0:
                            mu = hij * dvdr / (r * r + 0.01 * hij * hij)
                            c_bar = 0.5 * (cs[i] + cs[j])
                            rho_bar = 0.5 * (rho[i] + rho[j])
                            visc = (-alpha * c_bar * mu + beta * mu * mu) / rho_bar
                        grad = cubic_dwdr(r, hij) * (dr / r)
                        coeff = pr_i + pr_j + visc
                        a += -mass[j] * coeff * grad
                        dudt += 0.5 * mass[j] * coeff * dv.dot(grad)
        acc[i] = a
        du[i] = dudt


@ti.kernel
def grav_self_direct(pos: ti.template(), mass: ti.template(),
                     acc: ti.template(), n: ti.i32, eps2: ti.f64):
    """Add direct-N^2 self-gravity (all particles) to acc."""
    for i in range(n):
        av = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
        pi = pos[i]
        for j in range(n):
            d = pos[j] - pi
            r2 = d.dot(d) + eps2
            inv = 1.0 / ti.sqrt(r2)
            av += (G * mass[j] * inv * inv * inv) * d
        acc[i] += av


@ti.kernel
def add_acc(acc: ti.template(), acc_g: ti.template(), n: ti.i32):
    """acc += acc_g  (fold a separately-computed gravity field into acc)."""
    for i in range(n):
        acc[i] += acc_g[i]


@ti.kernel
def grav_analytic(pos: ti.template(), acc: ti.template(), n: ti.i32,
                  M_d: ti.f64, a: ti.f64, b: ti.f64,
                  M_h: ti.f64, a_h: ti.f64,
                  is_hernquist: ti.i32, smbh_mass: ti.f64):
    """Add the analytic disk (Miyamoto-Nagai) + halo potential to acc.

    A no-op when ``M_d == M_h == 0`` (the fully-live galaxy case), so it is safe
    to always call after self-gravity regardless of the gravity solver in use.
    """
    for i in range(n):
        pi = pos[i]
        R2 = pi.x * pi.x + pi.y * pi.y
        S = ti.sqrt(pi.z * pi.z + b * b)
        aS = a + S
        invD3 = (R2 + aS * aS) ** (-1.5)
        av = ti.Vector([-G * M_d * pi.x * invD3, -G * M_d * pi.y * invD3,
                        -G * M_d * pi.z * aS * invD3 / S], dt=ti.f64)
        if is_hernquist == 1:
            r = pi.norm()
            r_safe = ti.max(r, 1e-6)
            av += -(G * M_h / ((r_safe + a_h) ** 2 * r_safe)) * pi
        else:
            rh = R2 + pi.z * pi.z + a_h * a_h
            av += -(G * M_h * rh ** (-1.5)) * pi
        if smbh_mass > 0.0:
            r_safe = pi.norm() + 1e-6
            av += -(G * smbh_mass / (r_safe * r_safe * r_safe)) * pi
        acc[i] += av


# ======================= Star formation (Schmidt law, N-independent) ==========

@ti.kernel
def form_stars_schmidt(rho: ti.template(), mass: ti.template(),
                      is_star: ti.template(), birth: ti.template(),
                      t_life: ti.template(), u: ti.template(),
                      n: ti.i32, time: ti.f64, dt: ti.f64,
                      rho_min: ti.f64, eps_ff: ti.f64,
                      t_sn_max: ti.f64, u_max_sf: ti.f64):
    """Schmidt-law star formation: p_sf = eps_ff * dt / t_ff(rho).

    N-independent because the probability depends on local density and dt,
    not on the number of particles.  Each new star gets a random lifetime:
    ~5% are massive (t_life < t_sn_max -> will fire SN), the rest are
    long-lived and simply age.
    """
    for i in range(n):
        # Only form stars in COLD, dense gas (u < u_max_sf)
        if is_star[i] == 0 and rho[i] > rho_min and u[i] < u_max_sf:
            # Free-fall time: t_ff = sqrt(3*pi / (32 * G * rho))
            t_ff = _TFF_COEFF / ti.sqrt(G * ti.max(rho[i], 1e-12))
            prob = eps_ff * dt / ti.max(t_ff, 1e-8)
            prob = ti.min(prob, 0.01)  # cap to prevent massive burst SF in dense cores
            if ti.random(ti.f64) < prob:
                is_star[i] = 1
                birth[i] = time
                # Assign stochastic lifetime from IMF-like distribution:
                # ~5% massive (short-lived, will SN), ~95% long-lived
                r = ti.random(ti.f64)
                if r < 0.05:
                    # Massive star: lifetime 40%-100% of t_sn_max
                    # e.g. t_sn_max=0.05 → 20-50 Myr (40-100 steps at dt=5e-4)
                    # This prevents SNe from firing within the first few steps
                    t_min_massive = 0.4 * t_sn_max
                    t_life[i] = t_min_massive + ti.random(ti.f64) * (t_sn_max - t_min_massive)
                else:
                    # Low/intermediate mass: 0.5 - 12 Gyr (0.5 - 12 code units)
                    t_life[i] = 0.5 + ti.random(ti.f64) * 11.5


# ======================= Supernova marking (stochastic lifetimes) ==============

@ti.kernel
def mark_supernovae(is_star: ti.template(), birth: ti.template(),
                    t_life: ti.template(),
                    sn_done: ti.template(), exploding: ti.template(),
                    evt: ti.template(), fate: ti.template(),
                    n: ti.i32, time: ti.f64, t_sn_max: ti.f64):
    """Flag stars whose age exceeds their assigned lifetime.

    Only stars with t_life < t_sn_max are SN-capable (massive stars).
    Long-lived stars (t_life >> t_sn_max) never explode — they just age.
    ``evt`` accumulates cumulative event counts: evt[0] += supernovae,
    evt[1] += hypernovae (read out for the stats panel).
    """
    for i in range(n):
        exploding[i] = 0
        if is_star[i] == 1 and sn_done[i] == 0:
            age = time - birth[i]
            if age >= t_life[i] and t_life[i] < t_sn_max:
                # Per-step SN rate limiter: only 30% of eligible stars
                # explode each step. This prevents chain reactions where
                # all massive stars detonate simultaneously.
                if ti.random(ti.f64) < 0.3:
                    sn_done[i] = 1
                    if ti.random(ti.f64) < 0.05:
                        exploding[i] = 2  # Hypernova
                        fate[i] = 2       # -> collapses to a black hole
                        ti.atomic_add(evt[1], 1)
                    else:
                        exploding[i] = 1  # Supernova
                        fate[i] = 1       # -> ejected as gas at next rebuild
                        ti.atomic_add(evt[0], 1)


# ======================= Feedback (N-scaled energy budget) =====================

@ti.kernel
def count_neighbors(pos: ti.template(), is_star: ti.template(),
                    exploding: ti.template(), h: ti.template(),
                    n_neigh: ti.template(), n: ti.i32, r_fb: ti.f64):
    """Count gas neighbors within feedback radius for each exploding star."""
    for i in range(n):
        n_neigh[i] = 0
        if exploding[i] > 0:
            pi = pos[i]
            cnt = 0
            for j in range(n):
                if is_star[j] == 0:
                    r = (pos[j] - pi).norm()
                    h_fb = ti.min(h[j] * 2.0, r_fb)
                    if r < h_fb:
                        cnt += 1
            n_neigh[i] = ti.max(cnt, 1)


@ti.kernel
def feedback(pos: ti.template(), vel: ti.template(), u: ti.template(),
             mass: ti.template(), h: ti.template(), is_star: ti.template(),
             exploding: ti.template(), n_neigh: ti.template(),
             n: ti.i32, r_fb: ti.f64, du_sn: ti.f64, v_sn: ti.f64,
             m_ref: ti.f64):
    """Inject SN energy as a fixed budget split among neighbors (N-independent).

    Total energy per SN = du_sn * (m_star / m_ref), split evenly among
    N_neighbor gas particles.  This makes feedback resolution-independent.
    """
    for j in range(n):
        if is_star[j] == 0:
            pj = pos[j]
            for i in range(n):
                if exploding[i] > 0:
                    dir_v = pj - pos[i]
                    r = dir_v.norm()
                    h_fb = ti.min(h[j] * 2.0, r_fb)
                    if r < h_fb:
                        mult = 10.0 if exploding[i] == 2 else 1.0
                        m_scale = mass[i] / ti.max(m_ref, 1e-12)
                        nn = ti.cast(n_neigh[i], ti.f64)
                        # Energy budget split among neighbors
                        du_per = du_sn * mult * m_scale / nn
                        u[j] += du_per
                        # Kinetic kick, also budget-split
                        dir_norm = dir_v / ti.max(r, 1e-6)
                        kick = v_sn * mult * m_scale * (1.0 - r / h_fb) / nn
                        vel[j] += dir_norm * kick


# ======================= Mass return (partial, from SN) =======================

@ti.kernel
def mass_return(pos: ti.template(), mass: ti.template(), h: ti.template(),
                is_star: ti.template(), exploding: ti.template(),
                n_neigh: ti.template(), u: ti.template(), metal: ti.template(),
                n: ti.i32, r_fb: ti.f64, f_return: ti.f64, y_metal: ti.f64):
    """Return a fraction of the exploding star's mass (metal-enriched) to gas.

    The star keeps (1 - f_return) of its mass (remnant: neutron star / BH).
    The returned ejecta is shared among nearby gas particles and carries a metal
    yield ``y_metal`` (fraction of ejecta mass that is newly-synthesised metals),
    raising the local metallicity Z = metal_mass / mass.
    """
    for i in range(n):
        if exploding[i] > 0 and f_return > 0.0:
            m_return = mass[i] * f_return
            nn = ti.cast(n_neigh[i], ti.f64)
            dm_per = m_return / nn
            dz_per = dm_per * y_metal      # metal mass injected per neighbor
            mass[i] *= (1.0 - f_return)    # star becomes remnant
            pi = pos[i]
            for j in range(n):
                if is_star[j] == 0:
                    r = (pos[j] - pi).norm()
                    h_fb = ti.min(h[j] * 2.0, r_fb)
                    if r < h_fb:
                        ti.atomic_add(mass[j], dm_per)
                        ti.atomic_add(metal[j], dz_per)


@ti.kernel
def kick(vel: ti.template(), acc: ti.template(), u: ti.template(),
         du: ti.template(), is_star: ti.template(), n: ti.i32, dt_half: ti.f64):
    for i in range(n):
        vel[i] += acc[i] * dt_half
        if is_star[i] == 0:
            u[i] = ti.max(u[i] + du[i] * dt_half, 1e-3)


@ti.kernel
def drift(pos: ti.template(), vel: ti.template(), n: ti.i32, dt: ti.f64, max_v: ti.f64):
    for i in range(n):
        # Prevent CFL violations / numerical explosions
        v_sq = vel[i].norm_sqr()
        if v_sq > max_v * max_v:
            vel[i] = vel[i] * (max_v / ti.sqrt(v_sq))
        pos[i] += vel[i] * dt


@ti.kernel
def thermal_energy_k(mass: ti.template(), u: ti.template(),
                     is_star: ti.template(), n: ti.i32) -> ti.f64:
    """Total thermal energy of the gas, sum_i m_i u_i (gas only)."""
    e = 0.0
    for i in range(n):
        if is_star[i] == 0:
            e += mass[i] * u[i]
    return e


@ti.kernel
def cfl_min_dt(h: ti.template(), cs: ti.template(), vel: ti.template(),
               acc: ti.template(), is_star: ti.template(), n: ti.i32,
               courant: ti.f64, c_acc: ti.f64, soft: ti.f64) -> ti.f64:
    """Smallest stable timestep over all particles (CFL + acceleration).

    Gas obeys the SPH Courant condition dt < C * h / (c_s + |v|); every particle
    obeys an acceleration condition dt < C_acc * sqrt(eps / |a|).  The engine
    sub-divides each requested step so every sub-step respects this, which keeps
    the collapsing (and therefore densifying) gas stable without shrinking the
    global step everywhere.
    """
    dt_min = 1.0e30
    for i in range(n):
        a_mag = acc[i].norm()
        if a_mag > 1e-12:
            ti.atomic_min(dt_min, c_acc * ti.sqrt(soft / a_mag))
        if is_star[i] == 0:
            signal = cs[i] + vel[i].norm() + 1e-6
            ti.atomic_min(dt_min, courant * h[i] / signal)
    return dt_min


@ti.kernel
def cool_gas(u: ti.template(), is_star: ti.template(), metal: ti.template(),
             mass: ti.template(), n: ti.i32, dt: ti.f64,
             u_floor: ti.f64, inv_tcool: ti.f64, z_ref: ti.f64):
    """Metallicity-dependent radiative cooling on gas only (exp, stable).

    The cooling rate scales with metallicity, inv_t = inv_tcool * (1 + Z/Z_ref):
    metal-enriched gas (downstream of supernovae) radiates faster, so enriched
    regions cool and form stars sooner -- a simple metal-line cooling proxy.
    Lets shock + SN heat radiate away so gas settles into a thin cold disk.
    """
    for i in range(n):
        if is_star[i] == 0:
            excess = u[i] - u_floor
            if excess > 0.0:
                z = metal[i] / ti.max(mass[i], 1e-12)
                inv_t = inv_tcool * (1.0 + z / z_ref)
                u[i] = u_floor + excess * ti.exp(-dt * inv_t)


# ------------------------------------------------- grid-accelerated SPH kernels
# These mirror the N^2 kernels above but visit only the 3x3x3 block of grid cells
# around each particle (see core.solvers.neighbor_grid).  ``gsort`` holds gas
# global indices ordered by cell; ``cell_start[c]..cell_start[c+1]`` is cell c's
# slice.  The grid cell size >= every SPH search radius, so a 3x3x3 stencil is
# exact.  All binned particles are gas, so no is_star check is needed on j.
@ti.func
def _cell_of(pi, lox: ti.f64, loy: ti.f64, loz: ti.f64, inv_cell: ti.f64,
             nx: ti.i32, ny: ti.i32, nz: ti.i32):
    cx = ti.min(ti.max(ti.i32((pi.x - lox) * inv_cell), 0), nx - 1)
    cy = ti.min(ti.max(ti.i32((pi.y - loy) * inv_cell), 0), ny - 1)
    cz = ti.min(ti.max(ti.i32((pi.z - loz) * inv_cell), 0), nz - 1)
    return cx, cy, cz


@ti.kernel
def gas_density_grid(pos: ti.template(), mass: ti.template(),
                     rho: ti.template(), h: ti.template(),
                     is_star: ti.template(), n: ti.i32,
                     gsort: ti.template(), cell_start: ti.template(),
                     nx: ti.i32, ny: ti.i32, nz: ti.i32,
                     lox: ti.f64, loy: ti.f64, loz: ti.f64, inv_cell: ti.f64):
    for i in range(n):
        if is_star[i] == 0:
            acc = 0.0
            pi = pos[i]
            hi = h[i]
            cx, cy, cz = _cell_of(pi, lox, loy, loz, inv_cell, nx, ny, nz)
            for dx in range(-1, 2):
                bx = cx + dx
                if 0 <= bx < nx:
                    for dy in range(-1, 2):
                        by = cy + dy
                        if 0 <= by < ny:
                            for dz in range(-1, 2):
                                bz = cz + dz
                                if 0 <= bz < nz:
                                    c = (bx * ny + by) * nz + bz
                                    for s in range(cell_start[c],
                                                   cell_start[c + 1]):
                                        j = gsort[s]
                                        if is_star[j] == 0:
                                            r = (pos[j] - pi).norm()
                                            if r < 2.0 * hi:
                                                acc += mass[j] * cubic_w(r, hi)
            rho[i] = ti.max(acc, 1e-12)


@ti.kernel
def gas_forces_grid(pos: ti.template(), vel: ti.template(), mass: ti.template(),
                    rho: ti.template(), pressure: ti.template(),
                    cs: ti.template(), h: ti.template(), acc: ti.template(),
                    du: ti.template(), is_star: ti.template(), n: ti.i32,
                    alpha: ti.f64, beta: ti.f64,
                    gsort: ti.template(), cell_start: ti.template(),
                    nx: ti.i32, ny: ti.i32, nz: ti.i32,
                    lox: ti.f64, loy: ti.f64, loz: ti.f64, inv_cell: ti.f64):
    for i in range(n):
        a = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
        dudt = 0.0
        if is_star[i] == 0:
            pi = pos[i]
            vi = vel[i]
            pr_i = pressure[i] / (rho[i] * rho[i])
            cx, cy, cz = _cell_of(pi, lox, loy, loz, inv_cell, nx, ny, nz)
            for dx in range(-1, 2):
                bx = cx + dx
                if 0 <= bx < nx:
                    for dy in range(-1, 2):
                        by = cy + dy
                        if 0 <= by < ny:
                            for dz in range(-1, 2):
                                bz = cz + dz
                                if 0 <= bz < nz:
                                    c = (bx * ny + by) * nz + bz
                                    for s in range(cell_start[c],
                                                   cell_start[c + 1]):
                                        j = gsort[s]
                                        if is_star[j] == 0 and j != i:
                                            dr = pi - pos[j]
                                            r = dr.norm()
                                            hij = 0.5 * (h[i] + h[j])
                                            if 1e-12 < r < 2.0 * hij:
                                                dv = vi - vel[j]
                                                pr_j = pressure[j] / (
                                                    rho[j] * rho[j])
                                                visc = 0.0
                                                dvdr = dv.dot(dr)
                                                if dvdr < 0.0:
                                                    mu = hij * dvdr / (
                                                        r * r + 0.01 * hij * hij)
                                                    c_bar = 0.5 * (cs[i] + cs[j])
                                                    rho_bar = 0.5 * (
                                                        rho[i] + rho[j])
                                                    visc = (-alpha * c_bar * mu
                                                            + beta * mu * mu
                                                            ) / rho_bar
                                                grad = cubic_dwdr(r, hij) * (
                                                    dr / r)
                                                coeff = pr_i + pr_j + visc
                                                a += -mass[j] * coeff * grad
                                                dudt += 0.5 * mass[j] * coeff \
                                                    * dv.dot(grad)
        acc[i] = a
        du[i] = dudt


@ti.kernel
def feedback_grid(pos: ti.template(), vel: ti.template(), u: ti.template(),
                  mass: ti.template(), h: ti.template(), is_star: ti.template(),
                  exploding: ti.template(), n_neigh: ti.template(),
                  n: ti.i32, r_fb: ti.f64,
                  du_sn: ti.f64, v_sn: ti.f64, m_ref: ti.f64,
                  gsort: ti.template(),
                  cell_start: ti.template(), nx: ti.i32, ny: ti.i32, nz: ti.i32,
                  lox: ti.f64, loy: ti.f64, loz: ti.f64, inv_cell: ti.f64):
    """Inject SN energy using grid-accelerated neighbor search, N-scaled."""
    for i in range(n):
        if exploding[i] > 0:
            pi = pos[i]
            mult = 10.0 if exploding[i] == 2 else 1.0
            m_scale = mass[i] / ti.max(m_ref, 1e-12)
            nn = ti.cast(n_neigh[i], ti.f64)
            eff_du = du_sn * mult * m_scale / nn
            eff_v = v_sn * mult * m_scale / nn

            cx, cy, cz = _cell_of(pi, lox, loy, loz, inv_cell, nx, ny, nz)
            for dx in range(-1, 2):
                bx = cx + dx
                if 0 <= bx < nx:
                    for dy in range(-1, 2):
                        by = cy + dy
                        if 0 <= by < ny:
                            for dz in range(-1, 2):
                                bz = cz + dz
                                if 0 <= bz < nz:
                                    c = (bx * ny + by) * nz + bz
                                    for s in range(cell_start[c],
                                                   cell_start[c + 1]):
                                        j = gsort[s]
                                        if is_star[j] == 0:
                                            dir_v = pos[j] - pi
                                            r = dir_v.norm()
                                            h_fb = ti.min(h[j] * 2.0, r_fb)
                                            if r < h_fb:
                                                ti.atomic_add(u[j], eff_du)
                                                dir_norm = dir_v / ti.max(r, 1e-6)
                                                kick = eff_v * (1.0 - r / h_fb)
                                                ti.atomic_add(vel[j][0], dir_norm[0] * kick)
                                                ti.atomic_add(vel[j][1], dir_norm[1] * kick)
                                                ti.atomic_add(vel[j][2], dir_norm[2] * kick)


class LivingGalaxyEngine:
    def __init__(self, pot=None, gamma=5.0 / 3.0, alpha=1.0, beta=2.0,
                 eta=1.3, softening=0.2,
                 sf_density_factor=8.0,
                 # --- N-independent Schmidt-law star formation ---
                 eps_ff=0.01,         # SF efficiency per free-fall time
                 sf_prob=None,        # legacy flat probability (unused if eps_ff set)
                 # --- supernova / feedback ---
                 t_sn=0.03,           # legacy (used as t_sn_max default)
                 t_sn_max=0.05,       # only stars with t_life < this fire SN (~49 Myr)
                 r_fb=0.6, du_sn=400.0, v_sn=50.0,
                 f_return=0.4,        # mass return fraction
                 # --- cooling ---
                 cooling=True, u_floor=60.0, t_cool=0.02,
                 # --- metal enrichment ---
                 z_init=0.001, z_ref=0.02, y_metal=0.1,
                 gravity_mode="direct", theta=0.6,
                 # --- adaptive timestep (CFL sub-stepping) ---
                 adaptive=True, courant=0.3, c_acc=0.25, max_substeps=16,
                 # --- live SMBH: Bondi accretion + AGN feedback + friction ---
                 # c_df kept low so the hole responds to clumps and wanders
                 # instead of locking rigidly to the exact centre.
                 smbh_physics=True, r_acc=0.5, r_df=2.0, c_df=1.0,
                 acc_eff=0.1, u_acc_max=150.0, agn_feedback=True,
                 eps_agn=1.0e-5, v_agn=300.0,
                 # --- dynamic N / baryon cycle ---
                 dynamic_baryons=True, rebuild_every=20, sn_gas_split=4,
                 gas_inflow=True, inflow_radius=40.0, escape_radius=60.0):
        self.pot = pot or dict(M_d=5.0, a=3.0, b=0.3, M_h=12.0, a_h=8.0)
        self.gamma, self.alpha, self.beta, self.eta = gamma, alpha, beta, eta
        self.softening = float(softening)
        self.eps2 = softening ** 2
        self.gravity_mode = gravity_mode
        self.theta = float(theta)
        self._bh = None
        self._grid = None
        self.sf_density_factor = sf_density_factor
        # Schmidt-law SF: eps_ff is primary; sf_prob is legacy fallback
        self.eps_ff = float(eps_ff) if eps_ff is not None else None
        self.sf_prob = float(sf_prob) if sf_prob is not None else None
        # Supernova / feedback
        self.t_sn = float(t_sn)
        self.t_sn_max = float(t_sn_max)
        self.r_fb = float(r_fb)
        self.du_sn, self.v_sn = float(du_sn), float(v_sn)
        self.f_return = float(f_return)
        # Radiative cooling
        self.cooling = bool(cooling)
        self.u_floor = float(u_floor)
        self.inv_tcool = 1.0 / float(t_cool) if t_cool > 0 else 0.0
        # Metal enrichment: initial Z, solar-ref Z for cooling, SN metal yield.
        self.z_init = float(z_init)
        self.z_ref = float(z_ref)
        self.y_metal = float(y_metal)
        # Adaptive timestep
        self.adaptive = bool(adaptive)
        self.courant = float(courant)
        self.c_acc = float(c_acc)
        self.max_substeps = int(max_substeps)
        self.last_substeps = 1
        # Live SMBH (Bondi accretion + Chandrasekhar dynamical friction).
        self.smbh_physics = bool(smbh_physics)
        self.r_acc = float(r_acc)
        self.r_df = float(r_df)
        self.c_df = float(c_df)
        # SMBH update touches the GPU<->host boundary; it is a slow secular
        # process, so run it only every ``smbh_every`` outer steps (with a
        # correspondingly larger dt) to avoid a per-step sync stall.
        self.smbh_every = 4
        self._smbh_accum = 0.0
        self._smbh_count = 0
        self.acc_eff = float(acc_eff)   # fraction of Bondi inflow actually accreted
        self.u_acc_max = float(u_acc_max)  # only gas colder than this is accreted
        self.agn_feedback = bool(agn_feedback)
        self.eps_agn = float(eps_agn)   # AGN thermal coupling (fraction of dM c^2)
        self.v_agn = float(v_agn)       # AGN kinetic outflow velocity scale (km/s)
        self.smbh_idx = -1     # auto-detected in setup (-1 = no SMBH)
        # Cumulative event counters (read from the GPU ``evt`` field): SNe,
        # hypernovae, black holes formed, particles ejected.  Plus SFR state.
        self._prev_star_mass = None
        self._prev_time = 0.0
        self._sfr = 0.0
        # Dynamic N: structural changes queued on host, applied every
        # ``rebuild_every`` steps by re-allocating the Taichi fields.
        self.rebuild_every = int(rebuild_every)
        self._q_remove = []
        self._q_add = {nm: [] for nm in
                       ("pos", "vel", "mass", "u", "h", "birth", "t_life",
                        "metal", "is_star", "sn_done", "is_bh", "fate")}
        # Baryon cycle (dynamic N): SN eject gas, hypernovae form black holes,
        # unbound particles are removed and replaced by fresh infalling gas.
        self.dynamic_baryons = bool(dynamic_baryons)
        self.sn_gas_split = int(sn_gas_split)  # gas particles spawned per supernova
        self.u_ejecta = 4000.0        # hot SN ejecta internal energy
        self.escape_removal = True
        self.escape_radius = float(escape_radius)  # removal only beyond this radius
        self.gas_inflow = bool(gas_inflow)  # replenish removed particles as gas
        self.inflow_radius = float(inflow_radius)  # outer shell for new gas
        self.inflow_rotation = 0.6    # tangential fraction of v_circ
        self.inflow_infall = 0.25     # inward radial fraction of v_circ
        self.m_ejected = 0.0          # cumulative mass removed from the box
        self.m_inflow = 0.0           # cumulative mass added as inflow
        # 4b: newly-formed stars absorb nearby cold gas -> star particles are
        # mass-aggregated clusters (many gas -> one star), which also balances
        # the gas particles spawned by supernovae (population control).
        self.gas_merge = True
        self.sf_absorb = 3            # cold-gas particles a new star absorbs
        self._last_rebuild_time = 0.0
        self._rng = np.random.default_rng(12345)
        self.n = 0
        self.time = 0.0
        self.step_count = 0
        self._f = {}
        self._rho_thresh = 0.0
        self._m_ref = 1e-4  # reference particle mass for N-scaling

    def setup(self, pos, vel, mass, u, species=None):
        """Initialise particles.

        ``species`` (optional) sets each particle's kind in the engine's
        convention -- 0 gas, 1 star, 2 dark matter.  When omitted every particle
        is gas (the classic living-galaxy start, where stars form from gas).
        Pre-existing stars are flagged as having already passed their supernova
        so only newly *formed* stars feed back.
        """
        n = pos.shape[0]
        self.n = n
        f = {name: ti.Vector.field(3, ti.f64, shape=n)
             for name in ("pos", "vel", "acc", "acc_g")}
        for name in ("mass", "u", "du", "rho", "pressure", "cs", "h", "birth",
                     "t_life", "metal"):
            f[name] = ti.field(ti.f64, shape=n)
        for name in ("is_star", "sn_done", "exploding", "n_neigh", "is_bh", "fate"):
            f[name] = ti.field(ti.i32, shape=n)
        # Cumulative event counters [n_sn, n_hypernova, n_bh_formed, n_ejected].
        f["evt"] = ti.field(ti.i32, shape=4)
        self._f = f
        f["evt"].fill(0)
        f["is_bh"].fill(0)
        f["fate"].fill(0)
        f["pos"].from_numpy(np.ascontiguousarray(pos, np.float64))
        f["vel"].from_numpy(np.ascontiguousarray(vel, np.float64))
        f["mass"].from_numpy(np.ascontiguousarray(mass, np.float64))
        f["u"].from_numpy(np.ascontiguousarray(u, np.float64))
        if species is None:
            species = np.zeros(n, np.int32)
        else:
            species = np.ascontiguousarray(species, np.int32)
        f["is_star"].from_numpy(species)
        # Stars present at t=0 are "old" -> mark their SN already done.
        f["sn_done"].from_numpy((species == 1).astype(np.int32))
        f["exploding"].from_numpy(np.zeros(n, np.int32))
        f["n_neigh"].from_numpy(np.zeros(n, np.int32))
        f["birth"].from_numpy(np.zeros(n, np.float64))
        # Old stars get long lifetimes (won't SN); new stars get assigned at birth
        t_life_init = np.where(species == 1, 100.0, 0.0)  # 100 code units = ~98 Gyr
        f["t_life"].from_numpy(t_life_init.astype(np.float64))
        f["h"].from_numpy(np.full(n, 0.4))
        # Metal mass per particle (Z = metal/mass); start at primordial floor.
        f["metal"].from_numpy((self.z_init * np.ascontiguousarray(mass, np.float64)))
        # Reference particle mass for N-scaled feedback
        gas_mask = (species == 0)
        if gas_mask.any():
            self._m_ref = float(np.median(mass[gas_mask]))
        else:
            self._m_ref = float(np.median(mass))
        # Auto-detect a central SMBH: the single most-massive star particle, but
        # only if it dominates (>= 30x the median particle mass).  Scenarios
        # without an SMBH (gas disks, mergers) leave smbh_idx = -1 -> no-op.
        self.smbh_idx = -1
        if self.smbh_physics and n > 1:
            cand = int(np.argmax(mass))
            # A genuine SMBH is the single heaviest particle, is a star, and
            # dominates -- >= 4x the 95th-percentile of every other particle.
            others = np.delete(mass, cand)
            thresh = 4.0 * float(np.percentile(others, 95))
            if species[cand] == 1 and mass[cand] >= thresh:
                self.smbh_idx = cand
                f["is_bh"].from_numpy((np.arange(n) == cand).astype(np.int32))
        self._rebuild_solvers(n)
        self._density_h(20)
        # Star-formation density floor = factor * median gas density
        # (soft floor below which SF is disabled — prevents halo SF)
        rho_arr = f["rho"].to_numpy()
        gas_rho = rho_arr[gas_mask] if gas_mask.any() else rho_arr
        self._rho_thresh = self.sf_density_factor * float(np.median(gas_rho))
        self._forces()

    def _rebuild_solvers(self, n):
        """(Re)create the Barnes-Hut tree and SPH neighbour grid for N particles.

        Barnes-Hut pays off only for large N; tiny systems stay on direct N^2.
        The grid pays off only for large gas counts.  Both are sized to N, so
        they must be rebuilt whenever the particle count changes (dynamic N).
        """
        if self.gravity_mode == "bh" and n >= 256:
            from core.solvers.barnes_hut import BarnesHut
            self._bh = BarnesHut(n, theta=self.theta, softening=self.softening)
        else:
            self._bh = None
        if n >= 2000:
            from core.solvers.neighbor_grid import NeighborGrid
            self._grid = NeighborGrid(n, min_radius=self.r_fb)
        else:
            self._grid = None

    # ------------------------------------------------------- dynamic N (rebuild)
    # Structural changes (particle birth/death) are *queued* on the host during a
    # step and applied in one batch every ``rebuild_every`` steps by
    # ``_apply_structural_changes`` -- which rebuilds the Taichi fields at the new
    # count.  Re-allocating ti.field does not recompile kernels, so this is just a
    # realloc + a light density prime.  Per-step kernels stay fixed-N in between.
    _PRESERVE_VEC = ("pos", "vel")
    _PRESERVE_SCALAR = ("mass", "u", "h", "birth", "t_life", "metal")
    _PRESERVE_INT = ("is_star", "sn_done", "is_bh", "fate")

    def queue_remove(self, idx):
        """Queue particle indices (into the *current* arrays) for deletion."""
        if len(idx):
            self._q_remove.extend(int(i) for i in idx)

    def queue_add(self, fields: dict):
        """Queue new particles. ``fields`` maps every preserved field name to an
        array of equal length (pos/vel are (k,3); the rest (k,))."""
        k = len(fields["pos"])
        if k == 0:
            return
        for name in self._PRESERVE_VEC + self._PRESERVE_SCALAR + self._PRESERVE_INT:
            self._q_add[name].append(np.asarray(fields[name]))

    def _has_queued(self) -> bool:
        return bool(self._q_remove) or bool(self._q_add["pos"])

    def _apply_structural_changes(self):
        f = self._f
        n = self.n
        keep = np.ones(n, dtype=bool)
        if self._q_remove:
            keep[np.asarray(self._q_remove, dtype=np.int64)] = False
        data = {}
        for name in self._PRESERVE_VEC + self._PRESERVE_SCALAR + self._PRESERVE_INT:
            data[name] = f[name].to_numpy()[keep]
        if self._q_add["pos"]:
            for name in data:
                add = np.concatenate(self._q_add[name], axis=0)
                data[name] = np.concatenate([data[name], add], axis=0)
        evt_saved = f["evt"].to_numpy()
        self._q_remove = []
        self._q_add = {nm: [] for nm in
                       (self._PRESERVE_VEC + self._PRESERVE_SCALAR + self._PRESERVE_INT)}
        self._realloc(data, evt_saved)

    def _realloc(self, data: dict, evt_saved):
        """Allocate fresh fields at the new N and load the preserved arrays."""
        n_new = len(data["pos"])
        self.n = n_new
        nf = {nm: ti.Vector.field(3, ti.f64, shape=n_new)
              for nm in ("pos", "vel", "acc", "acc_g")}
        for nm in ("mass", "u", "du", "rho", "pressure", "cs", "h", "birth",
                   "t_life", "metal"):
            nf[nm] = ti.field(ti.f64, shape=n_new)
        for nm in ("is_star", "sn_done", "exploding", "n_neigh", "is_bh", "fate"):
            nf[nm] = ti.field(ti.i32, shape=n_new)
        nf["evt"] = ti.field(ti.i32, shape=4)
        self._f = nf
        for nm in self._PRESERVE_VEC:
            nf[nm].from_numpy(np.ascontiguousarray(data[nm], np.float64))
        for nm in self._PRESERVE_SCALAR:
            nf[nm].from_numpy(np.ascontiguousarray(data[nm], np.float64))
        for nm in self._PRESERVE_INT:
            nf[nm].from_numpy(np.ascontiguousarray(data[nm], np.int32))
        for nm in ("acc", "acc_g"):
            nf[nm].fill(0)
        for nm in ("du", "rho", "pressure", "cs"):
            nf[nm].fill(0.0)
        for nm in ("exploding", "n_neigh"):
            nf[nm].fill(0)
        nf["evt"].from_numpy(np.ascontiguousarray(evt_saved, np.int32))
        self._rebuild_solvers(n_new)
        # Re-derive the primary SMBH index from the carried is_bh flags.
        is_bh = nf["is_bh"].to_numpy()
        bh_idx = np.where(is_bh == 1)[0]
        self.smbh_idx = int(bh_idx[0]) if len(bh_idx) else -1
        self._density_h(iters=3)
        self._forces()

    def _baryon_events(self):
        """Discrete baryon-cycle events, queued for the next rebuild.

        4a SN -> hot gas ejecta (remove star, spawn ``sn_gas_split`` gas);
        4c hypernova -> black hole (flag is_bh in place);
        4d unbound escapers removed; 4e replaced by fresh infalling gas.
        """
        f = self._f
        fate = f["fate"].to_numpy()
        is_star = f["is_star"].to_numpy()
        is_bh = f["is_bh"].to_numpy()
        pos = f["pos"].to_numpy()
        vel = f["vel"].to_numpy()
        mass = f["mass"].to_numpy()
        metal = f["metal"].to_numpy()
        u = f["u"].to_numpy()
        birth = f["birth"].to_numpy()
        rng = self._rng
        flags_changed = False

        # --- 4c: hypernova -> black hole ---
        hyper = np.where((fate == 2) & (is_star == 1) & (is_bh == 0))[0]
        if len(hyper):
            is_bh[hyper] = 1
            fate[hyper] = 0
            evt = f["evt"].to_numpy()
            evt[2] += len(hyper)
            f["evt"].from_numpy(evt)
            flags_changed = True

        # --- 4a: supernova -> hot gas ejecta ---
        sne = np.where((fate == 1) & (is_star == 1) & (is_bh == 0))[0]
        if len(sne):
            K = max(int(self.sn_gas_split), 1)
            ns = len(sne)
            d = rng.normal(0.0, 1.0, (ns, K, 3))
            d /= (np.linalg.norm(d, axis=2, keepdims=True) + 1e-9)
            newpos = pos[sne][:, None, :] + d * 0.3
            newvel = vel[sne][:, None, :] + d * self.v_sn
            m_per = np.repeat(mass[sne] / K, K)
            z_per = np.repeat((metal[sne] + self.y_metal * mass[sne]) / K, K)
            kt = ns * K
            self.queue_remove(sne)
            self.queue_add(dict(
                pos=newpos.reshape(-1, 3), vel=newvel.reshape(-1, 3),
                mass=m_per, u=np.full(kt, self.u_ejecta),
                h=np.full(kt, self.softening), birth=np.zeros(kt),
                t_life=np.zeros(kt), metal=z_per,
                is_star=np.zeros(kt, np.int32), sn_done=np.zeros(kt, np.int32),
                is_bh=np.zeros(kt, np.int32), fate=np.zeros(kt, np.int32)))
            fate[sne] = 0
            flags_changed = True

        # --- 4b: newly-formed stars absorb nearby cold gas (many gas -> one
        #     star cluster).  Conserves mass + momentum; balances SN gas spawning. ---
        if self.gas_merge and self.sf_absorb > 0:
            young = np.where((is_star == 1) & (is_bh == 0)
                             & (birth > self._last_rebuild_time))[0]
            if len(young):
                young = young[:200]                 # bound the per-rebuild cost
                gas_all = np.where(is_star == 0)[0]
                cold = u[gas_all] < self.u_acc_max
                gas_all = gas_all[cold]
                consumed = np.zeros(len(pos), bool)
                r_merge = 3.0 * self.r_acc
                changed = False
                for s in young:
                    avail = gas_all[~consumed[gas_all]]
                    if len(avail) == 0:
                        break
                    dd = np.linalg.norm(pos[avail] - pos[s], axis=1)
                    near = avail[dd < r_merge]
                    if len(near) == 0:
                        continue
                    if len(near) > self.sf_absorb:
                        near = near[np.argsort(np.linalg.norm(
                            pos[near] - pos[s], axis=1))[:self.sf_absorb]]
                    m_new = mass[s] + mass[near].sum()
                    # momentum-conserving velocity, mass + metals aggregated
                    vel[s] = (mass[s] * vel[s] + (mass[near, None] * vel[near]).sum(0)) / m_new
                    mass[s] = m_new
                    metal[s] += metal[near].sum()
                    consumed[near] = True
                    changed = True
                if changed:
                    self.queue_remove(np.where(consumed)[0])
                    f["mass"].from_numpy(np.ascontiguousarray(mass))
                    f["metal"].from_numpy(np.ascontiguousarray(metal))
                    f["vel"].from_numpy(np.ascontiguousarray(vel))

        # --- 4d/4e: escaper removal + inflow replenishment ---
        # Only with fully live gravity: the unbound test uses the particle mass
        # sum, which is wrong when an external analytic potential dominates.
        analytic = (self.pot.get("M_d", 0.0) > 0.0
                    or self.pot.get("M_h", 0.0) > 0.0)
        if self.escape_removal and not analytic:
            r = np.linalg.norm(pos, axis=1)
            v2 = np.einsum("ij,ij->i", vel, vel)
            outward = np.einsum("ij,ij->i", pos, vel) > 0.0
            unbound = 0.5 * v2 > (G * mass.sum() / np.maximum(r, 1e-6))
            # Only remove unbound *baryons* (gas/stars); leave the DM halo and
            # black holes intact so the halo isn't silently converted to gas.
            esc = np.where((r > self.escape_radius) & outward & unbound
                           & (is_bh == 0) & (is_star != 2))[0]
            if len(esc):
                self.m_ejected += float(mass[esc].sum())
                self.queue_remove(esc)
                evt = f["evt"].to_numpy()
                evt[3] += len(esc)
                f["evt"].from_numpy(evt)
                if self.gas_inflow:
                    self._queue_inflow(len(esc), pos, mass, is_star)

        if flags_changed:
            f["fate"].from_numpy(np.ascontiguousarray(fate, np.int32))
            f["is_bh"].from_numpy(np.ascontiguousarray(is_bh, np.int32))
            bh = np.where(is_bh == 1)[0]
            if len(bh):
                self.smbh_idx = int(bh[np.argmax(mass[bh])])
        # Mark this as the reference time for "newly-formed" stars next rebuild.
        self._last_rebuild_time = self.time

    def _queue_inflow(self, count, pos, mass, is_star):
        """Spawn ``count`` fresh gas particles on an outer shell, co-rotating +
        infalling -- a cosmological/CGM accretion source that keeps N bounded."""
        rng = self._rng
        R = self.inflow_radius
        m_enc = float(mass[np.linalg.norm(pos, axis=1) < R].sum())
        vc = float(np.sqrt(G * max(m_enc, 1e-6) / max(R, 1e-6)))
        d = rng.normal(0.0, 1.0, (count, 3))
        d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        newpos = d * (R * rng.uniform(0.9, 1.1, (count, 1)))
        rhat = newpos / (np.linalg.norm(newpos, axis=1, keepdims=True) + 1e-9)
        ephi = np.cross(np.array([0.0, 0.0, 1.0]), rhat)
        ephi /= (np.linalg.norm(ephi, axis=1, keepdims=True) + 1e-9)
        v = (ephi * (self.inflow_rotation * vc)
             - rhat * (self.inflow_infall * vc)
             + rng.normal(0.0, 0.05 * vc, (count, 3)))
        gas = is_star == 0
        mgas = float(np.median(mass[gas])) if gas.any() else float(np.median(mass))
        self.m_inflow += mgas * count
        self.queue_add(dict(
            pos=newpos, vel=v, mass=np.full(count, mgas),
            u=np.full(count, 300.0), h=np.full(count, 0.4),
            birth=np.zeros(count), t_life=np.zeros(count),
            metal=np.full(count, self.z_init * mgas),
            is_star=np.zeros(count, np.int32), sn_done=np.zeros(count, np.int32),
            is_bh=np.zeros(count, np.int32), fate=np.zeros(count, np.int32)))

    def _grid_ready(self) -> bool:
        return self._grid is not None and self._grid.ngas > 0

    def _gas_density(self):
        """Density estimate; rebuilds the neighbour grid first when enabled."""
        f = self._f
        ng = self._grid.rebuild(f["pos"], f["h"], f["is_star"]) \
            if self._grid is not None else 0
        if ng > 0:
            g = self._grid
            gas_density_grid(f["pos"], f["mass"], f["rho"], f["h"], f["is_star"],
                             self.n, g.gsort, g.cell_start, g.nx, g.ny, g.nz,
                             g.lo[0], g.lo[1], g.lo[2], g.inv_cell)
        else:
            gas_density(f["pos"], f["mass"], f["rho"], f["h"], f["is_star"],
                        self.n)

    def _density_h(self, iters=3):
        f = self._f
        for _ in range(iters):
            # 1. Update SPH smoothing lengths from old density
            # Enforce min_h to prevent 1/h^4 force singularities in dense clumps
            min_h = self.softening * 0.25
            update_h_gas(f["mass"], f["rho"], f["h"], f["is_star"], self.n, self.eta, min_h)
            self._gas_density()

    def _forces(self):
        f = self._f
        gas_pressure(f["rho"], f["u"], f["pressure"], f["cs"], f["is_star"],
                     self.n, self.gamma)
        if self._grid_ready():
            g = self._grid
            gas_forces_grid(f["pos"], f["vel"], f["mass"], f["rho"],
                            f["pressure"], f["cs"], f["h"], f["acc"], f["du"],
                            f["is_star"], self.n, self.alpha, self.beta,
                            g.gsort, g.cell_start, g.nx, g.ny, g.nz,
                            g.lo[0], g.lo[1], g.lo[2], g.inv_cell)
        else:
            gas_forces(f["pos"], f["vel"], f["mass"], f["rho"], f["pressure"],
                       f["cs"], f["h"], f["acc"], f["du"], f["is_star"], self.n,
                       self.alpha, self.beta)
        # Self-gravity over all particles (gas + stars + DM).
        if self._bh is not None:
            self._bh.compute(f["pos"], f["mass"], f["acc_g"])
            add_acc(f["acc"], f["acc_g"], self.n)
        else:
            grav_self_direct(f["pos"], f["mass"], f["acc"], self.n, self.eps2)
        # Analytic disk+halo potential (a no-op when M_d == M_h == 0).
        grav_analytic(f["pos"], f["acc"], self.n,
                      self.pot["M_d"], self.pot["a"], self.pot["b"],
                      self.pot["M_h"], self.pot["a_h"],
                      int(self.pot.get("is_hernquist", 0)),
                      float(self.pot.get("smbh_mass", 0.0)))

    def cfl_dt(self):
        """The largest stable timestep right now (CFL + acceleration limited)."""
        f = self._f
        return float(cfl_min_dt(f["h"], f["cs"], f["vel"], f["acc"],
                                f["is_star"], self.n, self.courant,
                                self.c_acc, self.softening))

    def step(self, dt):
        """Advance by total time ``dt``, sub-dividing for CFL stability.

        The number of sub-steps adapts to the current state (more where the gas
        is dense/fast), so the run stays stable through collapse while keeping
        the caller's snapshot cadence (one ``step`` == one frame of length dt).
        """
        n_sub = 1
        if self.adaptive:
            stable = self.cfl_dt()
            if stable > 0.0:
                n_sub = int(np.ceil(dt / stable))
                n_sub = max(1, min(n_sub, self.max_substeps))
        self.last_substeps = n_sub
        sub = dt / n_sub
        for _ in range(n_sub):
            self._substep(sub)
        # SMBH accretion/friction: accumulate dt and apply every smbh_every steps.
        if self.smbh_physics and self.smbh_idx >= 0:
            self._smbh_accum += dt
            self._smbh_count += 1
            if self._smbh_count >= self.smbh_every:
                self._update_smbh(self._smbh_accum)
                self._smbh_accum = 0.0
                self._smbh_count = 0
        self.step_count += 1
        # At the rebuild boundary: gather discrete baryon-cycle events (SN->gas,
        # hypernova->BH, escaper removal + inflow), then apply all queued
        # births/deaths in one batch (dynamic N).
        if self.rebuild_every > 0 and self.step_count % self.rebuild_every == 0:
            if self.dynamic_baryons:
                self._baryon_events()
            if self._has_queued():
                self._apply_structural_changes()

    def _update_smbh(self, dt):
        """Grow + move EVERY black hole (the seed SMBH and any formed by
        hypernovae) by Bondi accretion, AGN feedback and dynamical friction.

        A cheap host-side update run once per ``smbh_every`` outer steps:
          * Bondi-Hoyle accretion of nearby COLD gas (mass conserved: gas shrinks,
            the hole grows), with AGN thermal + kinetic feedback that self-regulates;
          * Chandrasekhar dynamical friction so each hole sinks/orbits realistically.
        Holes are processed in turn on shared host arrays, so gas drained by one is
        seen depleted by the next.
        """
        f = self._f
        is_bh = f["is_bh"].to_numpy()
        bh_list = np.where(is_bh == 1)[0]
        if len(bh_list) == 0:
            return
        pos = f["pos"].to_numpy()
        vel = f["vel"].to_numpy()
        mass = f["mass"].to_numpy()
        is_star = f["is_star"].to_numpy()
        rho = f["rho"].to_numpy()
        cs = f["cs"].to_numpy()
        u = f["u"].to_numpy()
        gas = (is_star == 0)
        changed_mass = changed_vel = changed_u = False

        for i in bh_list:
            p0, v0, M = pos[i], vel[i].copy(), float(mass[i])
            d = pos - p0
            r = np.linalg.norm(d, axis=1)

            # --- Bondi accretion from nearby COLD gas ---
            near = gas & (r < self.r_acc) & (r > 1e-9) & (u < self.u_acc_max)
            if near.any():
                rho_loc = float(np.median(rho[near]))
                cs_loc = float(np.median(cs[near])) + 1e-6
                v_rel = float(np.linalg.norm(v0 - vel[near].mean(axis=0)))
                denom = (cs_loc * cs_loc + v_rel * v_rel) ** 1.5
                mdot = self.acc_eff * 4.0 * np.pi * G * G * M * M * rho_loc / max(denom, 1e-12)
                dM = min(mdot * dt, 0.05 * float(mass[near].sum()))
                if dM > 0.0:
                    idx_near = np.where(near)[0]
                    w = mass[near] / mass[near].sum()
                    take = np.minimum(dM * w, 0.95 * mass[near])
                    mass[idx_near] -= take
                    M_acc = float(take.sum())
                    mass[i] += M_acc
                    changed_mass = True
                    # AGN thermal + kinetic feedback (self-regulating).
                    if self.agn_feedback and self.eps_agn > 0.0:
                        C2 = 299792.458 ** 2
                        E_fb = self.eps_agn * M_acc * C2
                        m_near = float(mass[idx_near].sum())
                        if m_near > 0.0:
                            u[idx_near] += E_fb / m_near
                            changed_u = True
                            if self.v_agn > 0.0:
                                dirs = pos[idx_near] - p0
                                rr = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
                                vel[idx_near] += (dirs / rr) * (self.v_agn * (M_acc / m_near))
                                changed_vel = True

            # --- Chandrasekhar dynamical friction from background (stars + DM) ---
            bg = (~gas) & (r < self.r_df) & (r > 1e-9)
            bg[i] = False
            speed = float(np.linalg.norm(v0))
            if bg.any() and speed > 1e-6:
                vol = (4.0 / 3.0) * np.pi * self.r_df ** 3
                rho_bg = float(mass[bg].sum()) / vol
                a_df = -self.c_df * 4.0 * np.pi * G * G * mass[i] * rho_bg \
                    / (speed ** 3) * v0
                dv = a_df * dt
                dv_mag = float(np.linalg.norm(dv))
                if dv_mag > speed:        # friction only decelerates
                    dv *= speed / dv_mag
                vel[i] = v0 + dv
                changed_vel = True

        if changed_mass:
            f["mass"].from_numpy(np.ascontiguousarray(mass))
        if changed_vel:
            f["vel"].from_numpy(np.ascontiguousarray(vel))
        if changed_u:
            f["u"].from_numpy(np.ascontiguousarray(u))

    def _substep(self, dt):
        f = self._f
        half = 0.5 * dt
        kick(f["vel"], f["acc"], f["u"], f["du"], f["is_star"], self.n, half)
        drift(f["pos"], f["vel"], self.n, dt, 1000.0)
        self.time += dt
        # h is near-converged between (small) sub-steps, so 2 Jacobi iterations
        # track it -- cheaper than the 3 used for a cold start.
        self._density_h(iters=2)

        # --- Star formation (Schmidt law or legacy flat probability) ---
        u_max_sf = 150.0  # Cold gas threshold for SF (prevents SN-heated gas from forming stars)
        if self.eps_ff is not None and self.eps_ff > 0:
            form_stars_schmidt(f["rho"], f["mass"], f["is_star"], f["birth"],
                               f["t_life"], f["u"], self.n, self.time, dt,
                               self._rho_thresh, self.eps_ff, self.t_sn_max, u_max_sf)
        elif self.sf_prob is not None and self.sf_prob > 0:
            # Legacy fallback: flat probability per step (not N-independent)
            _form_stars_legacy(f["rho"], f["is_star"], f["birth"],
                               f["t_life"], f["u"], self.n, self.time,
                               self._rho_thresh, self.sf_prob, self.t_sn_max, u_max_sf)

        # --- Supernova marking (stochastic lifetimes) ---
        mark_supernovae(f["is_star"], f["birth"], f["t_life"],
                        f["sn_done"], f["exploding"], f["evt"], f["fate"],
                        self.n, self.time, self.t_sn_max)

        # --- Count neighbors for budget-split feedback ---
        count_neighbors(f["pos"], f["is_star"], f["exploding"], f["h"],
                        f["n_neigh"], self.n, self.r_fb)

        # --- Feedback (N-scaled energy budget) ---
        if self._grid_ready():
            g = self._grid
            feedback_grid(f["pos"], f["vel"], f["u"], f["mass"],
                          f["h"], f["is_star"], f["exploding"], f["n_neigh"],
                          self.n, self.r_fb, self.du_sn, self.v_sn,
                          self._m_ref, g.gsort, g.cell_start,
                          g.nx, g.ny, g.nz, g.lo[0], g.lo[1], g.lo[2],
                          g.inv_cell)
        else:
            feedback(f["pos"], f["vel"], f["u"], f["mass"],
                     f["h"], f["is_star"], f["exploding"], f["n_neigh"],
                     self.n, self.r_fb, self.du_sn, self.v_sn, self._m_ref)

        # --- Mass return (partial, from SN) ---
        # With the dynamic baryon cycle on, the exploding star is converted to
        # hot ejecta gas at the next rebuild (4a), which returns its mass -- so
        # the in-place mass_return is skipped to avoid double-counting.
        if self.f_return > 0 and not self.dynamic_baryons:
            mass_return(f["pos"], f["mass"], f["h"], f["is_star"],
                        f["exploding"], f["n_neigh"], f["u"], f["metal"],
                        self.n, self.r_fb, self.f_return, self.y_metal)

        self._forces()
        kick(f["vel"], f["acc"], f["u"], f["du"], f["is_star"], self.n, half)
        if self.cooling and self.inv_tcool > 0.0:
            cool_gas(f["u"], f["is_star"], f["metal"], f["mass"], self.n, dt,
                     self.u_floor, self.inv_tcool, self.z_ref)

    # ------------------------------------------------------------ diagnostics
    def energies(self):
        """Return (kinetic, potential, total mechanical) energy in code units.

        Potential energy is the exact softened direct-N^2 sum (a diagnostic, so
        accuracy beats speed); kinetic is over all particles.  ``total`` here is
        the mechanical KE + PE (thermal is reported separately).
        """
        from core.solvers.gravity import kinetic_energy, potential_energy
        f = self._f
        ke = float(kinetic_energy(f["vel"], f["mass"], self.n))
        pe = float(potential_energy(f["pos"], f["mass"], self.n, G, self.eps2))
        return ke, pe, ke + pe

    def thermal_energy(self):
        f = self._f
        return float(thermal_energy_k(f["mass"], f["u"], f["is_star"], self.n))

    def toomre_q(self, r_min=1.0, r_max=15.0, nbins=8):
        """Median stellar Toomre Q over the disk, Q = sigma_R * kappa / (3.36 G Sigma).

        Q < 1 is locally unstable (clumps/spurs), Q ~ 1-2 grows spiral arms/bars,
        Q >> 2 is featureless and hot.  Returned per-frame so the user can see
        whether the disk is in the spiral-forming regime.  None when too few
        disk stars exist yet (e.g. a proto-galaxy that has not formed a disk).
        """
        is_star = self._f["is_star"].to_numpy()
        disk = is_star == 1
        if self.smbh_idx >= 0:
            disk[self.smbh_idx] = False
        if int(disk.sum()) < 50:
            return None
        pos = self._f["pos"].to_numpy()[disk]
        vel = self._f["vel"].to_numpy()[disk]
        mass = self._f["mass"].to_numpy()[disk]
        R = np.hypot(pos[:, 0], pos[:, 1])
        eRx, eRy = pos[:, 0] / (R + 1e-9), pos[:, 1] / (R + 1e-9)
        vR = vel[:, 0] * eRx + vel[:, 1] * eRy
        vT = -vel[:, 0] * eRy + vel[:, 1] * eRx
        edges = np.linspace(r_min, r_max, nbins + 1)
        qs = []
        for k in range(nbins):
            m = (R >= edges[k]) & (R < edges[k + 1])
            if int(m.sum()) < 20:
                continue
            area = np.pi * (edges[k + 1] ** 2 - edges[k] ** 2)
            sigma = mass[m].sum() / area
            sig_R = float(np.std(vR[m]))
            v_c = abs(float(np.mean(vT[m])))
            r_mid = 0.5 * (edges[k] + edges[k + 1])
            kappa = np.sqrt(2.0) * v_c / r_mid
            if sigma > 0.0 and kappa > 0.0:
                qs.append(sig_R * kappa / (3.36 * G * sigma))
        return float(np.median(qs)) if qs else None

    def rotation_speed(self, r0=8.0, dr=1.5):
        """Mean tangential speed of baryons in an annulus at R~r0 (rotation curve).

        A single sample of the rotation curve (km/s) at a reference radius, so the
        user can watch the disk spin up / settle.  None if the annulus is empty.
        """
        is_star = self._f["is_star"].to_numpy()
        bary = is_star != 2
        if self.smbh_idx >= 0:
            bary[self.smbh_idx] = False
        pos = self._f["pos"].to_numpy()
        R = np.hypot(pos[:, 0], pos[:, 1])
        m = bary & (np.abs(R - r0) < dr)
        if int(m.sum()) < 10:
            return None
        vel = self._f["vel"].to_numpy()
        Rm = R[m] + 1e-9
        vT = (-vel[m, 0] * pos[m, 1] + vel[m, 1] * pos[m, 0]) / Rm
        return float(np.mean(np.abs(vT)))

    def diagnostics(self):
        """Energies + virial ratio Q = 2(KE+U_therm)/|PE| (Q ~ 1 in equilibrium).

        The virial ratio is only reported for fully live-gravity runs: with an
        external analytic potential (M_d/M_h != 0) the particle-only PE is
        incomplete, so 2(KE+U)/|PE| would be meaningless and is omitted.
        """
        ke, pe, _ = self.energies()
        u_th = self.thermal_energy()
        out = dict(kinetic=ke, potential=pe, thermal=u_th,
                   total=ke + pe + u_th, substeps=self.last_substeps)
        analytic = (self.pot.get("M_d", 0.0) > 0.0
                    or self.pot.get("M_h", 0.0) > 0.0)
        if not analytic and pe != 0.0:
            out["virial"] = 2.0 * (ke + u_th) / abs(pe)
        q = self.toomre_q()
        if q is not None:
            out["toomre_q"] = q
        v_rot = self.rotation_speed()
        if v_rot is not None:
            out["v_rot"] = v_rot

        # --- gas / star budget (cheap host reductions) ---
        is_star = self._f["is_star"].to_numpy()
        mass = self._f["mass"].to_numpy()
        metal = self._f["metal"].to_numpy()
        gas = is_star == 0
        star = is_star == 1
        m_gas = float(mass[gas].sum())
        m_star = float(mass[star].sum())
        out["gas_count"] = int(gas.sum())
        out["star_count"] = int(star.sum())
        out["gas_mass"] = m_gas
        out["star_mass"] = m_star
        out["gas_fraction"] = m_gas / (m_gas + m_star) if (m_gas + m_star) > 0 else 0.0
        # Black-hole mass + box mass budget (source/sink terms) for conservation.
        is_bh = self._f["is_bh"].to_numpy()
        out["bh_count"] = int(is_bh.sum())
        out["bh_mass"] = float(mass[is_bh == 1].sum()) if is_bh.any() else 0.0
        out["m_ejected"] = self.m_ejected
        out["m_inflow"] = self.m_inflow
        if gas.any():
            out["gas_metal"] = float((metal[gas].sum()) / max(m_gas, 1e-12))

        # --- star-formation rate: d(star mass)/dt since the last call ---
        if self._prev_star_mass is not None and self.time > self._prev_time:
            out["sfr"] = max(m_star - self._prev_star_mass, 0.0) / (self.time - self._prev_time)
        else:
            out["sfr"] = 0.0
        self._prev_star_mass = m_star
        self._prev_time = self.time

        # --- cumulative event counters [n_sn, n_hypernova, n_bh, n_ejected] ---
        evt = self._f["evt"].to_numpy()
        out["n_sn"] = int(evt[0])
        out["n_hypernova"] = int(evt[1])
        out["n_bh_formed"] = int(evt[2])
        out["n_ejected"] = int(evt[3])

        if self.smbh_idx >= 0:
            out["smbh_mass"] = float(mass[self.smbh_idx])
        return out

    def get(self, name):
        return self._f[name].to_numpy()

    def star_count(self):
        return int((self._f["is_star"].to_numpy() == 1).sum())

    def ages(self):
        is_star = self._f["is_star"].to_numpy()
        birth = self._f["birth"].to_numpy()
        return np.where(is_star == 1, self.time - birth, -1.0)

    def metallicity(self):
        """Per-particle metallicity Z = metal_mass / mass (dimensionless)."""
        mass = self._f["mass"].to_numpy()
        return self._f["metal"].to_numpy() / np.maximum(mass, 1e-12)

    def to_state(self):
        """Export current state as a core.State (gas + star particle types)."""
        from core.state import State, PTYPE_GAS, PTYPE_STAR, PTYPE_DM
        n = self.n
        is_star = self._f["is_star"].to_numpy()
        # Engine species (0 gas, 1 star, 2 DM) -> State ptype codes.
        ptype = np.full(n, PTYPE_GAS, np.int32)
        ptype[is_star == 1] = PTYPE_STAR
        ptype[is_star == 2] = PTYPE_DM
        return State(
            pos=self.get("pos"), vel=self.get("vel"), mass=self.get("mass"),
            ptype=ptype, ids=np.arange(n, dtype=np.int64),
            time=self.time, step=self.step_count,
            u=self.get("u"), rho=self.get("rho"), age=self.ages(),
            metallicity=self.metallicity(),
            meta={"engine": "living_galaxy"},
        )


# ======================= Legacy star formation kernel (flat probability) ======

@ti.kernel
def _form_stars_legacy(rho: ti.template(), is_star: ti.template(),
                      birth: ti.template(), t_life: ti.template(), u: ti.template(),
                      n: ti.i32, time: ti.f64,
                      rho_thresh: ti.f64, prob: ti.f64, t_sn_max: ti.f64, u_max_sf: ti.f64):
    """Legacy flat-probability star formation (for backward compat)."""
    for i in range(n):
        if is_star[i] == 0 and rho[i] > rho_thresh and u[i] < u_max_sf:
            prob_capped = ti.min(prob, 0.01)
            if ti.random(ti.f64) < prob_capped:
                is_star[i] = 1
                birth[i] = time
                r = ti.random(ti.f64)
                if r < 0.05:
                    t_min_massive = 0.4 * t_sn_max
                    t_life[i] = t_min_massive + ti.random(ti.f64) * (t_sn_max - t_min_massive)
                else:
                    t_life[i] = 0.5 + ti.random(ti.f64) * 11.5
