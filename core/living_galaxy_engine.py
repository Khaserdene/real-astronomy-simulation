"""Living-galaxy engine: gas + stars with star formation and SN feedback.

Gas and star particles share one field set.  All particles feel gravity and the
analytic galaxy potential; only gas particles carry SPH pressure.  Over time:

  * star formation  -- dense gas particles convert in place into star particles
    (type flips, formation time recorded);
  * stellar ageing  -- a star's age = current time - birth time drives its colour
    at render (young = hot/blue, old = cool/red);
  * supernova feedback -- a star, a short time after birth, injects thermal energy
    into the surrounding gas, heating and pushing it (bubbles / outflows).

This is the "alive" galaxy: gas collapses, lights up as young stars, and those
stars blow the gas back out.  Self-contained Taichi kernels (reusing the SPH
spline helpers) -- no ``from __future__`` so kernel annotations stay live.
"""
import numpy as np
import taichi as ti

from core.units import G
from core.solvers.sph import cubic_w, cubic_dwdr


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


@ti.kernel
def form_stars(rho: ti.template(), is_star: ti.template(), birth: ti.template(),
               n: ti.i32, time: ti.f64, rho_thresh: ti.f64, prob: ti.f64):
    """Convert dense gas particles into stars (in place) with probability."""
    for i in range(n):
        if is_star[i] == 0 and rho[i] > rho_thresh:
            if ti.random(ti.f64) < prob:
                is_star[i] = 1
                birth[i] = time


@ti.kernel
def mark_supernovae(is_star: ti.template(), birth: ti.template(),
                    sn_done: ti.template(), exploding: ti.template(),
                    n: ti.i32, time: ti.f64, t_sn: ti.f64):
    """Flag stars crossing their supernova age this step."""
    for i in range(n):
        exploding[i] = 0
        if is_star[i] == 1 and sn_done[i] == 0 and (time - birth[i]) >= t_sn:
            sn_done[i] = 1
            if ti.random(ti.f64) < 0.05:
                exploding[i] = 2  # Hypernova
            else:
                exploding[i] = 1  # Supernova


@ti.kernel
def feedback(pos: ti.template(), vel: ti.template(), u: ti.template(), h: ti.template(), is_star: ti.template(),
             exploding: ti.template(), n: ti.i32, r_fb: ti.f64, du_sn: ti.f64, v_sn: ti.f64):
    """Inject thermal and kinetic energy into gas around each exploding star using adaptive radius."""
    for j in range(n):
        if is_star[j] == 0:
            pj = pos[j]
            h_fb = ti.min(h[j] * 2.0, r_fb)
            for i in range(n):
                if exploding[i] > 0:
                    dir = pj - pos[i]
                    r = dir.norm()
                    if r < h_fb:
                        mult = 10.0 if exploding[i] == 2 else 1.0
                        u[j] += du_sn * mult
                        
                        dir_norm = dir / ti.max(r, 1e-6)
                        kick = v_sn * mult * (1.0 - r / h_fb)
                        vel[j] += dir_norm * kick


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
def return_mass(is_star: ti.template(), exploding: ti.template(), sn_done: ti.template(), u: ti.template(), n: ti.i32):
    """Return exploding stars back to gas (mass return) with reset thermal energy."""
    for i in range(n):
        if exploding[i] > 0:
            is_star[i] = 0
            sn_done[i] = 0
            u[i] = 100.0  # Hot recycled gas


@ti.kernel
def cool_gas(u: ti.template(), is_star: ti.template(), n: ti.i32, dt: ti.f64,
             u_floor: ti.f64, inv_tcool: ti.f64):
    """Radiative cooling on gas only: relax u toward a floor (exp, stable).

    Lets shock + supernova heat radiate away so gas settles into a thin cold
    disk instead of being puffed up and dispersed.  Skips stars/DM (is_star!=0).
    """
    decay = ti.exp(-dt * inv_tcool)
    for i in range(n):
        if is_star[i] == 0:
            excess = u[i] - u_floor
            if excess > 0.0:
                u[i] = u_floor + excess * decay


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
def feedback_grid(pos: ti.template(), vel: ti.template(), u: ti.template(), h: ti.template(), is_star: ti.template(),
                  exploding: ti.template(), n: ti.i32, r_fb: ti.f64,
                  du_sn: ti.f64, v_sn: ti.f64, gsort: ti.template(),
                  cell_start: ti.template(), nx: ti.i32, ny: ti.i32, nz: ti.i32,
                  lox: ti.f64, loy: ti.f64, loz: ti.f64, inv_cell: ti.f64):
    """Inject SN energy and momentum into nearby gas, walking the grid from each exploder."""
    for i in range(n):
        if exploding[i] > 0:
            pi = pos[i]
            mult = 10.0 if exploding[i] == 2 else 1.0
            eff_du = du_sn * mult
            eff_v = v_sn * mult
            
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
                                            dir = pos[j] - pi
                                            r = dir.norm()
                                            h_fb = ti.min(h[j] * 2.0, r_fb)
                                            if r < h_fb:
                                                ti.atomic_add(u[j], eff_du)
                                                dir_norm = dir / ti.max(r, 1e-6)
                                                kick = eff_v * (1.0 - r / h_fb)
                                                ti.atomic_add(vel[j][0], dir_norm[0] * kick)
                                                ti.atomic_add(vel[j][1], dir_norm[1] * kick)
                                                ti.atomic_add(vel[j][2], dir_norm[2] * kick)


class LivingGalaxyEngine:
    def __init__(self, pot=None, gamma=5.0 / 3.0, alpha=1.0, beta=2.0,
                 eta=1.3, softening=0.2,
                 sf_density_factor=8.0, sf_prob=0.05,
                 t_sn=0.02, r_fb=0.6, du_sn=400.0, v_sn=50.0,
                 cooling=True, u_floor=60.0, t_cool=0.02,
                 gravity_mode="direct", theta=0.6):
        self.pot = pot or dict(M_d=5.0, a=3.0, b=0.3, M_h=12.0, a_h=8.0)
        self.gamma, self.alpha, self.beta, self.eta = gamma, alpha, beta, eta
        self.softening = float(softening)
        self.eps2 = softening ** 2
        # Self-gravity solver: "direct" (N^2) or "bh" (Barnes-Hut treecode).
        self.gravity_mode = gravity_mode
        self.theta = float(theta)
        self._bh = None
        # Uniform-grid neighbour search for the SPH kernels (built for large N).
        self._grid = None
        self.sf_density_factor = sf_density_factor
        self.sf_prob, self.t_sn, self.r_fb = sf_prob, t_sn, r_fb
        self.du_sn, self.v_sn = du_sn, v_sn
        # Radiative cooling (gas only): relax u toward u_floor on timescale t_cool.
        self.cooling = bool(cooling)
        self.u_floor = float(u_floor)
        self.inv_tcool = 1.0 / float(t_cool) if t_cool > 0 else 0.0
        self.n = 0
        self.time = 0.0
        self.step_count = 0
        self._f = {}
        self._rho_thresh = 0.0

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
        for name in ("mass", "u", "du", "rho", "pressure", "cs", "h", "birth"):
            f[name] = ti.field(ti.f64, shape=n)
        for name in ("is_star", "sn_done", "exploding"):
            f[name] = ti.field(ti.i32, shape=n)
        self._f = f
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
        f["birth"].from_numpy(np.zeros(n, np.float64))
        f["h"].from_numpy(np.full(n, 0.4))
        # Barnes-Hut pays off only for large N; small systems stay on direct N^2.
        if self.gravity_mode == "bh" and n >= 256:
            from core.solvers.barnes_hut import BarnesHut
            self._bh = BarnesHut(n, theta=self.theta, softening=self.softening)
        else:
            self._bh = None
        # Grid neighbour search pays off only for large gas counts; small runs
        # keep the exact N^2 SPH kernels (no binning overhead, no edge cases).
        if n >= 2000:
            from core.solvers.neighbor_grid import NeighborGrid
            self._grid = NeighborGrid(n, min_radius=self.r_fb)
        else:
            self._grid = None
        self._density_h(20)
        # Star-formation threshold = factor * median gas density.
        self._rho_thresh = self.sf_density_factor * float(
            np.median(f["rho"].to_numpy()))
        self._forces()

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
        # SPH pressure forces reuse the grid the last density pass rebuilt
        # (positions are unchanged since then); stars are filtered inside.
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

    def step(self, dt):
        f = self._f
        half = 0.5 * dt
        kick(f["vel"], f["acc"], f["u"], f["du"], f["is_star"], self.n, half)
        drift(f["pos"], f["vel"], self.n, dt, 1000.0)
        self.time += dt
        self._density_h()
        form_stars(f["rho"], f["is_star"], f["birth"], self.n, self.time,
                   self._rho_thresh, self.sf_prob)
        mark_supernovae(f["is_star"], f["birth"], f["sn_done"], f["exploding"],
                        self.n, self.time, self.t_sn)
        
        if self._grid_ready():
            g = self._grid
            feedback_grid(f["pos"], f["vel"], f["u"], f["h"], f["is_star"], f["exploding"], self.n,
                          self.r_fb, self.du_sn, self.v_sn, g.gsort, g.cell_start,
                          g.nx, g.ny, g.nz, g.lo[0], g.lo[1], g.lo[2],
                          g.inv_cell)
        else:
            feedback(f["pos"], f["vel"], f["u"], f["h"], f["is_star"], f["exploding"], self.n,
                     self.r_fb, self.du_sn, self.v_sn)
                     
        # Mass return: Exploded stars become gas again
        return_mass(f["is_star"], f["exploding"], f["sn_done"], f["u"], self.n)
        
        self._forces()
        kick(f["vel"], f["acc"], f["u"], f["du"], f["is_star"], self.n, half)
        if self.cooling and self.inv_tcool > 0.0:
            cool_gas(f["u"], f["is_star"], self.n, dt, self.u_floor, self.inv_tcool)
        self.step_count += 1

    def get(self, name):
        return self._f[name].to_numpy()

    def star_count(self):
        return int((self._f["is_star"].to_numpy() == 1).sum())

    def ages(self):
        is_star = self._f["is_star"].to_numpy()
        birth = self._f["birth"].to_numpy()
        return np.where(is_star == 1, self.time - birth, -1.0)

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
            meta={"engine": "living_galaxy"},
        )
