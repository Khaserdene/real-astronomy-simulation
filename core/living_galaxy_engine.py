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
def update_h_gas(rho: ti.template(), mass: ti.template(), h: ti.template(),
                 is_star: ti.template(), n: ti.i32, eta: ti.f64):
    for i in range(n):
        if is_star[i] == 0:
            h[i] = eta * (mass[i] / rho[i]) ** (1.0 / 3.0)


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
def grav_ext(pos: ti.template(), mass: ti.template(), acc: ti.template(),
             n: ti.i32, eps2: ti.f64, M_d: ti.f64, a: ti.f64, b: ti.f64,
             M_h: ti.f64, a_h: ti.f64):
    """Add self-gravity (all particles) + analytic disk+halo potential to acc."""
    for i in range(n):
        av = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
        pi = pos[i]
        for j in range(n):
            d = pos[j] - pi
            r2 = d.dot(d) + eps2
            inv = 1.0 / ti.sqrt(r2)
            av += (G * mass[j] * inv * inv * inv) * d
        R2 = pi.x * pi.x + pi.y * pi.y
        S = ti.sqrt(pi.z * pi.z + b * b)
        aS = a + S
        invD3 = (R2 + aS * aS) ** (-1.5)
        av += ti.Vector([-G * M_d * pi.x * invD3, -G * M_d * pi.y * invD3,
                         -G * M_d * pi.z * aS * invD3 / S], dt=ti.f64)
        rh = R2 + pi.z * pi.z + a_h * a_h
        av += -(G * M_h * rh ** (-1.5)) * pi
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
            exploding[i] = 1


@ti.kernel
def feedback(pos: ti.template(), u: ti.template(), is_star: ti.template(),
             exploding: ti.template(), n: ti.i32, r_fb: ti.f64, du_sn: ti.f64):
    """Inject thermal energy into gas around each exploding star."""
    for j in range(n):
        if is_star[j] == 0:
            pj = pos[j]
            add = 0.0
            for i in range(n):
                if exploding[i] == 1:
                    if (pos[i] - pj).norm() < r_fb:
                        add += du_sn
            if add > 0.0:
                u[j] += add


@ti.kernel
def kick(vel: ti.template(), acc: ti.template(), u: ti.template(),
         du: ti.template(), is_star: ti.template(), n: ti.i32, dt_half: ti.f64):
    for i in range(n):
        vel[i] += acc[i] * dt_half
        if is_star[i] == 0:
            u[i] = ti.max(u[i] + du[i] * dt_half, 1e-3)


@ti.kernel
def drift(pos: ti.template(), vel: ti.template(), n: ti.i32, dt: ti.f64):
    for i in range(n):
        pos[i] += vel[i] * dt


class LivingGalaxyEngine:
    def __init__(self, pot=None, gamma=5.0 / 3.0, alpha=1.0, beta=2.0,
                 eta=1.3, softening=0.2,
                 sf_density_factor=8.0, sf_prob=0.05,
                 t_sn=0.02, r_fb=0.6, du_sn=400.0):
        self.pot = pot or dict(M_d=5.0, a=3.0, b=0.3, M_h=12.0, a_h=8.0)
        self.gamma, self.alpha, self.beta, self.eta = gamma, alpha, beta, eta
        self.eps2 = softening ** 2
        self.sf_density_factor = sf_density_factor
        self.sf_prob, self.t_sn, self.r_fb, self.du_sn = sf_prob, t_sn, r_fb, du_sn
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
             for name in ("pos", "vel", "acc")}
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
        self._density_h(20)
        # Star-formation threshold = factor * median gas density.
        self._rho_thresh = self.sf_density_factor * float(
            np.median(f["rho"].to_numpy()))
        self._forces()

    def _density_h(self, iters=3):
        f = self._f
        for _ in range(iters):
            gas_density(f["pos"], f["mass"], f["rho"], f["h"], f["is_star"], self.n)
            update_h_gas(f["rho"], f["mass"], f["h"], f["is_star"], self.n, self.eta)
        gas_density(f["pos"], f["mass"], f["rho"], f["h"], f["is_star"], self.n)

    def _forces(self):
        f = self._f
        gas_pressure(f["rho"], f["u"], f["pressure"], f["cs"], f["is_star"],
                     self.n, self.gamma)
        gas_forces(f["pos"], f["vel"], f["mass"], f["rho"], f["pressure"],
                   f["cs"], f["h"], f["acc"], f["du"], f["is_star"], self.n,
                   self.alpha, self.beta)
        grav_ext(f["pos"], f["mass"], f["acc"], self.n, self.eps2,
                 self.pot["M_d"], self.pot["a"], self.pot["b"],
                 self.pot["M_h"], self.pot["a_h"])

    def step(self, dt):
        f = self._f
        half = 0.5 * dt
        kick(f["vel"], f["acc"], f["u"], f["du"], f["is_star"], self.n, half)
        drift(f["pos"], f["vel"], self.n, dt)
        self.time += dt
        self._density_h()
        form_stars(f["rho"], f["is_star"], f["birth"], self.n, self.time,
                   self._rho_thresh, self.sf_prob)
        mark_supernovae(f["is_star"], f["birth"], f["sn_done"], f["exploding"],
                        self.n, self.time, self.t_sn)
        feedback(f["pos"], f["u"], f["is_star"], f["exploding"], self.n,
                 self.r_fb, self.du_sn)
        self._forces()
        kick(f["vel"], f["acc"], f["u"], f["du"], f["is_star"], self.n, half)
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
