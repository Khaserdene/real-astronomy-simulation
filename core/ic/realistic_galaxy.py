"""Realistic Isolated Galaxy ICs with Toomre-Q stabilized disk, Hernquist halo/bulge, and SMBH.

The proto-galaxy mode builds a warm, turbulent gas cloud that collapses into a
disk.  Its morphology (the initial substructure) and kinematics (how much it
spins vs. falls in) are fully tunable:

  * ``noise_pattern`` -- which kind of substructure the cloud starts with
    (clumps / spiral / filaments / shells / smooth);
  * ``rotation_frac`` -- the fraction of the circular speed given as ordered
    rotation (0 = pure radial collapse, ~1 = centrifugally supported);
  * ``turbulence`` / ``radial_infall`` -- random and inward motions seeding the
    collapse.

Circular speeds are computed from the *softened* enclosed-mass profile and
capped, so no particle near the centre (or near the SMBH) is launched at an
unphysical speed -- this is what previously made the cloud "explode".
"""
import numpy as np
from core.ic.disk import _sample_exponential_disk, _sample_plummer_positions
from core.units import G
from core.ic.plummer import _random_directions
from core.ic.noise import fbm3, domain_warp

# Named noise patterns for the proto-galaxy gas cloud.  Exposed as an integer in
# the editor (float-only param UI) -- this dict documents the mapping.
NOISE_PATTERNS = {
    0: "clumps",      # multi-octave fractal blobs (classic)
    1: "spiral",      # two-arm logarithmic spiral over-densities
    2: "filaments",   # sharp fractal -> a thin cosmic-web-like network
    3: "shells",      # concentric radial shells
    4: "smooth",      # no substructure (uniform Plummer cloud)
}


def _sample_hernquist(rng, n, a, r_max_factor=8.0):
    """Hernquist positions, clamped so the long tail can't fling particles to
    absurd radii (raw inverse-CDF diverges as u -> 1)."""
    u = rng.uniform(0.0, 0.999, n)
    sq = np.sqrt(u)
    r = a * sq / (1.0 - sq)
    np.clip(r, 0.0, r_max_factor * a, out=r)
    return _random_directions(rng, n) * r[:, None]


def _sample_exp_sech2(rng, n, Rd, z0):
    u = rng.random(n)
    x = np.linspace(0.0, 30.0, 100000)
    cdf = 1.0 - (1.0 + x) * np.exp(-x)
    R = np.interp(u, cdf, x) * Rd
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    u_z = rng.uniform(0.001, 0.999, n)
    z = z0 * np.arctanh(2.0 * u_z - 1.0)
    return np.column_stack((R * np.cos(phi), R * np.sin(phi), z))


def _simple_fractal_noise(pos, scale, octaves=3):
    noise = np.zeros(len(pos))
    for i in range(int(octaves)):
        freq = (2 ** i) * scale
        amp = 1.0 / (2 ** i)
        noise += amp * np.sin(pos[:, 0] * freq + pos[:, 1] * freq * 1.3 + pos[:, 2] * freq * 1.7)
        noise += amp * np.cos(pos[:, 0] * freq * 0.8 - pos[:, 1] * freq * 1.1 + pos[:, 2] * freq * 0.9)
    noise = (noise + 2.0) / 4.0
    return np.clip(noise, 0.0, 1.0)


def _perlin_field(pos, halo_scale, noise_scale, octaves, seed,
                  frequency, lacunarity, persistence, warp):
    """Multi-octave Perlin/fBm field in [0,1] over the cloud (optionally warped)."""
    base = (2.0 / halo_scale) * max(noise_scale, 1e-3)
    p = pos
    if warp > 0.0:
        p = domain_warp(pos, seed=int(seed) + 7,
                        strength=warp * halo_scale * 0.1, frequency=base * 0.5)
    return fbm3(p, seed=int(seed), octaves=octaves, frequency=base * frequency,
                lacunarity=lacunarity, persistence=persistence)


def _clump_acceptance(pos, pattern, noise_scale, octaves, sharpness, halo_scale,
                      seed=0, frequency=1.0, lacunarity=2.0, persistence=0.5,
                      warp=0.0):
    """Acceptance probability in [0, 1] that shapes the proto-cloud morphology.

    The proto cloud is built by over-sampling a Plummer sphere and keeping each
    candidate with probability ``_clump_acceptance(...)``, so this field IS the
    initial gas density pattern.  ``pattern`` selects the morphology (see
    :data:`NOISE_PATTERNS`); the structure is carved from real Perlin/fBm noise
    (:mod:`core.ic.noise`) controlled by ``seed``, ``frequency``, ``lacunarity``,
    ``persistence`` and an optional domain ``warp``.  ``sharpness`` sets contrast.
    """
    pattern = int(pattern)

    def field():
        return _perlin_field(pos, halo_scale, noise_scale, octaves, seed,
                             frequency, lacunarity, persistence, warp)

    if pattern == 4:  # smooth
        return np.ones(len(pos))

    if pattern == 1:  # spiral arms (Perlin break-up modulates the arms)
        R = np.hypot(pos[:, 0], pos[:, 1]) + 1e-6
        phi = np.arctan2(pos[:, 1], pos[:, 0])
        winding = 3.0 * max(noise_scale, 1e-3)
        arm = 0.5 * (np.cos(2.0 * phi - winding * np.log(R + 1.0)) + 1.0)
        breakup = 0.5 + 0.5 * field()
        return np.clip(arm * breakup, 0.0, 1.0) ** sharpness

    if pattern == 3:  # concentric shells, jittered by noise
        base = (2.0 / halo_scale) * max(noise_scale, 1e-3)
        r = np.linalg.norm(pos, axis=1)
        shells = 0.5 * (np.sin(r * base * 4.0 + 3.0 * field()) + 1.0)
        return np.clip(shells, 0.0, 1.0) ** sharpness

    # 0 = clumps (default), 2 = filaments (a much sharper fractal)
    noise = field()
    power = sharpness if pattern == 0 else sharpness + 3.0
    return np.clip(noise, 0.0, 1.0) ** power


def _compute_vcirc(pos, mass, smbh_mass, halo_pot=None, soft=0.5, v_max=600.0):
    """Softened, capped circular speed from the enclosed-mass profile.

    Enclosed mass is ranked on the true radius, but the rotation speed divides by
    a *softened* radius ``sqrt(r^2 + soft^2)`` and is capped at ``v_max`` so that
    particles near the centre (or near the SMBH point mass) are never launched at
    runaway speeds.  Dropping this softening is what made the proto cloud explode.
    """
    r = np.linalg.norm(pos, axis=1)
    order = np.argsort(r)
    m_cum = np.empty_like(mass)
    m_cum[order] = np.cumsum(mass[order])

    # Add SMBH
    m_cum += smbh_mass

    # Add analytic halo if needed (Hernquist cumulative mass)
    if halo_pot is not None:
        M_h, a_h = halo_pot['M_h'], halo_pot['a_h']
        m_cum += M_h * (r ** 2) / ((r + a_h) ** 2)

    r_soft = np.sqrt(r * r + soft * soft)
    v_circ = np.sqrt(G * m_cum / r_soft)
    return np.minimum(v_circ, v_max)


def make_realistic_galaxy(n=40000, seed=0,
                          live_halo=True,
                          smbh_mass=0.1,    # 1e9 Msun
                          disk_mass=4.0, disk_scale=3.0, disk_height=0.3,
                          gas_mass=1.0,
                          bulge_mass=1.0, bulge_scale=1.0,
                          halo_mass=40.0, halo_scale=15.0,
                          toomre_q=1.5,
                          proto=False,
                          # --- equilibrium / resolution ---
                          halo_warmth=1.0,        # 1.0 = virial dispersion (Q~1)
                          gas_particle_frac=0.25,  # min fraction of N that is gas
                          # --- proto-galaxy morphology (Perlin/fBm substructure) ---
                          noise_pattern=0, noise_scale=1.0, noise_octaves=4,
                          clump_sharpness=3.0, noise_seed=0, noise_frequency=1.0,
                          noise_lacunarity=2.0, noise_persistence=0.5, noise_warp=0.0,
                          # --- proto-galaxy kinematics (collapse vs. spin) ---
                          rotation_frac=0.8, turbulence=40.0, radial_infall=0.0,
                          vcirc_softening=0.5):
    """Generates a realistic galaxy with SMBH, Hernquist Halo/Bulge, Exp+sech2 Disk, and Gas."""
    rng = np.random.default_rng(seed)

    total_m = disk_mass + gas_mass + bulge_mass + (halo_mass if live_halo else 0.0)

    n_gas = max(int(n * (gas_mass / total_m)), 1)
    n_disk = max(int(n * (disk_mass / total_m)), 1) if not proto else 0
    n_bulge = max(int(n * (bulge_mass / total_m)), 1)
    n_halo = max(int(n * (halo_mass / total_m)), 1) if live_halo else 0
    n_smbh = 1

    # Boost gas particle resolution: gas is a small mass fraction but SPH needs
    # many gas particles to resolve density/star formation.  Take the extra
    # count from the (over-resolved) halo; gas particles just get a smaller mass.
    if not proto and gas_particle_frac > 0.0 and n_halo > 1:
        n_gas_target = int(gas_particle_frac * n)
        if n_gas_target > n_gas:
            extra = min(n_gas_target - n_gas, n_halo - 1)
            n_gas += extra
            n_halo -= extra

    # In proto mode, the stellar disk and bulge masses become part of the gas mass
    if proto:
        n_gas += n_disk + n_bulge
        gas_mass += disk_mass + bulge_mass
        disk_mass = 0.0
        n_disk = 0
        bulge_mass = 0.0
        n_bulge = 0

    # 1. Positions
    pos_smbh = np.zeros((n_smbh, 3))

    pos_bulge = _sample_hernquist(rng, n_bulge, bulge_scale) if n_bulge > 0 else np.zeros((0, 3))
    pos_disk = _sample_exp_sech2(rng, n_disk, disk_scale, disk_height) if n_disk > 0 else np.zeros((0, 3))

    if proto:
        # Proto-galaxy gas is a warm, slowly rotating cloud (Plummer-like) whose
        # substructure is carved out by the chosen noise pattern.
        if n_gas > 0:
            raw_pos = _sample_plummer_positions(rng, n_gas * 3, halo_scale * 0.5)
            accept_prob = _clump_acceptance(
                raw_pos, noise_pattern, noise_scale, noise_octaves,
                clump_sharpness, halo_scale, seed=noise_seed,
                frequency=noise_frequency, lacunarity=noise_lacunarity,
                persistence=noise_persistence, warp=noise_warp)
            rand_vals = rng.uniform(0, 1, len(raw_pos))
            accepted = raw_pos[rand_vals < accept_prob]

            if len(accepted) < n_gas:
                pos_gas = np.vstack((accepted, raw_pos[:n_gas - len(accepted)]))
            else:
                pos_gas = accepted[:n_gas]
        else:
            pos_gas = np.zeros((0, 3))
    else:
        pos_gas = _sample_exp_sech2(rng, n_gas, disk_scale * 1.5, disk_height * 0.5) if n_gas > 0 else np.zeros((0, 3))

    pos_halo = _sample_hernquist(rng, n_halo, halo_scale) if n_halo > 0 else np.zeros((0, 3))

    pos = np.vstack((pos_smbh, pos_bulge, pos_disk, pos_gas, pos_halo))

    # 2. Masses
    m_smbh = np.array([smbh_mass])
    m_bulge = np.full(n_bulge, bulge_mass / n_bulge) if n_bulge > 0 else np.array([])
    m_disk = np.full(n_disk, disk_mass / n_disk) if n_disk > 0 else np.array([])
    m_gas = np.full(n_gas, gas_mass / n_gas) if n_gas > 0 else np.array([])
    m_halo = np.full(n_halo, halo_mass / n_halo) if n_halo > 0 else np.array([])

    mass = np.concatenate((m_smbh, m_bulge, m_disk, m_gas, m_halo))

    # 3. Velocities
    vel = np.zeros_like(pos)

    halo_pot = None if live_halo else {'M_h': halo_mass, 'a_h': halo_scale}
    v_circ = _compute_vcirc(pos, mass, smbh_mass, halo_pot=halo_pot,
                            soft=vcirc_softening)

    idx = 1  # skip SMBH
    inv_sqrt3 = 1.0 / np.sqrt(3.0)

    if n_bulge > 0:
        # Isotropic Maxwellian at the local virial dispersion sigma = v_c/sqrt(3)
        # (3 independent components -> <|v|^2> = v_c^2): pressure-supported in
        # equilibrium rather than artificially cold.
        sig_b = (halo_warmth * inv_sqrt3 * v_circ[idx:idx + n_bulge])[:, None]
        vel[idx:idx + n_bulge] = rng.normal(0.0, 1.0, (n_bulge, 3)) * sig_b
        idx += n_bulge

    if n_disk > 0:
        p_d = pos[idx:idx + n_disk]
        vc_d = v_circ[idx:idx + n_disk]
        R_d_arr = np.linalg.norm(p_d[:, :2], axis=1) + 1e-6
        e_R = np.column_stack((p_d[:, 0] / R_d_arr, p_d[:, 1] / R_d_arr, np.zeros(n_disk)))
        e_phi = np.column_stack((-p_d[:, 1] / R_d_arr, p_d[:, 0] / R_d_arr, np.zeros(n_disk)))

        Sigma = (disk_mass / (2 * np.pi * disk_scale ** 2)) * np.exp(-R_d_arr / disk_scale)
        kappa = np.sqrt(2.0) * vc_d / R_d_arr

        sigma_R = (3.36 * G * Sigma * toomre_q) / (kappa + 1e-6)
        sigma_R = np.clip(sigma_R, 0.05 * vc_d, 0.5 * vc_d)

        sigma_phi = sigma_R / np.sqrt(2.0)
        sigma_z = np.sqrt(2 * np.pi * G * Sigma * disk_height)

        d_R = rng.normal(0.0, sigma_R)
        d_phi = rng.normal(0.0, sigma_phi)
        d_z = rng.normal(0.0, sigma_z)

        vel[idx:idx + n_disk] = (e_R * d_R[:, None] + e_phi * (vc_d + d_phi)[:, None])
        vel[idx:idx + n_disk, 2] = d_z
        idx += n_disk

    if n_gas > 0:
        p_g = pos[idx:idx + n_gas]
        vc_g = v_circ[idx:idx + n_gas]
        R_g = np.linalg.norm(p_g[:, :2], axis=1) + 1e-6
        e_phi = np.column_stack((-p_g[:, 1] / R_g, p_g[:, 0] / R_g, np.zeros(n_gas)))
        if proto:
            # Ordered rotation: a fraction of the circular speed (rotation_frac=0
            # is pure radial collapse, ~1 is centrifugally supported).
            vel[idx:idx + n_gas] = e_phi * (vc_g * rotation_frac)[:, None]

            # Optional inward kick to seed/accelerate the collapse.
            if radial_infall != 0.0:
                r3 = np.linalg.norm(p_g, axis=1) + 1e-6
                e_r = p_g / r3[:, None]
                vel[idx:idx + n_gas] -= e_r * (vc_g * radial_infall)[:, None]

            # Turbulence modulated by noise -> clumps get localized vorticity.
            if turbulence > 0.0:
                ns = 3.0 / halo_scale
                turb_x = _simple_fractal_noise(p_g, scale=ns) - 0.5
                turb_y = _simple_fractal_noise(p_g + np.array([10., 0, 0]), scale=ns) - 0.5
                turb_z = _simple_fractal_noise(p_g + np.array([0, 10., 0]), scale=ns) - 0.5
                turb_vel = np.column_stack((turb_x, turb_y, turb_z)) * turbulence
                vel[idx:idx + n_gas] += turb_vel
        else:
            vel[idx:idx + n_gas] = e_phi * vc_g[:, None]
        idx += n_gas

    if n_halo > 0:
        # Live DM halo at its virial dispersion (was 0.5 v_c -> sub-virial and
        # made the whole galaxy contract).  sigma = v_c/sqrt(3) per component
        # gives <|v|^2> = v_c^2, so the halo supports itself and Q ~ 1.
        sig_h = (halo_warmth * inv_sqrt3 * v_circ[idx:idx + n_halo])[:, None]
        vel[idx:idx + n_halo] = rng.normal(0.0, 1.0, (n_halo, 3)) * sig_h
        idx += n_halo

    # Remove net drift (ignoring SMBH which stays at 0,0,0)
    if len(mass) > 1:
        vel[1:] -= np.average(vel[1:], axis=0, weights=mass[1:])

    # 4. Species
    species = np.zeros(len(pos), dtype=np.int32)
    sp_idx = 0
    species[sp_idx] = 1  # SMBH
    sp_idx += 1
    if n_bulge > 0:
        species[sp_idx:sp_idx + n_bulge] = 1
        sp_idx += n_bulge
    if n_disk > 0:
        species[sp_idx:sp_idx + n_disk] = 1
        sp_idx += n_disk
    if n_gas > 0:
        species[sp_idx:sp_idx + n_gas] = 0
        sp_idx += n_gas
    if n_halo > 0:
        species[sp_idx:sp_idx + n_halo] = 2
        sp_idx += n_halo

    # 5. Internal Energy (u)
    u = np.full(len(pos), 200.0)
    if n_gas > 0:
        gas_mask = (species == 0)
        gamma = 5.0 / 3.0
        sound_speed = 32.0 if proto else 12.0
        u[gas_mask] = sound_speed ** 2 / (gamma * (gamma - 1.0))

    return pos, vel, mass, u, species
