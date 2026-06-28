"""Realistic Isolated Galaxy ICs with Toomre-Q stabilized disk, Hernquist halo/bulge, and SMBH."""
import numpy as np
from core.ic.disk import _sample_exponential_disk, _sample_plummer_positions
from core.units import G
from core.ic.plummer import _random_directions

def _sample_hernquist(rng, n, a):
    u = rng.uniform(0.0, 0.999, n)
    r = a * np.sqrt(u) / (1.0 - np.sqrt(u))
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
    for i in range(octaves):
        freq = (2**i) * scale
        amp = 1.0 / (2**i)
        noise += amp * np.sin(pos[:,0]*freq + pos[:,1]*freq*1.3 + pos[:,2]*freq*1.7)
        noise += amp * np.cos(pos[:,0]*freq*0.8 - pos[:,1]*freq*1.1 + pos[:,2]*freq*0.9)
    noise = (noise + 2.0) / 4.0 
    return np.clip(noise, 0.0, 1.0)

def _compute_vcirc(pos, mass, smbh_mass, halo_pot=None):
    """Enclosed mass v_circ from particles + SMBH + optional analytic halo."""
    r_sph = np.linalg.norm(pos, axis=1) + 1e-6
    order = np.argsort(r_sph)
    m_cum = np.empty_like(mass)
    m_cum[order] = np.cumsum(mass[order])
    
    # Add SMBH
    m_cum += smbh_mass
    
    # Add analytic halo if needed
    if halo_pot is not None:
        M_h, a_h = halo_pot['M_h'], halo_pot['a_h']
        # Hernquist cumulative mass: M(r) = M_h * r^2 / (r + a_h)^2
        m_cum += M_h * (r_sph**2) / ((r_sph + a_h)**2)
        
    v_circ = np.sqrt(G * m_cum / r_sph)
    return v_circ

def make_realistic_galaxy(n=40000, seed=0,
                          live_halo=True,
                          smbh_mass=0.1,    # 1e9 Msun
                          disk_mass=4.0, disk_scale=3.0, disk_height=0.3,
                          gas_mass=1.0, 
                          bulge_mass=1.0, bulge_scale=1.0,
                          halo_mass=40.0, halo_scale=15.0,
                          toomre_q=1.5,
                          proto=False):
    """Generates a realistic galaxy with SMBH, Hernquist Halo/Bulge, Exp+sech2 Disk, and Gas."""
    rng = np.random.default_rng(seed)
    
    total_m = disk_mass + gas_mass + bulge_mass + (halo_mass if live_halo else 0.0)
    
    n_gas = max(int(n * (gas_mass / total_m)), 1)
    n_disk = max(int(n * (disk_mass / total_m)), 1) if not proto else 0
    n_bulge = max(int(n * (bulge_mass / total_m)), 1)
    n_halo = max(int(n * (halo_mass / total_m)), 1) if live_halo else 0
    n_smbh = 1

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
    
    pos_bulge = _sample_hernquist(rng, n_bulge, bulge_scale) if n_bulge > 0 else np.zeros((0,3))
    pos_disk = _sample_exp_sech2(rng, n_disk, disk_scale, disk_height) if n_disk > 0 else np.zeros((0,3))
        
    if proto:
        # Proto-galaxy gas is a warm, slowly rotating cloud (Plummer-like) with clumpy noise
        from core.ic.disk import _sample_plummer_positions
        if n_gas > 0:
            raw_pos = _sample_plummer_positions(rng, n_gas * 3, halo_scale * 0.5)
            noise_vals = _simple_fractal_noise(raw_pos, scale=2.0 / halo_scale, octaves=4)
            accept_prob = noise_vals ** 3  # Power of 3 creates sharper clumps
            rand_vals = rng.uniform(0, 1, len(raw_pos))
            accepted = raw_pos[rand_vals < accept_prob]
            
            if len(accepted) < n_gas:
                pos_gas = np.vstack((accepted, raw_pos[:n_gas - len(accepted)]))
            else:
                pos_gas = accepted[:n_gas]
        else:
            pos_gas = np.zeros((0,3))
    else:
        pos_gas = _sample_exp_sech2(rng, n_gas, disk_scale * 1.5, disk_height * 0.5) if n_gas > 0 else np.zeros((0,3))
        
    pos_halo = _sample_hernquist(rng, n_halo, halo_scale) if n_halo > 0 else np.zeros((0,3))

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
    v_circ = _compute_vcirc(pos, mass, smbh_mass, halo_pot=halo_pot)
    
    idx = 1 # skip SMBH
    
    if n_bulge > 0:
        v_b = v_circ[idx:idx+n_bulge]
        vel[idx:idx+n_bulge] = _random_directions(rng, n_bulge) * rng.normal(0.0, 0.6 * v_b)[:, None]
        idx += n_bulge
        
    if n_disk > 0:
        p_d = pos[idx:idx+n_disk]
        vc_d = v_circ[idx:idx+n_disk]
        R_d_arr = np.linalg.norm(p_d[:, :2], axis=1) + 1e-6
        e_R = np.column_stack((p_d[:, 0] / R_d_arr, p_d[:, 1] / R_d_arr, np.zeros(n_disk)))
        e_phi = np.column_stack((-p_d[:, 1] / R_d_arr, p_d[:, 0] / R_d_arr, np.zeros(n_disk)))
        
        Sigma = (disk_mass / (2 * np.pi * disk_scale**2)) * np.exp(-R_d_arr / disk_scale)
        kappa = np.sqrt(2.0) * vc_d / R_d_arr
        
        sigma_R = (3.36 * G * Sigma * toomre_q) / (kappa + 1e-6)
        sigma_R = np.clip(sigma_R, 0.05 * vc_d, 0.5 * vc_d)
        
        sigma_phi = sigma_R / np.sqrt(2.0)
        sigma_z = np.sqrt(2 * np.pi * G * Sigma * disk_height)
        
        d_R = rng.normal(0.0, sigma_R)
        d_phi = rng.normal(0.0, sigma_phi)
        d_z = rng.normal(0.0, sigma_z)
        
        vel[idx:idx+n_disk] = (e_R * d_R[:, None] + e_phi * (vc_d + d_phi)[:, None])
        vel[idx:idx+n_disk, 2] = d_z
        idx += n_disk
        
    if n_gas > 0:
        p_g = pos[idx:idx+n_gas]
        vc_g = v_circ[idx:idx+n_gas]
        R_g = np.linalg.norm(p_g[:, :2], axis=1) + 1e-6
        e_phi = np.column_stack((-p_g[:, 1] / R_g, p_g[:, 0] / R_g, np.zeros(n_gas)))
        if proto:
            # 80% of circular velocity so it collapses into a disk over time
            vel[idx:idx+n_gas] = e_phi * (vc_g * 0.8)[:, None]
            
            # Add turbulence modulated by noise to give clumps their own localized vorticity
            turb_x = _simple_fractal_noise(p_g, scale=3.0 / halo_scale) - 0.5
            turb_y = _simple_fractal_noise(p_g + np.array([10., 0, 0]), scale=3.0 / halo_scale) - 0.5
            turb_z = _simple_fractal_noise(p_g + np.array([0, 10., 0]), scale=3.0 / halo_scale) - 0.5
            turb_vel = np.column_stack((turb_x, turb_y, turb_z)) * 40.0
            vel[idx:idx+n_gas] += turb_vel
        else:
            vel[idx:idx+n_gas] = e_phi * vc_g[:, None]
        idx += n_gas
        
    if n_halo > 0:
        v_h = v_circ[idx:idx+n_halo]
        vel[idx:idx+n_halo] = _random_directions(rng, n_halo) * np.abs(rng.normal(0.0, 0.5 * v_h[:, None]))
        idx += n_halo
        
    # Remove net drift (ignoring SMBH which stays at 0,0,0)
    if len(mass) > 1:
        vel[1:] -= np.average(vel[1:], axis=0, weights=mass[1:])
        
    # 4. Species
    species = np.zeros(len(pos), dtype=np.int32)
    sp_idx = 0
    species[sp_idx] = 1 # SMBH
    sp_idx += 1
    if n_bulge > 0:
        species[sp_idx:sp_idx+n_bulge] = 1
        sp_idx += n_bulge
    if n_disk > 0:
        species[sp_idx:sp_idx+n_disk] = 1
        sp_idx += n_disk
    if n_gas > 0:
        species[sp_idx:sp_idx+n_gas] = 0
        sp_idx += n_gas
    if n_halo > 0:
        species[sp_idx:sp_idx+n_halo] = 2
        sp_idx += n_halo
        
    # 5. Internal Energy (u)
    u = np.full(len(pos), 200.0)
    if n_gas > 0:
        gas_mask = (species == 0)
        gamma = 5.0/3.0
        sound_speed = 32.0 if proto else 12.0
        u[gas_mask] = sound_speed**2 / (gamma * (gamma - 1.0))
        
    return pos, vel, mass, u, species
