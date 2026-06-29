"""Turn a simulation State into renderable points + colours for the viewport.

Shared by the live GUI viewport and the snapshot-timeline playback, so a running
sim and a saved frame look identical.  Colouring is physically motivated and
type-aware:

  * stars -- blackbody colour from age (young = hot/blue) or, lacking ages, from
    a radius-based temperature proxy;
  * gas   -- warm, brightened by SPH density and whitened by internal energy
    (shock-heated gas glows);
  * dark matter -- faint blue-grey.

Returns float32 positions (N,3) and RGBA (N,4) in [0,1].
"""
from __future__ import annotations

import numpy as np

from core.state import State, PTYPE_STAR, PTYPE_GAS, PTYPE_DM
from export.blackbody import blackbody_rgb
from export.to_pointcloud import synth_temperature


def state_to_display(state: State, clip_kpc: float = 120.0,
                     show_dm: bool = True, dm_alpha: float = 0.18):
    """Return (pos[N,3] float32, rgba[N,4] float32) for the given State.

    ``show_dm`` toggles dark-matter points (they dominate the count and can bury
    the visible galaxy); ``dm_alpha`` sets their opacity when shown.
    """
    if state.n == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 4), np.float32)

    keep = np.linalg.norm(state.pos, axis=1) < clip_kpc
    if not show_dm:
        keep = keep & (state.ptype != PTYPE_DM)
    pos = state.pos[keep].astype(np.float32)
    ptype = state.ptype[keep]
    rgba = np.zeros((pos.shape[0], 4), np.float32)
    rgba[:, 3] = 1.0

    is_star = ptype == PTYPE_STAR
    is_gas = ptype == PTYPE_GAS
    is_dm = ptype == PTYPE_DM
    # Pure-DM scenarios (e.g. Plummer, cosmo) have no stars -> show DM as stars.
    if not (is_star.any() or is_gas.any()):
        is_star = np.ones_like(is_star)
        is_dm = np.zeros_like(is_dm)

    if is_star.any():
        rgba[is_star, :3] = _star_colours(state, keep, is_star)
    if is_gas.any():
        rgba[is_gas] = _gas_colours(state, keep, is_gas)
    if is_dm.any():
        rgba[is_dm] = np.array([0.35, 0.40, 0.55, dm_alpha], np.float32)

    return pos, rgba


def _star_colours(state, keep, is_star) -> np.ndarray:
    """Blackbody RGB for star particles (young = hot/blue), plus an SMBH glow.

    Age -> effective temperature follows an exponential cooling track: the
    youngest stars are O/B-hot (~25000 K, blue-white) and redden toward ~3000 K
    as they age, which reads like a real H-R colour spread rather than a flat
    blue->red ramp.  A dominant central mass (the SMBH) is drawn as a hot
    accretion-disk glow so the black hole stands out.
    """
    pos_k = state.pos[keep][is_star]
    if state.age is not None:
        age = state.age[keep][is_star]
        # T = T_old + (T_young - T_old) * exp(-age / tau)
        temp = np.where(age >= 0.0,
                        3000.0 + 22000.0 * np.exp(-np.clip(age, 0.0, None) / 0.4),
                        synth_temperature(pos_k, seed=state.seed))
    else:
        temp = synth_temperature(pos_k, seed=state.seed)
    # Metal-rich stars read cooler/redder (line blanketing): T_eff lowered by a
    # factor that grows with metallicity Z relative to ~solar (Z_ref = 0.02).
    if state.metallicity is not None:
        z = state.metallicity[keep][is_star]
        temp = temp / (1.0 + 0.30 * np.clip(z / 0.02, 0.0, 3.0))
    rgb = blackbody_rgb(temp).astype(np.float32)

    # SMBH (single dominant star mass) -> bright accretion glow.
    if state.mass is not None:
        mass_k = state.mass[keep][is_star]
        if mass_k.size > 1:
            j = int(np.argmax(mass_k))
            if mass_k[j] >= 20.0 * np.median(mass_k):
                rgb[j] = np.array([1.0, 0.95, 0.8], np.float32)
    return rgb


def _gas_colours(state, keep, is_gas) -> np.ndarray:
    """Gas on a nebular palette, distinct from the blackbody stars.

    Temperature (absolute, not population-normalised so a uniform cloud is never
    all-black) drives the hue: cold dense gas = deep blue; warm star-forming gas
    = teal/cyan; hot shock/SN/AGN gas = magenta -> white (Halpha-like).  This is
    deliberately blue/cyan/magenta so gas never looks like the warm-white stars.
    """
    n = int(is_gas.sum())
    rho = state.rho[keep][is_gas] if state.rho is not None else np.ones(n)
    u = state.u[keep][is_gas] if state.u is not None else np.ones(n)

    b = _norm(np.log10(rho + 1e-6))                  # density -> brightness
    lu = np.log10(np.clip(u, 1.0, None))
    hot = np.clip((lu - np.log10(50.0)) / (np.log10(5000.0) - np.log10(50.0)),
                  0.0, 1.0)

    cold = np.array([0.10, 0.22, 0.45])              # deep blue (cold dust)
    warm = np.array([0.10, 0.70, 0.65])              # teal/cyan (star-forming)
    hotc = np.array([0.95, 0.25, 0.65])              # magenta (ionized/shocked)
    whit = np.array([1.0, 0.92, 1.0])                # white-hot (extreme)
    t = hot[:, None]
    lo = (1.0 - np.clip(t * 2.0, 0, 1)) * cold + np.clip(t * 2.0, 0, 1) * warm
    hi = (1.0 - np.clip((t - 0.5) * 2.0, 0, 1)) * warm + np.clip((t - 0.5) * 2.0, 0, 1) * hotc
    col = np.where(t < 0.5, lo, hi)
    w = np.clip((t - 0.85) / 0.15, 0.0, 1.0)         # extreme-hot -> white
    col = (1.0 - w) * col + w * whit

    # Brightness density-driven with a floor (always visible); cold gas dimmer.
    bright = (0.30 + 0.70 * b) * (0.45 + 0.55 * hot)
    rgba = np.zeros((n, 4), np.float32)
    rgba[:, :3] = (col * bright[:, None]).astype(np.float32)
    rgba[:, 3] = (0.3 + 0.6 * b).astype(np.float32)  # denser gas more opaque
    return rgba


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, 5), np.percentile(x, 98)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)
