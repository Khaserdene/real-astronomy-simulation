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


def state_to_display(state: State, clip_kpc: float = 120.0):
    """Return (pos[N,3] float32, rgba[N,4] float32) for the given State."""
    if state.n == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 4), np.float32)

    keep = np.linalg.norm(state.pos, axis=1) < clip_kpc
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
        rgba[is_dm] = np.array([0.35, 0.40, 0.55, 0.25], np.float32)

    return pos, rgba


def _star_colours(state, keep, is_star) -> np.ndarray:
    """Blackbody RGB for star particles (young = hot/blue)."""
    pos_k = state.pos[keep][is_star]
    if state.age is not None:
        age = state.age[keep][is_star]
        temp = np.where(age >= 0.0,
                        np.clip(9500.0 - (age / 0.3) * 6000.0, 3200.0, 9500.0),
                        synth_temperature(pos_k, seed=state.seed))
    else:
        temp = synth_temperature(pos_k, seed=state.seed)
    return blackbody_rgb(temp).astype(np.float32)


def _gas_colours(state, keep, is_gas) -> np.ndarray:
    """Warm gas, brightened by density and whitened by internal energy."""
    n = int(is_gas.sum())
    rho = state.rho[keep][is_gas] if state.rho is not None else np.ones(n)
    u = state.u[keep][is_gas] if state.u is not None else np.ones(n)

    b = _norm(np.log10(rho + 1e-6))                  # density -> brightness
    hot = _norm(np.log10(u + 1e-6))                  # energy  -> whiteness
    base = np.array([1.0, 0.55, 0.25])               # warm orange
    white = np.array([1.0, 1.0, 1.0])
    col = (1.0 - hot)[:, None] * base + hot[:, None] * white
    rgba = np.zeros((n, 4), np.float32)
    rgba[:, :3] = col * (0.25 + 0.75 * b)[:, None]
    rgba[:, 3] = (0.3 + 0.6 * b).astype(np.float32)  # denser gas more opaque
    return rgba


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, 5), np.percentile(x, 98)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)
