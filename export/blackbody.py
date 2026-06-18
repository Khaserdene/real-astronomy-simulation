"""Blackbody colour: map a temperature in Kelvin to an sRGB colour.

Used to colour star particles physically -- hot stars are blue, cool stars red --
so a rendered galaxy carries real colour information rather than a flat tint.
Uses the well-known Tanner Helland approximation of the Planckian locus, which is
fast, vectorised, and accurate enough for visualisation over ~1000-40000 K.
"""
from __future__ import annotations

import numpy as np


def blackbody_rgb(temperature_k) -> np.ndarray:
    """Return sRGB in [0,1] for one or many temperatures (Kelvin).

    Accepts a scalar or array; returns shape (..., 3).
    """
    t = np.atleast_1d(np.asarray(temperature_k, dtype=np.float64))
    t = np.clip(t, 1000.0, 40000.0) / 100.0

    # ``t - 60`` is only used on the t > 66 branch; clip its base to stay
    # positive so the fractional powers never hit a negative number (NaN).
    t60 = np.clip(t - 60.0, 1e-6, None)

    # --- red ---
    red = np.where(t <= 66.0, 255.0,
                   329.698727446 * t60 ** -0.1332047592)

    # --- green ---
    green = np.where(
        t <= 66.0,
        99.4708025861 * np.log(t) - 161.1195681661,
        288.1221695283 * t60 ** -0.0755148492,
    )

    # --- blue ---
    blue = np.where(
        t >= 66.0, 255.0,
        np.where(t <= 19.0, 0.0,
                 138.5177312231 * np.log(np.clip(t - 10.0, 1e-6, None))
                 - 305.0447927307),
    )

    rgb = np.stack([red, green, blue], axis=-1)
    rgb = np.clip(rgb, 0.0, 255.0) / 255.0
    return rgb[0] if np.isscalar(temperature_k) or np.ndim(temperature_k) == 0 else rgb
