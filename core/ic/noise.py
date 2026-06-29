"""Vectorised 3D Perlin / fBm noise for initial-condition substructure.

Pure numpy (no extra dependency).  Used to carve realistic substructure into the
proto-galaxy gas cloud (clumps, filaments, spiral over-densities) -- a proper
gradient noise instead of the earlier sinusoidal pseudo-noise.

  * :func:`perlin3`     -- Ken Perlin's improved 3D gradient noise, ~[-1, 1].
  * :func:`fbm3`        -- fractional Brownian motion (octave sum), mapped to [0, 1].
  * :func:`domain_warp` -- offset sample coordinates by a noise field for swirl.

All functions are vectorised over an (N, 3) array of points and are deterministic
in ``seed`` (identical seed -> identical field at any resolution).
"""
from __future__ import annotations

import numpy as np


def perm_table(seed: int) -> np.ndarray:
    """A doubled 512-entry permutation table for the given seed."""
    rng = np.random.default_rng(int(seed))
    p = np.arange(256, dtype=np.int64)
    rng.shuffle(p)
    return np.concatenate([p, p])          # length 512 -> indices never overflow


def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a, b, t):
    return a + t * (b - a)


def _grad(h, x, y, z):
    """Classic 12-direction gradient hash (improved Perlin)."""
    h = h & 15
    u = np.where(h < 8, x, y)
    v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
    return np.where(h & 1 == 0, u, -u) + np.where(h & 2 == 0, v, -v)


def perlin3(x, y, z, perm) -> np.ndarray:
    """Improved 3D Perlin noise at points (x, y, z); returns ~[-1, 1]."""
    xi = np.floor(x).astype(np.int64) & 255
    yi = np.floor(y).astype(np.int64) & 255
    zi = np.floor(z).astype(np.int64) & 255
    xf = x - np.floor(x)
    yf = y - np.floor(y)
    zf = z - np.floor(z)
    u, v, w = _fade(xf), _fade(yf), _fade(zf)

    A = perm[xi] + yi
    AA = perm[A] + zi
    AB = perm[A + 1] + zi
    B = perm[xi + 1] + yi
    BA = perm[B] + zi
    BB = perm[B + 1] + zi

    x1 = _lerp(_grad(perm[AA], xf, yf, zf),
               _grad(perm[BA], xf - 1, yf, zf), u)
    x2 = _lerp(_grad(perm[AB], xf, yf - 1, zf),
               _grad(perm[BB], xf - 1, yf - 1, zf), u)
    y1 = _lerp(x1, x2, v)
    x3 = _lerp(_grad(perm[AA + 1], xf, yf, zf - 1),
               _grad(perm[BA + 1], xf - 1, yf, zf - 1), u)
    x4 = _lerp(_grad(perm[AB + 1], xf, yf - 1, zf - 1),
               _grad(perm[BB + 1], xf - 1, yf - 1, zf - 1), u)
    y2 = _lerp(x3, x4, v)
    return _lerp(y1, y2, w)


def fbm3(points, seed=0, octaves=4, frequency=1.0,
         lacunarity=2.0, persistence=0.5) -> np.ndarray:
    """Fractional Brownian motion (sum of Perlin octaves), mapped to [0, 1].

    ``frequency`` sets the base feature size, ``lacunarity`` the frequency step
    per octave, ``persistence`` the amplitude falloff.  Returns one value per point.
    """
    points = np.asarray(points, dtype=np.float64)
    perm = perm_table(seed)
    total = np.zeros(len(points))
    amp, freq, norm = 1.0, float(frequency), 0.0
    for _ in range(max(int(octaves), 1)):
        total += amp * perlin3(points[:, 0] * freq, points[:, 1] * freq,
                               points[:, 2] * freq, perm)
        norm += amp
        amp *= persistence
        freq *= lacunarity
    val = total / max(norm, 1e-9)          # ~[-1, 1]
    return np.clip(0.5 * (val + 1.0), 0.0, 1.0)


def domain_warp(points, seed=0, strength=1.0, frequency=1.0) -> np.ndarray:
    """Return ``points`` displaced by an independent Perlin vector field.

    Feeding warped coordinates into :func:`fbm3` produces swirled, organic
    structure instead of axis-aligned blobs.  ``strength`` is in the same length
    units as ``points``.
    """
    points = np.asarray(points, dtype=np.float64)
    if strength == 0.0:
        return points
    px, py, pz = (points[:, 0] * frequency, points[:, 1] * frequency,
                  points[:, 2] * frequency)
    wx = perlin3(px, py, pz, perm_table(seed + 11))
    wy = perlin3(px, py, pz, perm_table(seed + 37))
    wz = perlin3(px, py, pz, perm_table(seed + 101))
    return points + strength * np.column_stack([wx, wy, wz])
