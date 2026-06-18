"""Projection maths for the 3D particle brush (pure numpy, GUI-free, testable).

The viewport hands these the camera's model-view-projection matrix:

* :func:`world_to_screen` projects particle positions to pixel coordinates so the
  *erase* brush can pick the points under the cursor;
* :func:`screen_ray_to_plane` unprojects the cursor onto a world plane (the
  galactic plane ``z=0`` by default) so the *add* brush knows where to drop new
  particles -- resolving the depth ambiguity of a 2D click.
"""
from __future__ import annotations

import numpy as np


def qmat_to_np(m) -> np.ndarray:
    """Convert a Qt ``QMatrix4x4`` to a row-major 4x4 numpy array.

    ``QMatrix4x4.data()`` is column-major (OpenGL order), so we reshape and
    transpose to get the usual row-major matrix.
    """
    return np.array(m.data(), dtype=np.float64).reshape(4, 4).T


def world_to_screen(pos: np.ndarray, mvp: np.ndarray, w: int, h: int):
    """Project (N,3) world points to pixel (sx, sy) + a visibility mask.

    Points behind the camera or outside the view frustum are marked not-visible.
    Screen origin is top-left (Qt convention).
    """
    n = len(pos)
    hom = np.empty((n, 4), np.float64)
    hom[:, :3] = pos
    hom[:, 3] = 1.0
    clip = hom @ mvp.T                       # (N,4)
    wc = clip[:, 3]
    safe = np.where(np.abs(wc) < 1e-12, 1e-12, wc)
    ndc = clip[:, :3] / safe[:, None]
    sx = (ndc[:, 0] * 0.5 + 0.5) * w
    sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * h
    visible = (wc > 0) & (np.abs(ndc[:, 0]) <= 1.0) & (np.abs(ndc[:, 1]) <= 1.0)
    return sx, sy, visible


def pick_radius(pos: np.ndarray, mvp: np.ndarray, w: int, h: int,
                mx: float, my: float, radius_px: float) -> np.ndarray:
    """Boolean mask of points whose projection is within ``radius_px`` of (mx,my)."""
    if len(pos) == 0:
        return np.zeros(0, bool)
    sx, sy, vis = world_to_screen(pos, mvp, w, h)
    d2 = (sx - mx) ** 2 + (sy - my) ** 2
    return vis & (d2 <= radius_px ** 2)


def screen_ray_to_plane(mx: float, my: float, w: int, h: int,
                        mvp: np.ndarray, plane_z: float = 0.0):
    """Unproject pixel (mx,my) onto the world plane z=plane_z. Returns (3,) or None.

    Returns ``None`` when the ray is parallel to the plane.
    """
    inv = np.linalg.inv(mvp)
    ndc_x = (mx / w) * 2.0 - 1.0
    ndc_y = (1.0 - my / h) * 2.0 - 1.0

    def unproject(ndc_z):
        p = inv @ np.array([ndc_x, ndc_y, ndc_z, 1.0])
        return p[:3] / p[3]

    near = unproject(-1.0)
    far = unproject(1.0)
    d = far - near
    if abs(d[2]) < 1e-12:
        return None
    t = (plane_z - near[2]) / d[2]
    return near + t * d


def scatter_points(center: np.ndarray, n: int, radius: float,
                   seed: int | None = None) -> np.ndarray:
    """``n`` points uniformly in a sphere of ``radius`` about ``center`` (N,3)."""
    rng = np.random.default_rng(seed)
    r = radius * rng.random(n) ** (1.0 / 3.0)
    cos_t = rng.uniform(-1.0, 1.0, n)
    sin_t = np.sqrt(1.0 - cos_t ** 2)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    offs = np.column_stack((r * sin_t * np.cos(phi),
                            r * sin_t * np.sin(phi), r * cos_t))
    return center[None, :] + offs
