"""Convert simulation snapshots (HDF5) into coloured point-cloud PLY files.

This is the bridge from the engine's physical state to a Blender-renderable asset.
Star particles are written as a binary PLY with per-vertex RGB colour derived from
a blackbody temperature.

Until the engine tracks real stellar ages/temperatures (Phase 5+), we synthesise
a physically-motivated temperature gradient: the old bulge (small radius) is
cooler/redder, the young disk and spiral arms (large radius) are hotter/bluer.
The function signature already accepts a real ``temperature`` array, so swapping
in physical values later changes nothing downstream.

Run as a batch over a folder:
    .venv/Scripts/python.exe -m export.to_pointcloud output/run_high
"""
from __future__ import annotations

import glob
import os
import struct

import numpy as np

from core.state import State, PTYPE_STAR
from export.blackbody import blackbody_rgb


def synth_temperature(pos: np.ndarray, seed: int = 0) -> np.ndarray:
    """Radius-based placeholder stellar temperature [K] (bulge red, disk blue).

    Wide range (~3200 K bulge to ~11000 K outer disk) so the colour gradient
    reads clearly once rendered -- a yellow-red core fading to blue arms, like a
    real spiral.
    """
    R = np.linalg.norm(pos[:, :2], axis=1)
    base = 3200.0 + 7800.0 * np.clip(R / 14.0, 0.0, 1.0)
    scatter = np.random.default_rng(seed).normal(0.0, 500.0, size=R.shape)
    return np.clip(base + scatter, 3000.0, 12000.0)


def snapshot_to_ply(h5_path: str, ply_path: str) -> int:
    """Write the star particles of a snapshot to a coloured binary PLY.

    Returns the number of points written.
    """
    state = State.load(h5_path)
    mask = state.ptype == PTYPE_STAR
    if not mask.any():               # e.g. pure-Plummer DM test -> render all
        mask = np.ones(state.n, dtype=bool)
    pos = state.pos[mask].astype(np.float32)

    if state.temperature is not None:
        temp = np.asarray(state.temperature)[mask]
    else:
        temp = synth_temperature(state.pos[mask], seed=state.seed)
    rgb = (blackbody_rgb(temp) * 255.0).astype(np.uint8)

    _write_ply(ply_path, pos, rgb)
    return pos.shape[0]


def batch(folder: str, out_subdir: str = "ply") -> None:
    """Convert every snap_*.h5 in ``folder`` to a PLY under ``folder/ply``."""
    out = os.path.join(folder, out_subdir)
    os.makedirs(out, exist_ok=True)
    snaps = sorted(glob.glob(os.path.join(folder, "snap_*.h5")))
    if not snaps:
        raise FileNotFoundError(f"no snap_*.h5 in {folder}")
    for h5 in snaps:
        name = os.path.splitext(os.path.basename(h5))[0]
        n = snapshot_to_ply(h5, os.path.join(out, name + ".ply"))
        print(f"  {name}: {n:,} points")
    print(f"done: {len(snaps)} PLY files in {out}/")


def _write_ply(path: str, pos: np.ndarray, rgb: np.ndarray) -> None:
    """Write a binary little-endian PLY with x,y,z float + r,g,b uchar."""
    n = pos.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    # Interleave each vertex as <fff BBB>.
    packer = struct.Struct("<fffBBB")
    with open(path, "wb") as f:
        f.write(header)
        buf = bytearray()
        for i in range(n):
            buf += packer.pack(pos[i, 0], pos[i, 1], pos[i, 2],
                               int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2]))
        f.write(buf)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m export.to_pointcloud <run_folder>")
    batch(sys.argv[1])
