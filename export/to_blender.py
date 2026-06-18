"""Export a snapshot as separate, attribute-rich PLY files per particle type.

For the *interactive* Blender workflow we want each particle type as its own
object with its own material, driven by physical attributes.  This writes up to
three PLYs into a folder:

    stars.ply  -- positions + RGB (blackbody by age/temperature) + float
                  attributes ``temperature`` and ``age``
    gas.ply    -- positions + float attributes ``density`` and ``temperature``
    dm.ply     -- positions only (faint halo)

Blender's PLY importer brings custom float properties in as named mesh
attributes, so materials/geometry-nodes can key on ``temperature``/``density``
etc.  ``blender/template_scene.py`` consumes these.
"""
from __future__ import annotations

import os
import struct

import numpy as np

from core.state import State, PTYPE_STAR, PTYPE_GAS, PTYPE_DM
from export.blackbody import blackbody_rgb
from export.to_pointcloud import synth_temperature


def export_frame(h5_path: str, out_dir: str) -> dict:
    """Split a snapshot into per-type PLYs. Returns {type: path} written."""
    os.makedirs(out_dir, exist_ok=True)
    state = State.load(h5_path)
    written = {}

    star = state.ptype == PTYPE_STAR
    gas = state.ptype == PTYPE_GAS
    dm = state.ptype == PTYPE_DM
    if not (star.any() or gas.any()):     # pure-DM run -> render DM as stars
        star = np.ones(state.n, bool)
        dm = np.zeros(state.n, bool)

    if star.any():
        pos = state.pos[star].astype(np.float32)
        if state.age is not None:
            age = state.age[star]
            temp = np.where(age >= 0.0,
                            np.clip(9500.0 - (age / 0.3) * 6000.0, 3200.0, 9500.0),
                            synth_temperature(state.pos[star], seed=state.seed))
        else:
            age = np.full(pos.shape[0], -1.0)
            temp = synth_temperature(state.pos[star], seed=state.seed)
        rgb = (blackbody_rgb(temp) * 255).astype(np.uint8)
        p = os.path.join(out_dir, "stars.ply")
        _write_ply(p, pos, rgb=rgb,
                   floats={"temperature": temp.astype(np.float32),
                           "age": age.astype(np.float32)})
        written["stars"] = p

    if gas.any():
        pos = state.pos[gas].astype(np.float32)
        rho = state.rho[gas] if state.rho is not None else np.ones(pos.shape[0])
        u = state.u[gas] if state.u is not None else np.ones(pos.shape[0])
        p = os.path.join(out_dir, "gas.ply")
        _write_ply(p, pos, rgb=None,
                   floats={"density": _norm(rho).astype(np.float32),
                           "temperature": _norm(np.log10(u + 1e-6)).astype(np.float32)})
        written["gas"] = p

    if dm.any():
        pos = state.pos[dm].astype(np.float32)
        p = os.path.join(out_dir, "dm.ply")
        _write_ply(p, pos, rgb=None, floats=None)
        written["dm"] = p

    return written


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, 5), np.percentile(x, 98)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def _write_ply(path, pos, rgb=None, floats=None):
    """Binary little-endian PLY: float xyz [+ uchar rgb] [+ named float attrs]."""
    n = pos.shape[0]
    floats = floats or {}
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    if rgb is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    for name in floats:
        header.append(f"property float {name}")
    header.append("end_header\n")

    fmt = "<fff" + ("BBB" if rgb is not None else "") + "f" * len(floats)
    packer = struct.Struct(fmt)
    fkeys = list(floats)
    with open(path, "wb") as f:
        f.write(("\n".join(header)).encode("ascii"))
        buf = bytearray()
        for i in range(n):
            row = [float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])]
            if rgb is not None:
                row += [int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2])]
            row += [float(floats[k][i]) for k in fkeys]
            buf += packer.pack(*row)
        f.write(buf)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        raise SystemExit("usage: python -m export.to_blender <snap.h5> <out_dir>")
    print(export_frame(sys.argv[1], sys.argv[2]))
