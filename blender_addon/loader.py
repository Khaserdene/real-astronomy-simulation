"""Load a simulation output folder into Blender as an animation.

Reads the engine's portable ``snap_*.h5`` snapshots *directly* (via the bundled
``h5py`` wheel) and builds one art-directable object per particle type -- Stars
(emissive points coloured by blackbody temperature), Gas (Principled Volume by
density) and DarkMatter (faint points).  A ``frame_change_pre`` handler swaps each
object's geometry to the snapshot for the current Blender frame, so the whole run
plays back on Blender's timeline and renders like any other animation.

This module is used two ways:

* by the extension (``__init__.py`` calls :func:`load_folder` from an operator);
* as a standalone script -- ``blender --python loader.py -- <folder>`` -- which is
  how the desktop GUI's "Open in Blender (animation)" hands a run over.

It is deliberately self-contained (no import from the project's ``core``) so the
installed extension carries everything it needs.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

try:
    import bpy
except ImportError:          # allows non-Blender import (e.g. doc tooling)
    bpy = None

# Module-level playback state, keyed for the single active import.
_STATE: dict = {"folder": None, "snaps": [], "objects": {}, "current": -1}

PTYPE_DM, PTYPE_STAR, PTYPE_GAS = 0, 1, 2


# ============================================================ snapshot reading
def _import_h5py():
    """Import h5py, falling back to the user site-packages.

    When the extension is installed, its bundled h5py wheel is on the path.  When
    this module is run standalone (``blender --python loader.py``, the GUI's
    animation button), Blender does not add the user site by default -- so a
    user-level ``pip install h5py`` into Blender's Python would otherwise be
    missed.  Add the user site and retry once.
    """
    try:
        import h5py
        return h5py
    except ModuleNotFoundError:
        try:
            import site
            extra = site.getusersitepackages()
            paths = [extra] if isinstance(extra, str) else list(extra)
            for p in paths:
                if p and p not in sys.path:
                    sys.path.append(p)
        except Exception:
            pass
        import h5py            # retry; raises a clear error if still missing
        return h5py


def read_snapshot(path: str) -> dict:
    """Minimal HDF5 snapshot read (mirrors core.state.State.save layout)."""
    h5py = _import_h5py()

    with h5py.File(path, "r") as f:
        d = {"pos": f["pos"][:], "ptype": f["ptype"][:],
             "seed": int(f.attrs.get("seed", 0))}
        for opt in ("age", "u", "rho"):
            d[opt] = f[opt][:] if opt in f else None
    return d


def list_snapshots(folder: str) -> list[str]:
    return sorted(glob.glob(os.path.join(folder, "snap_*.h5")))


# ===================================================================== colour
def blackbody_rgb(temperature_k) -> np.ndarray:
    """sRGB in [0,1] for temperature(s) in Kelvin (Tanner Helland approx)."""
    t = np.atleast_1d(np.asarray(temperature_k, dtype=np.float64))
    t = np.clip(t, 1000.0, 40000.0) / 100.0
    t60 = np.clip(t - 60.0, 1e-6, None)
    red = np.where(t <= 66.0, 255.0, 329.698727446 * t60 ** -0.1332047592)
    green = np.where(t <= 66.0,
                     99.4708025861 * np.log(t) - 161.1195681661,
                     288.1221695283 * t60 ** -0.0755148492)
    blue = np.where(t >= 66.0, 255.0,
                    np.where(t <= 19.0, 0.0,
                             138.5177312231 * np.log(np.clip(t - 10.0, 1e-6, None))
                             - 305.0447927307))
    rgb = np.clip(np.stack([red, green, blue], axis=-1), 0.0, 255.0) / 255.0
    return rgb


def synth_temperature(pos: np.ndarray, seed: int = 0) -> np.ndarray:
    R = np.linalg.norm(pos[:, :2], axis=1)
    base = 3200.0 + 7800.0 * np.clip(R / 14.0, 0.0, 1.0)
    scatter = np.random.default_rng(seed).normal(0.0, 500.0, size=R.shape)
    return np.clip(base + scatter, 3000.0, 12000.0)


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, 5), np.percentile(x, 98)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def _type_arrays(d: dict) -> dict:
    """Split a snapshot dict into per-type {pos, col/density/temp} payloads."""
    ptype = d["ptype"]
    star = ptype == PTYPE_STAR
    gas = ptype == PTYPE_GAS
    dm = ptype == PTYPE_DM
    if not (star.any() or gas.any()):     # pure-DM run -> show DM as stars
        star = np.ones(len(ptype), bool)
        dm = np.zeros(len(ptype), bool)

    out = {}
    if star.any():
        pos = d["pos"][star]
        if d["age"] is not None:
            age = d["age"][star]
            temp = np.where(age >= 0.0,
                            np.clip(9500.0 - (age / 0.3) * 6000.0, 3200.0, 9500.0),
                            synth_temperature(pos, seed=d["seed"]))
        else:
            temp = synth_temperature(pos, seed=d["seed"])
        out["Stars"] = {"pos": pos.astype(np.float32),
                        "Col": blackbody_rgb(temp).astype(np.float32)}
    if gas.any():
        pos = d["pos"][gas]
        rho = d["rho"][gas] if d["rho"] is not None else np.ones(len(pos))
        u = d["u"][gas] if d["u"] is not None else np.ones(len(pos))
        out["Gas"] = {"pos": pos.astype(np.float32),
                      "density": _norm(rho).astype(np.float32),
                      "temperature": _norm(np.log10(u + 1e-6)).astype(np.float32)}
    if dm.any():
        out["DarkMatter"] = {"pos": d["pos"][dm].astype(np.float32)}
    return out


# ============================================================ mesh / materials
def _set_mesh(obj, pos: np.ndarray, color: np.ndarray | None = None,
              floats: dict | None = None):
    """Replace ``obj``'s geometry with a vertex cloud + named attributes."""
    me = obj.data
    me.clear_geometry()
    me.from_pydata(pos.tolist(), [], [])
    me.update()
    n = len(pos)
    if color is not None:
        attr = me.attributes.get("Col") or me.attributes.new(
            "Col", "FLOAT_COLOR", "POINT")
        rgba = np.empty((n, 4), np.float32)
        rgba[:, :3] = color
        rgba[:, 3] = 1.0
        attr.data.foreach_set("color", rgba.ravel())
    for name, vals in (floats or {}).items():
        attr = me.attributes.get(name) or me.attributes.new(name, "FLOAT", "POINT")
        attr.data.foreach_set("value", np.asarray(vals, np.float32).ravel())
    me.update()


def _new_object(name: str):
    me = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    return obj


def emissive_points_material(name, color_attr=None, color=(1, 1, 1), strength=1.3):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    if color_attr:
        attr = nt.nodes.new("ShaderNodeAttribute")
        attr.attribute_name = color_attr
        nt.links.new(attr.outputs["Color"], emit.inputs["Color"])
    else:
        emit.inputs["Color"].default_value = (*color, 1.0)
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def gas_volume_material():
    mat = bpy.data.materials.new("Gas")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    vol.inputs["Color"].default_value = (1.0, 0.55, 0.25, 1.0)
    vol.inputs["Density"].default_value = 0.15
    vol.inputs["Emission Strength"].default_value = 0.8
    vol.inputs["Emission Color"].default_value = (1.0, 0.55, 0.25, 1.0)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])
    return mat


def points_modifier(obj, radius, mat):
    ng = bpy.data.node_groups.new(obj.name + "_pts", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin = ng.nodes.new("NodeGroupInput")
    gout = ng.nodes.new("NodeGroupOutput")
    m2p = ng.nodes.new("GeometryNodeMeshToPoints")
    m2p.inputs["Radius"].default_value = radius
    sm = ng.nodes.new("GeometryNodeSetMaterial")
    sm.inputs["Material"].default_value = mat
    ng.links.new(gin.outputs[0], m2p.inputs["Mesh"])
    ng.links.new(m2p.outputs["Points"], sm.inputs["Geometry"])
    ng.links.new(sm.outputs["Geometry"], gout.inputs[0])
    obj.modifiers.new("Points", "NODES").node_group = ng


def _set_voxel_size_mode(p2v):
    """Select voxel-size resolution on a Points-to-Volume node, cross-version.

    Blender <=4.x exposes a ``resolution_mode`` enum property; Blender 5.x moved
    it to a "Resolution Mode" menu input socket (values "Amount"/"Size").
    """
    try:
        if hasattr(p2v, "resolution_mode"):
            p2v.resolution_mode = "VOXEL_SIZE"
        elif "Resolution Mode" in p2v.inputs:
            p2v.inputs["Resolution Mode"].default_value = "Size"
    except Exception:
        pass  # fall back to the node's default (Amount) mode


def volume_modifier(obj, voxel, radius, mat):
    ng = bpy.data.node_groups.new(obj.name + "_vol", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin = ng.nodes.new("NodeGroupInput")
    gout = ng.nodes.new("NodeGroupOutput")
    p2v = ng.nodes.new("GeometryNodePointsToVolume")
    _set_voxel_size_mode(p2v)
    p2v.inputs["Voxel Size"].default_value = voxel
    p2v.inputs["Radius"].default_value = radius
    sm = ng.nodes.new("GeometryNodeSetMaterial")
    sm.inputs["Material"].default_value = mat
    ng.links.new(gin.outputs[0], p2v.inputs["Points"])
    ng.links.new(p2v.outputs["Volume"], sm.inputs["Geometry"])
    ng.links.new(sm.outputs["Geometry"], gout.inputs[0])
    obj.modifiers.new("Volume", "NODES").node_group = ng


def setup_world_camera():
    import math
    if bpy.context.scene.world is None:
        world = bpy.data.worlds.new("Space")
        world.use_nodes = True
        world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
        bpy.context.scene.world = world
    if bpy.context.scene.camera is None:
        target = bpy.data.objects.new("Target", None)
        bpy.context.collection.objects.link(target)
        cam_data = bpy.data.cameras.new("Cam")
        cam_data.clip_end = 1e5
        cam = bpy.data.objects.new("Cam", cam_data)
        bpy.context.collection.objects.link(cam)
        el = math.radians(28)
        cam.location = (0, -60 * math.cos(el), 60 * math.sin(el))
        con = cam.constraints.new("TRACK_TO")
        con.target = target
        con.track_axis = "TRACK_NEGATIVE_Z"
        con.up_axis = "UP_Y"
        bpy.context.scene.camera = cam


# ================================================================ frame update
def _apply_frame(index: int):
    """Update all astro objects to snapshot ``index`` (clamped)."""
    snaps = _STATE["snaps"]
    if not snaps:
        return
    index = max(0, min(index, len(snaps) - 1))
    if index == _STATE["current"]:
        return
    data = _type_arrays(read_snapshot(snaps[index]))
    objs = _STATE["objects"]
    for name, obj in objs.items():
        payload = data.get(name)
        if payload is None:
            _set_mesh(obj, np.zeros((0, 3), np.float32))
            continue
        if name == "Stars":
            _set_mesh(obj, payload["pos"], color=payload["Col"])
        elif name == "Gas":
            _set_mesh(obj, payload["pos"],
                      floats={"density": payload["density"],
                              "temperature": payload["temperature"]})
        else:
            _set_mesh(obj, payload["pos"])
    _STATE["current"] = index


def _frame_handler(scene):
    # Blender frames are 1-based; snapshot indices are 0-based.
    _apply_frame(scene.frame_current - 1)


def _install_handler():
    handlers = bpy.app.handlers.frame_change_pre
    for h in list(handlers):
        if getattr(h, "__name__", "") == "_frame_handler":
            handlers.remove(h)
    handlers.append(_frame_handler)


# ===================================================================== public
def load_folder(folder: str) -> int:
    """Import ``folder``'s snapshots as an animation. Returns the frame count."""
    folder = os.path.abspath(folder)
    snaps = list_snapshots(folder)
    if not snaps:
        raise FileNotFoundError(f"no snap_*.h5 in {folder}")

    setup_world_camera()
    # (Re)create the three type objects with their materials/modifiers.
    objs = {}
    stars = _new_object("AstroStars")
    points_modifier(stars, 0.08,
                    emissive_points_material("Stars", color_attr="Col", strength=1.4))
    objs["Stars"] = stars
    gas = _new_object("AstroGas")
    volume_modifier(gas, 0.3, 0.6, gas_volume_material())
    objs["Gas"] = gas
    dm = _new_object("AstroDM")
    points_modifier(dm, 0.05,
                    emissive_points_material("DarkMatter", color=(0.25, 0.3, 0.45),
                                             strength=0.3))
    objs["DarkMatter"] = dm

    _STATE.update(folder=folder, snaps=snaps, objects=objs, current=-1)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = len(snaps)
    scene.render.engine = "CYCLES"
    _install_handler()
    scene.frame_set(1)            # triggers the handler -> builds frame 0
    print(f"[astro] loaded {len(snaps)} frames from {folder}")
    return len(snaps)


def _argv_folder():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if not argv:
        raise SystemExit("usage: blender --python loader.py -- <sim_folder>")
    return argv[0]


if __name__ == "__main__":
    load_folder(_argv_folder())
