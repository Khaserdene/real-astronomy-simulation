"""Find Blender and launch the two render workflows from the GUI.

- ``open_in_blender`` -- interactive: opens Blender's GUI with the frame loaded as
  art-directable objects (``blender/template_scene.py``).  Non-blocking.
- ``quick_render`` -- headless: renders a folder of point-cloud PLYs to PNGs
  (``blender/render_frames.py``).  Blocking; returns the output folder.
"""
from __future__ import annotations

import glob
import os
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_blender() -> str | None:
    """Locate blender.exe: $BLENDER, then the newest Program Files install."""
    env = os.environ.get("BLENDER")
    if env and os.path.exists(env):
        return env
    pats = [
        "C:/Program Files/Blender Foundation/Blender */blender.exe",
        "C:/Program Files/Blender Foundation/blender.exe",
    ]
    found = []
    for pat in pats:
        found += glob.glob(pat)
    if not found:
        return None
    return sorted(found)[-1]            # highest version dir sorts last


def open_in_blender(frame_dir: str) -> subprocess.Popen:
    """Open Blender's GUI with the per-type frame loaded (interactive)."""
    blender = _require_blender()
    script = os.path.join(_ROOT, "blender", "template_scene.py")
    return subprocess.Popen([blender, "--python", script, "--",
                             os.path.abspath(frame_dir)])


def quick_render(ply_dir: str, out_dir: str, **opts) -> str:
    """Headless-render a folder of PLYs to PNGs; returns the output folder."""
    blender = _require_blender()
    script = os.path.join(_ROOT, "blender", "render_frames.py")
    args = [blender, "--background", "--python", script, "--",
            "--ply", os.path.abspath(ply_dir), "--out", os.path.abspath(out_dir)]
    for k, v in opts.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    subprocess.run(args, check=True)
    return out_dir


def _require_blender() -> str:
    blender = find_blender()
    if blender is None:
        raise FileNotFoundError(
            "Blender not found. Install it, or set the BLENDER environment "
            "variable to blender.exe.")
    return blender
