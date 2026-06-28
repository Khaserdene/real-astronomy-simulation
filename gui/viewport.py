"""Live 3D particle viewport (pyqtgraph GLViewWidget).

A thin wrapper that renders the particles produced by the SimController:
positions (N,3) and RGBA (N,4).  Mouse-drag orbits, scroll zooms (built into
GLViewWidget).  Kept dumb on purpose -- it only displays whatever points it's
handed, so the same widget shows a live run or a played-back snapshot.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtCore import Qt, pyqtSignal

from gui.brush import qmat_to_np, screen_ray_to_plane


class Viewport(gl.GLViewWidget):
    # erase: (screen samples [list of (x,y)], mvp 4x4, width, height, radius_px)
    eraseStroke = pyqtSignal(object, object, int, int, float)
    # add: a single world-space centre (3,) on the z=0 plane
    addAt = pyqtSignal(object)

    def __init__(self, point_size: float = 2.5):
        super().__init__()
        self.setBackgroundColor("k")
        self.setCameraPosition(distance=70, elevation=25, azimuth=-60)
        self.point_size = point_size
        self._scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), np.float32), color=(1, 1, 1, 0.0),
            size=point_size, pxMode=True)
        # Additive blending makes overlapping stars/gas glow like emission.
        self._scatter.setGLOptions("additive")
        self.addItem(self._scatter)

        # --- brush state (off by default; the GUI toggles it) ---
        self.brush_enabled = False
        self.brush_mode = "erase"          # "erase" | "add"
        self.brush_radius_px = 25.0
        self.brush_plane_z = 0.0           # work-plane height (z) for Add depth
        self._painting = False
        self._erase_samples: list = []

    # ----------------------------------------------------------------- brush
    def _mvp(self) -> np.ndarray:
        return qmat_to_np(self.projectionMatrix()) @ qmat_to_np(self.viewMatrix())

    def _paint(self, ev):
        p = ev.position()
        if self.brush_mode == "erase":
            self._erase_samples.append((p.x(), p.y()))
        else:                              # add: drop on the work plane (z=plane_z)
            world = screen_ray_to_plane(p.x(), p.y(), self.width(),
                                        self.height(), self._mvp(),
                                        self.brush_plane_z)
            if world is not None:
                self.addAt.emit(world)

    def mousePressEvent(self, ev):
        if self.brush_enabled and ev.button() == Qt.MouseButton.LeftButton:
            self._painting = True
            self._erase_samples = []
            self._paint(ev)
            ev.accept()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._painting:
            self._paint(ev)
            ev.accept()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._painting:
            self._painting = False
            if self.brush_mode == "erase" and self._erase_samples:
                self.eraseStroke.emit(list(self._erase_samples), self._mvp(),
                                      self.width(), self.height(),
                                      self.brush_radius_px)
            self._erase_samples = []
            ev.accept()
        else:
            super().mouseReleaseEvent(ev)

    def set_points(self, pos: np.ndarray, rgba: np.ndarray) -> None:
        if pos.shape[0] == 0:
            self._scatter.setData(pos=np.zeros((1, 3), np.float32),
                                  color=(1, 1, 1, 0.0), size=self.point_size)
            return
        self._scatter.setData(pos=pos, color=rgba, size=self.point_size,
                              pxMode=True)

    def set_point_size(self, size: float) -> None:
        self.point_size = float(size)

    def frame_extent(self, radius: float) -> None:
        """Point the camera at the origin from a distance suiting ``radius``."""
        self.setCameraPosition(distance=max(radius * 3.0, 10.0))
