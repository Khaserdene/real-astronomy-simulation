"""Live 3D particle viewport (pyqtgraph GLViewWidget).

A thin wrapper that renders the particles produced by the SimController:
positions (N,3) and RGBA (N,4).  Mouse-drag orbits, scroll zooms (built into
GLViewWidget).  Kept dumb on purpose -- it only displays whatever points it's
handed, so the same widget shows a live run or a played-back snapshot.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl


class Viewport(gl.GLViewWidget):
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
