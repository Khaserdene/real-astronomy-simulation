"""Uniform-grid neighbour acceleration for SPH (host-side bin, GPU-side walk).

The living-galaxy SPH kernels (density, pressure forces, supernova feedback) are
naturally O(N^2): every gas particle scans every other.  This bins the *gas*
particles into a uniform grid whose cell size is the SPH search radius, so a
particle only has to look at its own cell and the 26 neighbours (a 3x3x3
stencil) -- O(N) on a roughly uniform distribution.

Design choices that keep it simple and correct:

* **Cell size = search radius** ``= max(2*h_max, min_radius)`` so a single 3x3x3
  stencil is guaranteed to contain every pair within ``2*h_ij`` (density/forces)
  *and* within ``min_radius`` (feedback's ``r_fb``).  Bigger cells are always
  safe -- only slower -- so using the max of the two radii lets one grid serve
  all three kernels.
* **Gas only.**  Stars/DM are excluded from the bins; gravity handles them
  separately (Barnes-Hut / direct).  As gas turns into stars over a run this
  keeps the neighbour scan from dragging in particles that never contribute SPH.
* **Bin on the host (numpy), walk on the GPU.**  A counting sort in numpy is a
  few milliseconds even at 10^5 cells and avoids a fiddly GPU prefix-sum; the
  resulting ``cell_start`` / ``sorted-index`` arrays upload once per rebuild.

The GPU traversal kernels live in ``living_galaxy_engine`` (they need the SPH
spline helpers); this class only owns the grid fields and the rebuild.

(No ``from __future__ import annotations`` needed -- there are no Taichi kernels
here -- but kept consistent with the rest of ``core/solvers``.)
"""
import numpy as np
import taichi as ti


class NeighborGrid:
    """Uniform grid over the gas particles, rebuilt each force evaluation."""

    def __init__(self, n: int, min_radius: float = 0.0):
        self.n = n
        self.min_radius = float(min_radius)
        # cell_start is sized to a cell budget proportional to N (avg ~1/cell).
        self.max_cells = max(64, 4 * n)
        self.gsort = ti.field(ti.i32, shape=max(n, 1))      # gas global indices
        self.cell_start = ti.field(ti.i32, shape=self.max_cells + 1)
        # Grid geometry for the current rebuild (read by the GPU kernels).
        self.nx = self.ny = self.nz = 1
        self.lo = (0.0, 0.0, 0.0)
        self.inv_cell = 1.0
        self.ngas = 0

    def rebuild(self, pos_field, h_field, is_star_field) -> int:
        """Bin the gas particles; return the gas count (0 means "no grid")."""
        pos = pos_field.to_numpy()
        h = h_field.to_numpy()
        is_star = is_star_field.to_numpy()
        gas = np.nonzero(is_star == 0)[0].astype(np.int32)
        self.ngas = int(gas.size)
        if self.ngas == 0:
            return 0

        gpos = pos[gas]
        h_max = float(h[gas].max())
        cell = max(2.0 * h_max, self.min_radius)
        cell = cell if cell > 1e-9 else 1.0

        lo = gpos.min(axis=0)
        span = gpos.max(axis=0) - lo
        dims = np.maximum(np.floor(span / cell).astype(np.int64) + 1, 1)
        # Keep the cell count within budget: grow the cells if the box is huge.
        while int(dims.prod()) > self.max_cells:
            cell *= 1.3
            dims = np.maximum(np.floor(span / cell).astype(np.int64) + 1, 1)
        n_cells = int(dims.prod())

        ijk = np.clip(((gpos - lo) / cell).astype(np.int64), 0, dims - 1)
        flat = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
        order = np.argsort(flat, kind="stable")
        sorted_global = gas[order]
        counts = np.bincount(flat, minlength=n_cells)
        starts = np.empty(n_cells + 1, np.int32)
        starts[0] = 0
        starts[1:] = np.cumsum(counts).astype(np.int32)

        # Upload (only the used prefixes; the rest of the fields is stale but
        # never read because the kernels index < ngas / <= n_cells).
        self.gsort.from_numpy(
            np.pad(sorted_global, (0, self.n - self.ngas), constant_values=0)
            if self.ngas < self.n else sorted_global)
        cs = np.zeros(self.max_cells + 1, np.int32)
        cs[:n_cells + 1] = starts
        self.cell_start.from_numpy(cs)

        self.nx, self.ny, self.nz = int(dims[0]), int(dims[1]), int(dims[2])
        self.lo = (float(lo[0]), float(lo[1]), float(lo[2]))
        self.inv_cell = 1.0 / cell
        return self.ngas

    def rebuild_all(self, pos_field, h_field) -> int:
        """Bin all particles. For pure-gas engines where every particle is gas."""
        pos = pos_field.to_numpy()
        h = h_field.to_numpy()
        gas = np.arange(self.n, dtype=np.int32)
        self.ngas = self.n
        if self.ngas == 0:
            return 0

        gpos = pos
        h_max = float(h.max())
        cell = max(2.0 * h_max, self.min_radius)
        cell = cell if cell > 1e-9 else 1.0

        lo = gpos.min(axis=0)
        span = gpos.max(axis=0) - lo
        dims = np.maximum(np.floor(span / cell).astype(np.int64) + 1, 1)
        # Keep the cell count within budget: grow the cells if the box is huge.
        while int(dims.prod()) > self.max_cells:
            cell *= 1.3
            dims = np.maximum(np.floor(span / cell).astype(np.int64) + 1, 1)
        n_cells = int(dims.prod())

        ijk = np.clip(((gpos - lo) / cell).astype(np.int64), 0, dims - 1)
        flat = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
        order = np.argsort(flat, kind="stable")
        sorted_global = gas[order]
        counts = np.bincount(flat, minlength=n_cells)
        starts = np.empty(n_cells + 1, np.int32)
        starts[0] = 0
        starts[1:] = np.cumsum(counts).astype(np.int32)

        self.gsort.from_numpy(sorted_global)
        cs = np.zeros(self.max_cells + 1, np.int32)
        cs[:n_cells + 1] = starts
        self.cell_start.from_numpy(cs)

        self.nx, self.ny, self.nz = int(dims[0]), int(dims[1]), int(dims[2])
        self.lo = (float(lo[0]), float(lo[1]), float(lo[2]))
        self.inv_cell = 1.0 / cell
        return self.ngas
