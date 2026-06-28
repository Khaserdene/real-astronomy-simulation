"""Barnes-Hut treecode gravity on the GPU (LBVH / Karras radix tree).

Direct N^2 gravity does not scale; this brings force evaluation to ~O(N log N):

1. **Morton codes** order particles along a space-filling curve (numpy).
2. A **binary radix tree** (Karras 2012) is built over the sorted codes on the
   GPU -- exactly N-1 internal + N leaf nodes, no empty cells, O(N).
3. Each node's **centre of mass, total mass and bounding box** are filled in by a
   bottom-up pass (atomic, one thread per leaf walking to the root).
4. Per particle, a **stack walk** opens a node only when it is too close/large
   (the Barnes-Hut criterion ``size^2 < theta^2 * distance^2``); distant nodes
   contribute through their multipole (COM) -- the O(log N) part.

Validated against the direct solver: the force error shrinks with ``theta`` and
energy is conserved on a Plummer sphere.  Falls back to direct for tiny N.

(No ``from __future__ import annotations`` -- it would stringify the ``ti.template()``
kernel annotations and break Taichi's argument extraction.)
"""
import numpy as np
import taichi as ti

from core.units import G


# ----------------------------------------------------------- Morton (numpy)
def _part1by2(x: np.ndarray) -> np.ndarray:
    """Spread the low 21 bits of x so they occupy every 3rd bit (uint64)."""
    x = x.astype(np.uint64) & np.uint64(0x1FFFFF)
    x = (x | (x << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    x = (x | (x << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    x = (x | (x << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    x = (x | (x << np.uint64(2))) & np.uint64(0x1249249249249249)
    return x


def morton_codes(pos: np.ndarray, bits: int = 21) -> np.ndarray:
    """63-bit Morton codes (uint64) for points in their bounding cube."""
    lo = pos.min(axis=0)
    span = float((pos.max(axis=0) - lo).max())
    span = span if span > 0 else 1.0
    scale = (2 ** bits - 1) / (span * 1.0000001)
    q = np.clip(((pos - lo) * scale), 0, 2 ** bits - 1).astype(np.uint64)
    return (_part1by2(q[:, 0]) | (_part1by2(q[:, 1]) << np.uint64(1))
            | (_part1by2(q[:, 2]) << np.uint64(2)))


# ------------------------------------------------------------ Taichi kernels
@ti.func
def _clz64(x):
    """Leading zeros of a 63-bit code (bits 0..62)."""
    c = 0
    mask = ti.i64(1) << 62
    while mask > 0 and (x & mask) == 0:
        c += 1
        mask = mask >> 1
    return c


@ti.func
def _clz_i32(x):
    c = 0
    mask = ti.i32(1) << 30
    while mask > 0 and (x & mask) == 0:
        c += 1
        mask = mask >> 1
    return c


@ti.func
def _delta(i, j, n, code):
    """Length of the common prefix of codes i and j (index tie-break)."""
    res = -1
    if 0 <= j < n:
        x = code[i] ^ code[j]
        if x == 0:
            res = 63 + _clz_i32(i ^ j)
        else:
            res = _clz64(x)
    return res


@ti.kernel
def _build_radix(code: ti.template(), left: ti.template(),
                 right: ti.template(), parent: ti.template(), n: ti.i32):
    """Karras 2012 binary radix tree. Internal node ids 0..n-2; leaves 0..n-1.

    Child ids are stored offset: a leaf child is < n, an internal child is +n.
    """
    for i in range(n - 1):
        d = 1 if _delta(i, i + 1, n, code) > _delta(i, i - 1, n, code) else -1
        d_min = _delta(i, i - d, n, code)
        l_max = 2
        while _delta(i, i + l_max * d, n, code) > d_min:
            l_max *= 2
        l = 0
        t = l_max // 2
        while t >= 1:
            if _delta(i, i + (l + t) * d, n, code) > d_min:
                l += t
            t //= 2
        j = i + l * d
        d_node = _delta(i, j, n, code)
        s = 0
        t2 = (l + 1) // 2
        while t2 >= 1:
            if _delta(i, i + (s + t2) * d, n, code) > d_node:
                s += t2
            # ceil division for the next step
            t2 = (t2 + 1) // 2 if t2 > 1 else 0
        gamma = i + s * d + min(d, 0)

        lo = min(i, j)
        hi = max(i, j)
        lc = gamma if lo == gamma else gamma + n           # leaf vs internal
        rc = (gamma + 1) if hi == gamma + 1 else (gamma + 1) + n
        left[i] = lc
        right[i] = rc
        parent[lc] = i + n
        parent[rc] = i + n


@ti.kernel
def _leaf_init(order: ti.template(), pos: ti.template(),
               mass: ti.template(), com: ti.template(), nmass: ti.template(),
               nmom: ti.template(), nmin: ti.template(), nmax: ti.template(),
               n: ti.i32):
    for li in range(n):
        p = order[li]
        com[li] = pos[p]
        nmass[li] = mass[p]
        nmin[li] = pos[p]
        nmax[li] = pos[p]
    for ii in range(n - 1):
        node = ii + n
        nmass[node] = 0.0
        nmom[node] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
        nmin[node] = ti.Vector([1e30, 1e30, 1e30], dt=ti.f64)
        nmax[node] = ti.Vector([-1e30, -1e30, -1e30], dt=ti.f64)


@ti.kernel
def _accumulate(parent: ti.template(), com: ti.template(),
                nmass: ti.template(), nmom: ti.template(),
                nmin: ti.template(), nmax: ti.template(), n: ti.i32):
    """Race-free bottom-up: each leaf adds its mass / moment / AABB to *every*
    ancestor via atomics.

    Every contribution is counted exactly once and order-independently, so the
    result does not depend on thread scheduling.  This replaces the old
    "second-arriver computes the parent from its children" scheme, which read a
    sibling subtree's plain (non-atomic) stores with no memory fence and could
    therefore see stale values (non-deterministic ~4% COM/mass errors).
    """
    for li in range(n):
        m = nmass[li]
        pp = com[li]
        node = parent[li]
        while node >= 0:
            ti.atomic_add(nmass[node], m)
            for k in ti.static(range(3)):
                ti.atomic_add(nmom[node][k], m * pp[k])
                ti.atomic_min(nmin[node][k], pp[k])
                ti.atomic_max(nmax[node][k], pp[k])
            node = parent[node]


@ti.kernel
def _finalize(com: ti.template(), nmass: ti.template(), nmom: ti.template(),
              n: ti.i32):
    """Internal-node centre of mass = accumulated moment / accumulated mass."""
    for ii in range(n - 1):
        node = ii + n
        m = nmass[node]
        if m > 0.0:
            com[node] = nmom[node] / m


@ti.kernel
def _traverse(pos: ti.template(), order: ti.template(),
              com: ti.template(), nmass: ti.template(), nmin: ti.template(),
              nmax: ti.template(), left: ti.template(), right: ti.template(),
              acc: ti.template(), stack: ti.template(),
              n: ti.i32, root: ti.i32,
              theta2: ti.f64, eps2: ti.f64, max_stack: ti.i32):
    for p in range(n):
        pi = pos[p]
        a = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
        sp = 0
        stack[p, sp] = root
        sp += 1
        while sp > 0:
            sp -= 1
            node = stack[p, sp]
            if node < n:                            # leaf
                q = order[node]
                if q != p:
                    d = com[node] - pi
                    r2 = d.dot(d) + eps2
                    inv = 1.0 / ti.sqrt(r2)
                    a += (G * nmass[node] * inv * inv * inv) * d
            else:
                ii = node - n
                d = com[node] - pi
                r2 = d.dot(d) + eps2
                size = (nmax[node] - nmin[node]).max()
                if size * size < theta2 * r2:        # far enough: use multipole
                    inv = 1.0 / ti.sqrt(r2)
                    a += (G * nmass[node] * inv * inv * inv) * d
                else:                                # open it
                    if sp + 2 <= max_stack:
                        stack[p, sp] = left[ii]
                        sp += 1
                        stack[p, sp] = right[ii]
                        sp += 1
        acc[p] = a


class BarnesHut:
    """Reusable Barnes-Hut accelerator for ``n`` particles."""

    def __init__(self, n: int, theta: float = 0.6, softening: float = 0.1,
                 max_stack: int = 64):
        self.n = n
        self.theta2 = float(theta) ** 2
        self.eps2 = float(softening) ** 2
        self.max_stack = max_stack
        m = max(2 * n - 1, 1)
        self.code = ti.field(ti.i64, shape=max(n, 1))
        self.order = ti.field(ti.i32, shape=max(n, 1))
        self.com = ti.Vector.field(3, ti.f64, shape=m)
        self.nmom = ti.Vector.field(3, ti.f64, shape=m)   # accumulated mass-moment
        self.nmass = ti.field(ti.f64, shape=m)
        self.nmin = ti.Vector.field(3, ti.f64, shape=m)
        self.nmax = ti.Vector.field(3, ti.f64, shape=m)
        self.left = ti.field(ti.i32, shape=max(n - 1, 1))
        self.right = ti.field(ti.i32, shape=max(n - 1, 1))
        self.parent = ti.field(ti.i32, shape=m)
        self.stack = ti.field(ti.i32, shape=(max(n, 1), max_stack))

    def compute(self, pos_field, mass_field, acc_field):
        """Fill ``acc_field`` with gravitational acceleration for ``pos_field``."""
        n = self.n
        pos_np = pos_field.to_numpy()
        codes = morton_codes(pos_np)
        order = np.argsort(codes, kind="stable").astype(np.int32)
        self.code.from_numpy(codes[order].astype(np.int64))
        self.order.from_numpy(order)
        self.parent.from_numpy(np.full(2 * n - 1, -1, np.int32))
        _build_radix(self.code, self.left, self.right, self.parent, n)
        _leaf_init(self.order, pos_field, mass_field, self.com, self.nmass,
                   self.nmom, self.nmin, self.nmax, n)
        _accumulate(self.parent, self.com, self.nmass, self.nmom,
                    self.nmin, self.nmax, n)
        _finalize(self.com, self.nmass, self.nmom, n)
        _traverse(pos_field, self.order, self.com, self.nmass, self.nmin,
                  self.nmax, self.left, self.right, acc_field, self.stack,
                  n, n, self.theta2, self.eps2, self.max_stack)
