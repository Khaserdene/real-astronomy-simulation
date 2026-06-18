"""Full simulation state container + portable HDF5 (de)serialization.

A :class:`State` holds everything needed to *resume* a run bit-for-bit on any
machine: particle phase-space coordinates, masses, per-particle attributes, the
current simulation time / step counter, and the RNG seed that generated the ICs.

Everything is stored in the physical galactic units defined in :mod:`core.units`,
so a checkpoint written on one computer (or at one resolution level) reloads
unchanged on another.  The same file format doubles as a render *snapshot* in
Phase 1; lighter render-only snapshots can be derived later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Particle type codes (GADGET-flavoured).
PTYPE_DM = 0
PTYPE_STAR = 1
PTYPE_GAS = 2


@dataclass
class State:
    """Mutable container for the full simulation state (all in code units)."""

    pos: np.ndarray            # (N, 3) float64  positions [kpc]
    vel: np.ndarray            # (N, 3) float64  velocities [km/s]
    mass: np.ndarray           # (N,)   float64  masses [1e10 Msun]
    ptype: np.ndarray          # (N,)   int32    particle type code
    ids: np.ndarray            # (N,)   int64    unique particle id

    time: float = 0.0          # current simulation time [code units]
    step: int = 0              # number of integration steps taken
    seed: int = 0              # RNG seed used to build the ICs

    # Optional per-particle attributes (filled in later phases).
    age: Optional[np.ndarray] = None          # [Gyr]
    temperature: Optional[np.ndarray] = None  # [K]
    metallicity: Optional[np.ndarray] = None  # [Z, dimensionless]

    # Free-form provenance (level name, IC type, parameters...).
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.pos.shape[0]

    def __post_init__(self) -> None:
        # Normalise dtypes / shapes so downstream Taichi copies are predictable.
        self.pos = np.ascontiguousarray(self.pos, dtype=np.float64).reshape(-1, 3)
        self.vel = np.ascontiguousarray(self.vel, dtype=np.float64).reshape(-1, 3)
        self.mass = np.ascontiguousarray(self.mass, dtype=np.float64).reshape(-1)
        self.ptype = np.ascontiguousarray(self.ptype, dtype=np.int32).reshape(-1)
        self.ids = np.ascontiguousarray(self.ids, dtype=np.int64).reshape(-1)

    # ------------------------------------------------------------------ I/O
    def save(self, path: str) -> None:
        """Write the full state to an HDF5 file."""
        import h5py

        with h5py.File(path, "w") as f:
            f.attrs["time"] = float(self.time)
            f.attrs["step"] = int(self.step)
            f.attrs["seed"] = int(self.seed)
            f.attrs["n"] = int(self.n)
            f.attrs["units"] = "kpc, km/s, 1e10 Msun"
            for k, v in self.meta.items():
                # HDF5 attrs accept scalars/strings; stringify anything else.
                f.attrs[f"meta_{k}"] = v if np.isscalar(v) else str(v)

            f.create_dataset("pos", data=self.pos, compression="gzip")
            f.create_dataset("vel", data=self.vel, compression="gzip")
            f.create_dataset("mass", data=self.mass, compression="gzip")
            f.create_dataset("ptype", data=self.ptype, compression="gzip")
            f.create_dataset("ids", data=self.ids, compression="gzip")
            for name in ("age", "temperature", "metallicity"):
                arr = getattr(self, name)
                if arr is not None:
                    f.create_dataset(name, data=np.asarray(arr, dtype=np.float64),
                                     compression="gzip")

    @classmethod
    def load(cls, path: str) -> "State":
        """Read a full state back from an HDF5 file."""
        import h5py

        with h5py.File(path, "r") as f:
            meta = {k[len("meta_"):]: f.attrs[k] for k in f.attrs
                    if k.startswith("meta_")}
            optional = {}
            for name in ("age", "temperature", "metallicity"):
                if name in f:
                    optional[name] = f[name][:]
            return cls(
                pos=f["pos"][:],
                vel=f["vel"][:],
                mass=f["mass"][:],
                ptype=f["ptype"][:],
                ids=f["ids"][:],
                time=float(f.attrs["time"]),
                step=int(f.attrs["step"]),
                seed=int(f.attrs["seed"]),
                meta=meta,
                **optional,
            )
