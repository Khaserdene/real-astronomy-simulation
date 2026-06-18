"""Bridge: read a GADGET-4 / GIZMO HDF5 snapshot into our State.

This is the "research arm" hook -- once a production cosmological or galaxy run
has been done with GADGET-4 or GIZMO (built on WSL2, see research/README.md), its
HDF5 snapshots load straight into our State and flow through the *same* Blender
export/render pipeline (``export.to_pointcloud`` -> ``blender/render_frames.py``).

GADGET HDF5 layout (the relevant bits):
    Header (attrs): Time, BoxSize, MassTable[6], NumPart_ThisFile[6]
    PartType0 (gas)  : Coordinates, Velocities, Masses?, InternalEnergy, ...
    PartType1 (halo) : Coordinates, Velocities, Masses?
    PartType4 (stars): Coordinates, Velocities, Masses?, StellarFormationTime

GADGET's default unit system (kpc/h, 1e10 Msun/h, km/s) is the same family as
ours, so positions/velocities/masses carry over directly.
"""
from __future__ import annotations

import numpy as np

from core.state import State, PTYPE_GAS, PTYPE_DM, PTYPE_STAR

# Map GADGET particle types -> our type codes (gas, halo/DM, disk, bulge, stars).
_TYPE_MAP = {0: PTYPE_GAS, 1: PTYPE_DM, 2: PTYPE_DM, 3: PTYPE_DM,
             4: PTYPE_STAR, 5: PTYPE_DM}  # 5 = black holes -> treat as point mass


def load_gadget_hdf5(path: str) -> State:
    """Read a GADGET-4/GIZMO HDF5 snapshot into a :class:`State`."""
    import h5py

    pos_all, vel_all, mass_all, type_all = [], [], [], []
    age_all = []
    with h5py.File(path, "r") as f:
        header = f["Header"].attrs
        time = float(header.get("Time", 0.0))
        mass_table = np.asarray(header.get("MassTable", np.zeros(6)))

        for ptype in range(6):
            grp = f.get(f"PartType{ptype}")
            if grp is None:
                continue
            coords = grp["Coordinates"][:]
            vels = grp["Velocities"][:]
            n = coords.shape[0]
            if "Masses" in grp:
                masses = grp["Masses"][:]
            else:                                   # mass from the header table
                masses = np.full(n, mass_table[ptype])
            pos_all.append(coords)
            vel_all.append(vels)
            mass_all.append(masses)
            type_all.append(np.full(n, _TYPE_MAP[ptype], dtype=np.int32))
            # Stellar ages (from formation scale factor/time) if present.
            if ptype == 4 and "StellarFormationTime" in grp:
                age_all.append(time - grp["StellarFormationTime"][:])
            else:
                age_all.append(np.full(n, -1.0))

    pos = np.vstack(pos_all)
    return State(
        pos=pos,
        vel=np.vstack(vel_all),
        mass=np.concatenate(mass_all),
        ptype=np.concatenate(type_all),
        ids=np.arange(pos.shape[0], dtype=np.int64),
        time=time,
        age=np.concatenate(age_all),
        meta={"ic": "gadget_import", "source": path},
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        raise SystemExit("usage: python -m research.gadget_to_state "
                         "<gadget_snap.hdf5> <out_state.h5>")
    state = load_gadget_hdf5(sys.argv[1])
    state.save(sys.argv[2])
    print(f"loaded {state.n} particles from {sys.argv[1]} -> {sys.argv[2]}")
