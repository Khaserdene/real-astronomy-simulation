# Research arm — GADGET-4 / GIZMO (production-grade physics)

The custom Taichi engine is great for interactive, single-GPU galaxy and gas
runs.  For **research-grade** cosmology and galaxy formation (proper periodic
gravity, mature SPH/moving-mesh hydro, cooling, star formation and feedback that
have been validated against observations), use an established code and feed its
snapshots into our existing render pipeline.

These codes are MPI/C++ and Linux-native, so build them under **WSL2** (Ubuntu),
not native Windows.

## Why a separate arm

| | Custom Taichi engine | GADGET-4 / GIZMO |
|---|---|---|
| Setup | `pip install`, runs now | build C++/MPI on WSL2 |
| Boundaries | open box | fully periodic (TreePM / Ewald) |
| Hydro | textbook SPH | production SPH / moving mesh |
| Physics | SF + SN feedback (demo) | cooling, multiphase ISM, AGN, validated |
| Best for | interactive, visuals, learning | publication-grade dynamic range |

## Build GADGET-4 on WSL2 (outline)

```bash
# In Ubuntu (WSL2):
sudo apt update
sudo apt install -y build-essential git libgsl-dev libfftw3-dev \
    libhdf5-openmpi-dev openmpi-bin

git clone https://gitlab.mpcdf.mpg.de/vrs/gadget4.git
cd gadget4
cp examples/CollidingGalaxiesSFR/Config.sh .        # or a cosmological example
cp examples/CollidingGalaxiesSFR/param.txt .
# edit Config.sh to enable: SELFGRAVITY, PERIODIC (cosmology), COOLING,
#   STARFORMATION, PMGRID=..., etc.
make -j4
mpirun -np 4 ./Gadget4 param.txt
```

GIZMO (meshless finite-mass hydro) builds similarly from
<http://www.tapir.caltech.edu/~phopkins/Site/GIZMO.html>.

## Initial conditions

- Cosmological: generate with **MUSIC** or **monofonIC** (Gaussian random field
  from a CAMB/CLASS power spectrum) — the proper version of our toy
  `core/ic/cosmo.py` Zel'dovich box.
- Isolated/merging galaxies: **makeNewDisk / galstep / pyICs**.

## Feed snapshots into our render pipeline

GADGET/GIZMO write HDF5 snapshots.  Convert one to our `State`, then reuse the
same export + Blender render as the custom engine:

```bash
# (Windows side, in the venv)
.venv/Scripts/python.exe -m research.gadget_to_state \
    /path/snapshot_010.hdf5 output/gadget/snap_0000.h5
.venv/Scripts/python.exe -m export.to_pointcloud output/gadget
"/c/Program Files/Blender Foundation/Blender 4.5/blender.exe" --background \
    --python blender/render_frames.py -- --ply output/gadget/ply \
    --out output/gadget/render
```

`research/gadget_to_state.py` maps PartType0/1/4 (gas/halo/stars) onto our type
codes and carries stellar ages, so the blackbody colouring and volume rendering
work unchanged.

> Status: the bridge (`gadget_to_state.py`) is implemented against the documented
> GADGET-4 HDF5 layout; the GADGET/GIZMO build itself must be done on the user's
> WSL2 install (not executed in this environment).
