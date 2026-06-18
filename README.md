# Real Astronomy Simulator

GPU N-body / hydrodynamics galaxy simulator that writes frame-by-frame snapshots
for offline rendering in Blender. Built for consumer hardware (developed on an
RTX 4070 SUPER) but resolution-scalable via config — not locked to any one machine.

See [`plan`](../../../.claude/plans/virtual-soaring-lampson.md) for the full
roadmap. Status: **Phase 1 complete** (headless N-body core).

**New here? Read [GUIDE.md](GUIDE.md)** — step-by-step usage (setup, running every
scenario, the editor, and rendering), with copy-paste commands.

**Prefer a window over the terminal?** Launch the desktop app:
`.venv/Scripts/python.exe -m gui.main` — PyQt6 + a live 3D viewport, every
scenario, snapshot playback, and one-click hand-off to Blender (interactive or
headless). Plan/architecture: [docs/GUI_PLAN.md](docs/GUI_PLAN.md).

## Setup

```bash
py -3.10 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Requires a CUDA GPU (falls back to `--arch cpu`). Python 3.10 (Taichi 1.7.x).

## Quick start

```bash
# Verify the engine (energy conservation, virial, I/O round-trip, disk plot)
.venv/Scripts/python.exe verify_phase1.py

# Run a disk galaxy, writing one snapshot every 20 steps into output/run1/
.venv/Scripts/python.exe cli.py --ic disk --n 20000 --steps 2000 \
    --dt 1e-4 --snap-every 20 --out output/run1

# Resume an interrupted run from any snapshot (same resolution)
.venv/Scripts/python.exe cli.py --resume output/run1/snap_0040.h5 \
    --steps 2000 --dt 1e-4 --snap-every 20 --out output/run1
```

## Interactive editor (GUI)

A standalone Taichi-GGUI editor with a live 3D viewport: build a galaxy,
run/pause it, orbit the camera (hold right-mouse), checkpoint/resume, promote a
preview to production, and export frames for Blender — all from one window.

```bash
.venv/Scripts/python.exe -m editor.app
```

(Requires a desktop display. The headless engine, CLI, and Blender render do not.)

## Rendering in Blender

Two steps: convert snapshots to coloured point clouds (venv), then batch-render
them in Blender (headless). Star colour comes from a blackbody temperature.

```bash
# 1) snapshots -> coloured PLY point clouds (writes <run>/ply/)
.venv/Scripts/python.exe -m export.to_pointcloud output/galaxy

# 2) PLY -> rendered PNGs (Cycles GPU + fog-glow bloom)
"/c/Program Files/Blender Foundation/Blender 4.5/blender.exe" --background \
    --python blender/render_frames.py -- \
    --ply output/galaxy/ply --out output/galaxy/render \
    --res 1280 --samples 48 --radius 0.1 --emission 2.5 --cam-elev 72
```

`--cam-elev` sets the viewing angle (90 = face-on, 0 = edge-on).

Snapshots are portable HDF5 files in physical galactic units (kpc, km/s,
1e10 Msun), so a run started on one computer resumes unchanged on another.

## Layout

| Path | Purpose |
|------|---------|
| `core/` | Headless engine: units, state I/O, gravity, integrators, IC generators |
| `core/solvers/gravity.py` | Direct N² gravity (Taichi GPU) — Barnes-Hut tree to come |
| `core/solvers/test_particle.py` | Fast "Preview" tier: tracers in a fixed analytic potential |
| `core/ic/` | Deterministic initial conditions (Plummer sphere, exponential disk) |
| `cli.py` | Headless runner → frame-by-frame snapshots |
| `editor/`, `export/`, `blender/` | GUI, Blender export, render (later phases) |

## Status / next

- [x] Phase 0 — environment (Taichi CUDA on RTX 4070 SUPER verified)
- [x] Phase 1 — N-body core, disk/Plummer ICs, snapshots, resume, verification
- [x] Phase 2 — level presets + hardware auto-detect, Simulation driver, resume
- [x] Phase 3 — standalone Taichi GGUI editor (live viewport + controls)
- [x] Phase 4 — Blender export (PLY + blackbody colour) + headless batch render
- [x] Phase 5a — SPH gas solver, validated on the Sod shock tube (`verify_phase5.py`)
- [x] Phase 5b — gas-disk engine (SPH + self-gravity + analytic potential) +
      Blender Points-to-Volume render (`core/gas_disk_engine.py`, `blender/render_gas.py`)
- [x] Phase 6a — galaxy mergers (`core/ic/merger.py`, `--ic merger`): tidal tails + bridges
- [x] Phase 6b — planet collisions / giant impact (`core/ic/impact.py`, SPH + self-gravity):
      shock-heated ejecta coloured by internal energy
- [x] Phase 6c — living galaxy (`core/living_galaxy_engine.py`): star formation,
      stellar ageing (age->colour), supernova feedback (self-regulated SF, gas fountains)
- [x] Phase 7 — cosmological structure formation (`core/ic/cosmo.py`, `--ic cosmo`):
      Zel'dovich box -> cosmic web; GADGET/GIZMO research bridge (`research/`)
