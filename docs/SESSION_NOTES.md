# Session notes — roadmap implementation + Barnes-Hut

Handoff for the next session. This session implemented the **entire** GUI/physics
roadmap that `README.md` previously listed as "for the next session" (§1–§7),
plus the Barnes-Hut treecode. Everything below was validated on real hardware
(Python 3.10 `.venv`, RTX 3070 / CUDA, desktop display).

## What was done

| Item | Where | Status |
|------|-------|--------|
| §1 Free particle count (spin box + presets + slow-N warning) | `gui/main.py` | ✅ |
| §6 Scenario `details` + "Details…" dialog | `core/scenarios.py`, `gui/main.py` | ✅ |
| §2 Working dir + folder resume (no-overwrite frame continuation) | `core/scene_spec.py` (new), `core/sim_controller.py`, `gui/main.py` | ✅ |
| §3 Video-editor timeline (transport, scrub, play→live) | `gui/timeline.py` (new), `gui/main.py` | ✅ |
| §4 Progress bar + off-thread render (`_AsyncTask`) | `core/sim_controller.py`, `gui/main.py` | ✅ |
| §5 Blender extension (reads `.h5`, animation playback) | `blender_addon/` (new) | ✅ code; not run in real Blender |
| §7 Object-composition scene builder (params) | `core/objects.py` (new), `gui/scene_builder.py` (new) | ✅ |
| §7 3D brush (erase/add in viewport) | `gui/brush.py` (new), `gui/viewport.py`, `core/sim_controller.py` | ✅ |
| §7 Unified galaxy (stars + **live DM** + gas + SF/feedback) | `core/ic/full_galaxy.py` (new), `core/living_galaxy_engine.py`, scenario `galaxy` | ✅ |
| Dust | render attribute from gas density (no particles) | ✅ by design |
| Barnes-Hut treecode (GPU LBVH, O(N log N)) | `core/solvers/barnes_hut.py` (new), `core/engine.py`, GUI "Fast gravity" | ✅ |

### Validation (on CUDA / RTX 3070)
- `verify_phase1.py`, `verify_gui_g1.py` — all 8 scenarios (incl. new `galaxy`) pass.
- GUI smoke (`python -m gui.main --smoke 5`) renders; scene-builder + timeline
  widget logic tested headlessly (offscreen Qt).
- Barnes-Hut vs direct N²: force rel-err ~0.6% @ θ=0.5 (0.08% @ θ=0.2), energy
  |dE/E|=4e-5 on Plummer; speedup ×3.8 (20k) / ×7.5 (50k) / ×13.7 (100k).
- Galaxy render saved to `output/galaxy_test.png` (disk + bulge + DM halo).

## Key design decisions
- **Domains are separate** in the editor: Galaxy mode vs Planetary mode (no mixing).
- **Scene spec** (`core/scene_spec.py`) is the forward-compatible persistence
  envelope — snapshot `meta["scene"]` stores the full object list + engine config
  (kind, dt, softening, gravity_mode, theta). Old files fall back to `meta["scenario"]`.
- **Live DM** reuses `LivingGalaxyEngine`: species `2`=DM in the `is_star` field;
  all SPH/SF/SN kernels gate on `is_star==0/1` so they skip DM automatically while
  gravity still includes it.

## Remaining / next session (optional)
1. **Interactive click-through validation** — launch the GUI and exercise the
   scene-builder (collide 2 galaxies), 3D brush drag, timeline scrub by hand
   (this session's `computer-use` request_access timed out).
2. **Barnes-Hut in the SPH/galaxy engine** — `LivingGalaxyEngine.grav_ext` still
   uses direct N²; hand its self-gravity to the BH treecode so the unified
   `galaxy` scenario also scales to large N.
3. **Blender extension in real Blender** — fetch the `h5py` wheel
   (`blender_addon/fetch_wheels.py`), build & install the extension, confirm
   `.h5` animation playback (`blender_addon/README.md`).

## Gotcha (important)
A Taichi-kernel module must **not** use `from __future__ import annotations` — it
stringifies `ti.template()`/`ti.i32` annotations and breaks Taichi arg extraction.
See the header of `core/solvers/barnes_hut.py`.
