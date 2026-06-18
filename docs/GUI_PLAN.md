# GUI plan — from CLI to a full desktop application

Goal: a real graphical interface so the whole workflow (choose scenario, set
parameters, run with a live 3D view, checkpoint, and render) happens in a window
instead of on the command line.  The CLI stays for headless/scripted/server use.

## Decisions (agreed)

- **Framework**: PyQt6 + pyqtgraph (`GLViewWidget` for the live 3D viewport).
- **Scope**: full — simulation + checkpoint + snapshot playback + rendering.
- **Engines**: unify all (gravity, gas, living galaxy, impact, cosmo) behind one
  interface so every scenario is runnable from the GUI.
- **Blender**: both modes, **interactive handoff primary** — "Open in Blender"
  loads the frames + a per-type starter material scene into Blender's GUI for
  hands-on art-direction; "Quick render" keeps the headless one-click path.

## Architecture

```
gui/ (new, PyQt6)            -> thin view; never touches Taichi directly
  main.py                    -> QApplication + MainWindow + sim QTimer/worker
  viewport.py                -> pyqtgraph GLViewWidget (particles from controller)
  panels/ scenario, run, output, render, diagnostics
editor/controller.py (reuse) -> backend "brain" (extended for all engines)
core/scenarios.py (new)      -> registry: name -> (IC builder, engine, params, dt)
core/engine_base.py (new)    -> common engine interface all engines satisfy
export/ (extended)           -> PLY/VDB with physical attributes per particle
blender/template_scene.py(new)-> interactive handoff (opens Blender GUI w/ data)
```

The existing **`EditorController`** already decouples GUI from physics
(`build/run/step/checkpoint/export/display_arrays`).  The GUI is a view on it.

## Key design points

- **Unified engine interface** (`engine_base.py`): every engine exposes
  `setup(state)/step(dt)/to_state()->State` plus `n, time, step_count`.  Gas/SPH
  engines carry internal energy `u` and density `rho` on the `State` (extend
  `State` with optional `u`, `rho`).  Wrappers adapt the existing engines.
- **Scenario registry** (`scenarios.py`): one place mapping each scenario to its
  IC, engine, default parameters, and a sensible `dt` — drives both the GUI
  scenario menu and (optionally) the CLI.
- **Attribute-rich export**: write per-particle attributes (type, temperature,
  density, age, internal energy, speed) so Blender materials can be *driven by
  physics* (e.g. "young stars blue", "hot gas red") via Attribute nodes.
- **Interactive Blender** (`template_scene.py`): launched non-headless; imports a
  frame, builds separate objects + named materials per type (stars = emissive
  point cloud by temperature; gas = volume by density/temperature; dust/DM =
  faint), sets camera/world.  The artist tweaks and renders inside Blender.
- **Live viewport**: a `QTimer` advances the sim a few steps per tick and pushes
  `controller.display_arrays()` into a pyqtgraph `GLScatterPlotItem`.  Heavy
  stepping moves to a `QThread` worker so the UI stays responsive.

## Build phases

- **G1 — Engine unification** (backend, headless-testable): `engine_base.py`,
  `scenarios.py`, extend `State` (u/rho), wrap all engines, extend
  `EditorController` to run any scenario.  Verify headless.
- **G2 — Attribute export + Blender template**: rich PLY export; `template_scene.py`
  interactive handoff; keep headless render.
- **G3 — PyQt skeleton + viewport**: MainWindow, pyqtgraph viewport, scenario +
  run panels, live preview, build/run/pause/step.
- **G4 — Output, checkpoint, timeline**: save/load dialogs, output folder,
  snapshot playback scrubber.
- **G5 — Render integration**: "Open in Blender" + "Quick render" + rendered-frame
  viewer.
- **G6 — Diagnostics + polish**: energy/star-count plots, threaded stepping,
  optional packaging to a standalone app.

## New dependencies

`PyQt6`, `pyqtgraph` (added to `requirements.txt`).

## Verification

- Headless test of the unified controller across all scenarios (G1).
- Launch-and-screenshot smoke test of the PyQt app (G3+).
- Manual: build each scenario, run, scrub timeline, Open in Blender, render.
```
