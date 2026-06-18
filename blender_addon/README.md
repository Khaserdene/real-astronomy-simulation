# Astronomy Sim Importer — Blender extension

Load a simulation output folder (the one full of `snap_*.h5`) into Blender as an
**animation**: each particle type becomes an art-directable object (stars =
emissive points coloured by blackbody temperature, gas = Principled Volume by
density, dark matter = faint points), and a frame handler swaps the geometry to
the right snapshot for each Blender frame. Press **Spacebar** to play, scrub the
timeline, tweak materials/lighting/camera, and render like any other animation.

It reads the engine's `.h5` snapshots **directly** — no pre-conversion step.

## Install

1. **Get the h5py wheel** (Blender ships `numpy` but not `h5py`). From this
   folder, using *any* Python:

   ```bash
   python fetch_wheels.py --pyver 311 --platform win_amd64
   ```

   Use the CPython tag matching your Blender (Blender 4.2 → `311`) and your OS
   platform tag (`win_amd64`, `macosx_11_0_arm64`, `manylinux2014_x86_64`, …).
   Then **uncomment and edit** the `wheels = [...]` line in
   `blender_manifest.toml` to list the downloaded file(s).

2. **Build the extension zip** (Blender 4.2+ ships the builder):

   ```bash
   blender --command extension build --source-dir . --output-dir .
   ```

   This produces `astro_sim_importer-0.1.0.zip` with the wheel bundled.

3. **Install in Blender**: Edit → Preferences → Get Extensions →
   ▾ → *Install from Disk…* → pick the zip. (Or just drag the zip into Blender.)

> Skipping the wheel step? The extension still installs, but Import will report
> that `h5py` is unavailable until you bundle the wheel (or `pip install h5py`
> into Blender's Python).

## Use

**Standalone (in Blender):** open the **N-sidebar** in the 3D viewport → **Astro
Sim** tab → choose your simulation folder → **Import simulation folder**. Press
Spacebar to play.

**From the desktop GUI:** the app's *Open in Blender (animation)* button launches
Blender running `loader.py` on the current run's folder — same result without
installing the extension (it still needs `h5py` available to Blender, so bundling
the wheel once via the extension is the simplest path).

## Files

| File | Purpose |
|------|---------|
| `blender_manifest.toml` | Extension metadata + bundled wheels list |
| `__init__.py` | Panel + Import operator (Blender UI) |
| `loader.py` | `.h5` reading, per-type objects + materials, frame handler |
| `fetch_wheels.py` | Download the matching `h5py` wheel into `wheels/` |
