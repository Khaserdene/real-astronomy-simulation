# Handoff → next session

Scope of the session that produced this file: finished the two performance items
the previous handoff (`SESSION_NOTES.md`) left open for the galaxy engine —
**Barnes-Hut self-gravity** and a **uniform-grid SPH neighbour search** in
`LivingGalaxyEngine` — and answered a batch of questions about how dark matter
and the gravity solver actually work. Everything here was run on the real
`.venv` (Python 3.10, Taichi GPU/CUDA).

---

## 1. What was done this session

### Barnes-Hut self-gravity in the galaxy/living engine
`LivingGalaxyEngine` used to compute self-gravity with a direct O(N²) loop
(`grav_ext`). That kernel was split into three:

- `grav_self_direct` — direct N² self-gravity (small-N fallback),
- `grav_analytic` — the analytic disk+halo potential (a no-op when `M_d==M_h==0`,
  i.e. the fully-live galaxy), always called after self-gravity,
- `add_acc` — folds a separately-computed gravity field into `acc`.

`setup()` now builds a `BarnesHut` accelerator when `gravity_mode=="bh"` and
`n>=256`; `_forces()` writes the tree gravity into a scratch `acc_g` field and
adds it in. `gravity_mode`/`theta` thread through `build_scenario` and
`resume_scenario` for the `living` + `galaxy` kinds (so the GUI "Fast gravity"
checkbox + Physics dialog now affect those, not just the pure-gravity scenarios).

Verified by `verify_bh_galaxy.py`: force rel-err mean ≈2.3% @ θ=0.6 on galaxy
ICs; KE matches direct within ≈0.5% after 20 steps; gravity-only speedup ×5.7 @
30k, ×9.7 @ 60k particles.

### Uniform-grid SPH neighbour search
The galaxy step's SPH (`gas_density`, `gas_forces`, `feedback`) was also O(N²).
Added `core/solvers/neighbor_grid.py` (`NeighborGrid`): a host-side counting-sort
that bins the **gas** particles into a uniform grid whose cell size is the SPH
search radius `max(2·h_max, r_fb)`, so a 3×3×3 stencil is provably exact. GPU
traversal kernels `gas_density_grid` / `gas_forces_grid` / `feedback_grid` live in
`living_galaxy_engine.py`. `setup()` builds the grid for `n>=2000`; it is rebuilt
each density pass (h changes) and reused by forces+feedback within the same step.

Verified by `verify_grid_sph.py`: with deterministic direct gravity the grid path
matches the N² path to **machine precision** (density 9e-16, accel 2e-16) and
steps stably. (Test uses `gravity_mode="direct"` on purpose — see the BH bug
below for why BH can't be the reference.)

---

## 2. Answers to the session's questions (DM + gravity solver)

### Are all dark-matter (DM) components analytic, or where do they run as real particles?
Both exist, per scenario:

| Scenario | DM / halo representation |
|----------|--------------------------|
| `disk`, `merger`, `cosmo`, `plummer`, `galaxy` | **Live particles** (`PTYPE_DM`) — a Plummer halo of real particles that sources and feels gravity. `galaxy` carries live DM as engine species `2`. |
| `gas`, `living` | **Analytic** — no DM particles; a fixed Miyamoto-Nagai disk + Plummer halo *potential* stands in for the halo (`grav_analytic` in the living engine, `add_grav_ext` in `GasDiskEngine`, halo mass `M_h`). |
| `impact` | No halo at all (`_NO_POTENTIAL`). |

So: the classic **`living`** galaxy uses an **analytic** halo; the unified
**`galaxy`** scenario uses a **live particle** halo.

### Was Barnes-Hut running on everything? Which engines support it?
- `Engine` (gravity kinds: `disk`, `merger`, `cosmo`, `plummer`): BH supported, `n>=256`.
- `LivingGalaxyEngine` (`living`, `galaxy`): BH supported **as of this session**,
  `n>=256`, over **all** particles including live DM.
- `GasDiskEngine` (`gas`, `impact`): **no BH** — always direct N² gas self-gravity
  plus the analytic potential.

### Unchecked → N², checked → Barnes-Hut?
Yes. `gravity_mode=="direct"` ⇒ direct N²; `"bh"` ⇒ treecode. Below the per-engine
N threshold (`256` for gravity/living/galaxy) it silently falls back to N² even
when "bh" is selected, because the tree isn't worth it for tiny systems.

### Can you switch freely between them mid-run, or must you rebuild?
**You must rebuild.** The GUI checkbox / Physics dialog only write the
`gravity_mode` (and `theta`) value onto the controller's `Settings`. The actual
solver is chosen **once, at Build/Resume**, when the engine is constructed and the
`BarnesHut` object is allocated in `setup()`/`load_state()`. Toggling the checkbox
during a run changes nothing until you press **Build** again — there is no hot
switch. (The checkbox label already says "applies on Build".)

---

## 3. Open items for the next session

1. **✅ FIXED (2026-06-19) — Barnes-Hut non-determinism.**
   `core/solvers/barnes_hut.py` `_bottom_up` had a memory-visibility race (a
   thread finalizing a parent read a sibling subtree's plain non-atomic
   `com`/`nmass` with no fence → stale, ~4% non-deterministic). **Fixed** with
   option (a): the flag/second-arriver scheme was replaced by race-free atomic
   accumulation — `_accumulate` has each leaf `atomic_add` its mass/moment and
   `atomic_min/max` its AABB into *every* ancestor, then `_finalize` sets
   internal COM = moment/mass (`flag` removed, `nmom` added). Verified on CUDA:
   two `compute()` calls now agree to 5.8e-11 (was ~4%), root mass exact (10.0,
   was 9.956), force median err 0.47%, `verify_bh_galaxy.py` KE within 0.19% of
   direct, all 8 scenarios + `verify_grid_sph.py` still pass. BH is now safe as a
   deterministic reference. (Change is local-only — not yet committed/pushed.)

2. **SPH still doesn't reach the grid's potential for the full step.** The grid is
   correct and exact, but the per-density-pass `rebuild()` pulls `pos`/`h`/`is_star`
   to the host each call (numpy counting sort). For very large N consider moving the
   bin/sort onto the GPU (atomic histogram + prefix sum) to drop the host transfer.
   Also: the grid currently bins **gas only**; gravity (BH) is what scales the
   star/DM majority. That split is intentional but worth re-checking if star
   fractions get very high.

3. **Hot-switch the gravity solver without a rebuild (UX).** Today switching
   `direct`↔`bh` needs a Build. To switch live, `LivingGalaxyEngine` / `Engine`
   would need a setter that (re)allocates or frees `_bh` and recomputes `_forces`
   on the current state. Also update the checkbox tooltip — it still says
   "(gravity scenarios)" but BH now applies to `living`/`galaxy` too.

4. **Carried over from `SESSION_NOTES.md` (still open):**
   - Interactive click-through validation of the GUI (scene-builder collide-two-
     galaxies, 3D brush drag, timeline scrub) — the earlier `computer-use`
     request timed out.
   - Run the Blender extension in real Blender (`blender_addon/fetch_wheels.py`,
     then install + confirm `.h5` animation playback).

---

## 4. Verification scripts (all green on CUDA)
- `verify_grid_sph.py` — grid SPH == N² to round-off, stable stepping.
- `verify_bh_galaxy.py` — galaxy BH self-gravity vs direct, error + speedup.
- `verify_gui_g1.py` — all 8 scenarios build/step/display/resume.

## 5. Gotcha (still applies)
Taichi-kernel modules must **not** `from __future__ import annotations` — it
stringifies `ti.template()`/`ti.i32` annotations and breaks Taichi arg
extraction (`core/solvers/barnes_hut.py`, `core/living_galaxy_engine.py`).
