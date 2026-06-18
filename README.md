# Real Astronomy Simulator

GPU N-body / hydrodynamics galaxy simulator that writes frame-by-frame snapshots
for offline rendering in Blender. Built for consumer hardware (developed on an
RTX 4070 SUPER) but resolution-scalable via config — not locked to any one machine.

See [`plan`](../../../.claude/plans/virtual-soaring-lampson.md) for the full
roadmap. Status: **Phase 1 complete** (headless N-body core).

**New here? Read [GUIDE.md](GUIDE.md)** — step-by-step usage (setup, running every
scenario, the editor, and rendering), with copy-paste commands.

**Just want to run it?** Double-click **`run.bat`** (Windows) — it sets up the
environment on first run and opens the desktop app.  Or `.venv/Scripts/python.exe
-m gui.main`.  The GUI (PyQt6 + a live 3D viewport) covers every scenario,
snapshot playback, and one-click hand-off to Blender (interactive or headless).
`run.bat --cli ...` runs the headless CLI instead.  Plan: [docs/GUI_PLAN.md](docs/GUI_PLAN.md).

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
| `core/solvers/gravity.py` | Direct N² gravity (Taichi GPU) |
| `core/solvers/barnes_hut.py` | Barnes-Hut treecode (LBVH, O(N log N)) — `gravity_mode="bh"` |
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
- [x] Phase 8 — unified multi-component galaxy (`core/ic/full_galaxy.py`,
      scenario `galaxy`): stars + a **live** dark-matter halo + SPH gas, all
      self-gravitating, with star formation + SN feedback (dust added at render).
- [x] Phase 9 — Barnes-Hut treecode gravity (`core/solvers/barnes_hut.py`,
      `Engine(gravity_mode="bh")`, GUI "Fast gravity" toggle): GPU LBVH (Karras)
      tree, O(N log N). Validated vs direct (force err ~0.6% @ θ=0.5, energy
      conserved); ~14× faster at N=100k. Wired for gravity scenarios; the
      live-galaxy/SPH self-gravity still uses N² (BH hand-off there is next).

---

## Roadmap — ✅ бүгд хэрэгжсэн (this list is now DONE)

> §1–§7 (+ Barnes-Hut) бүгд **хэрэгжиж, RTX 3070 дээр баталгаажсан**. Дэлгэрэнгүй
> тайлан, баталгаажуулалт, үлдсэн нэмэлт ажлуудыг
> [docs/SESSION_NOTES.md](docs/SESSION_NOTES.md)-ээс үзнэ үү. Доорх анхны
> жагсаалтыг түүхэн лавлагаа болгон үлдээв.

### 1. Particles — дурын тоо оруулах
- Одоо зөвхөн preset (5k/10k/20k/40k)-оос сонгоно — **учир дутагдалтай**.
- Хэрэглэгч **хүссэн тоогоо** шууд бичиж оруулдаг болгох (`QSpinBox`/`QLineEdit`,
  жишээ нь 1,000–2,000,000). `gui/main.py` `_N_PRESETS` → чөлөөт тоон оролт.

### 2. Симуляцийн ажлын хавтас (working directory) сонгох
- Симуляц эхлүүлэхийн **өмнө хавтас сонгоно**; үүсэх бүх файл **зөвхөн тэнд**.
- Шалтгаан: `C:` дүүрсэн (disk not enough space) тохиолдол байж болно → хэрэглэгч
  `D:\sim` гэх мэт хавтас сонгож, бүх гаралт тэнд үүснэ.
- Бүх фрэйм тэр хавтсанд хадгалагдана.
- Дараа нь **өөр төхөөрөмж эсвэл өөр цагт** тэр хавтсыг ачаалж, **хамгийн сүүлийн
  фрэймээс цааш үргэлжлүүлэх**. (Snapshot-ууд аль хэдийн зөөвөрлөгддөг — §checkpoint;
  GUI-д "Open simulation folder" → сүүлийн snap-аас resume логик нэмэх.)
- **Output section-ийг шинэчлэх** (хавтас сонгогч + folder-аас ачаалах/үргэлжлүүлэх).
- **Зөрчлийн логик:** ачаалсан симуляцийн **particle тоог өөрчилж Build дарвал**
  → "Энэ нь өмнөх фрэймүүдтэй залгамжлах боломжгүй. Эхнээс нь дахин эхлүүлж
  **overwrite** хийх үү?" гэж **асуух** (`QMessageBox`). Учир нь N өөрчлөгдөхөд
  хуучин файлуудтай зөрчилдөнө.

### 3. Timeline-ийг сайжруулах (видео засварын програм шиг)
- Тусдаа **дэлгэцийн доод хэсэгт**, видео editor шиг — фрэймүүд харагдаж, playback
  бүхий зурвас.
- Симуляцыг **дахин тоглуулах** боломжтой.
- Курсорыг **гүйлгэх/чирэх** үед тухайн фрэймд очдог; бүх scrub/drag логикийг
  сайжруулах.
- **Сүүлийн фрэйм дээр очих / симуляц хийгдээгүй үед → симуляц үргэлжилнэ**
  (playback дуусахад live run руу шилжих).

### 4. Прогресс бар (фрэйм боловсруулалт 0–100%)
- Симуляцийн тухайн фрэймийн тооцоолол **0–100%** прогресс бараар харагдах.
- Юу болж, ямар байдалтай байгааг хэрэглэгч мэдэж байх. (Алхмууд/фрэйм дэх
  явцыг `QProgressBar`-аар; G6-ийн threading-тэй хослуулбал зүйтэй.)

### 5. Blender хэсгийг эргэцүүлэх
- **Одоогийн зан төлөв (тодруулга):**
  - *Open in Blender* — **суусан Blender-ийг** (`gui/blender_launcher.find_blender`,
    эсвэл `BLENDER` env) GUI горимоор нээж, `blender/template_scene.py`-ээр
    **зөвхөн нэг (одоогийн) фрэймийг** объект+материалаар ачаална. Анимэйшн биш.
  - *Quick render* — мөн суусан Blender-ийг **`--background`**-аар нээж,
    `render_frames.py`-ээр одоогийн фрэймийг PNG болгож рэндэрлэнэ.
- **Хүссэн сайжруулалт:** *Open in Blender* үед симуляцийн **хавтастай холбогдож,
  бүх фрэймүүдийн анимэйшныг Blender дотор тоглуулах** боломжтой байх. (Сонголтууд:
  бүх фрэймийг mesh-sequence болгох; эсвэл `frame_change` handler-аар геометрийг
  фрэйм бүрт солих startup script; эсвэл **Blender extension/add-on** хийх — Blender
  4.2+ extensions платформыг судлах. Сүүлийн Blender update-д юу хэрэгтэйг шалгах.)

### 6. Scenario тус бүрийн тайлбар
- GUI-д scenario бүрийн **дэлгэрэнгүй тайлбар** (физик, юу болдог, параметрүүд).
  Одоо `core/scenarios.py`-д богино `description` бий — өргөтгөх, GUI-д тод харуулах.

### 7. Scenario build функц + editor доторх багажууд
- Одоо байгаа **editor дотор** IC-г интерактивээр зохиох багажууд нэмэх (тусдаа
  биш, одоогийн GUI/editor дотор):
  - **Particle brush** — партиклуудыг зураач шиг нэмэх/арилгах.
  - **Анхны хурд** (initial velocity) тохируулах.
  - **Эргэлтийн момент** (angular momentum / spin) өгөх.
  - IC-ийн бусад шаардлагатай бүх параметр/логик — **ярилцаж** тодорхойлох.

### 8. Дээрх бүгдийн логикийг сайжруулж ярилцах
- Эдгээрийг хэрэгжүүлэхийн өмнө архитектур/UX-ийг хэрэглэгчтэй **дэлгэрэнгүй
  ярилцаж** тодруулах (ялангуяа §2 folder/resume, §3 timeline, §7 editor багажууд).
