# Real Astronomy Simulator — бүрэн тайлбар (бүх түвшин)

> Энэ баримт нь төсөл **яаж ажилладаг**, **ямар технологи/техник** ашигладаг,
> **юуг чадна / юуг чадахгүй**, мөн **ирээдүйд хэрхэн хувьсах** (аль шинэчлэлт
> юу авчрах, юунд хохирох) талаар дээдээс доош нь тайлбарлана.

---

## 0. Нэг өгүүлбэрээр

Энэ бол **GPU дээр ажилладаг одон орны N-body + гидродинамик симулятор**: галактик,
галактикийн мөргөлдөөн, орчлонгийн бүтэц, гараг мөргөлдөх зэргийг физикийн хуулиар
тооцоолж, фрэйм фрэймээр хадгалаад, Blender-т фото-реалистик рэндэр хийдэг.
Хэрэглэгч PyQt6 десктоп аппаас бүгдийг удирдана.

---

## 1. Өндөр түвшний тойм — юу хийдэг вэ

1. **Initial Conditions (IC)** үүсгэнэ — эхний байрлал/хурд (диск галактик, Plummer
   бөмбөрцөг, мөргөлдөх 2 галактик, орчлонгийн хайрцаг, хийн диск, гараг г.м.).
2. **Engine** цаг хугацааны дагуу алхам алхмаар хөгжүүлнэ (таталцал + хий).
3. **Snapshot** (`.h5`) фрэйм бүрээр хадгална — физик нэгжтэй, зөөвөрлөгддөг.
4. **Viewport** дээр амьдаар харуулна (pyqtgraph 3D), эсвэл хадгалсан фрэймийг
   timeline-аар тоглуулна.
5. **Blender** руу дамжуулж кино чанарын рэндэр хийнэ (extension эсвэл headless).

Хоёр ажиллах горим:
- **CLI / headless** — дэлгэцгүй, серверт тохиромжтой (`cli.py`).
- **GUI** — амьд 3D харагдацтай десктоп апп (`gui/main.py`, `run.bat`).

---

## 2. Технологийн стек (юу, яагаад)

| Технологи | Үүрэг | Яагаад |
|-----------|-------|--------|
| **Taichi** (1.7, CUDA backend) | Бүх хүнд тооцоол (таталцал, SPH) GPU дээр | Python дотроос GPU kernel бичих, `@ti.kernel`. C++/CUDA бичихгүйгээр RTX-ийн хүчийг ашиглана. `default_fp=f64` — таталцалд нарийвчлал чухал |
| **NumPy** | IC генерац, Morton код, grid bin, диагностик | Вектор тооцоол, GPU-д орохын өмнөх/дараах боловсруулалт |
| **h5py / HDF5** | Snapshot хадгалах/унших | Зөөвөрлөгддөг, өөр машинд resume хийгддэг бинар формат |
| **PyQt6** | Десктоп GUI (цонх, панел, диалог) | Боловсорсон, кросс-платформ UI |
| **pyqtgraph + PyOpenGL** | Амьд 3D viewport (`GLScatterPlotItem`) | Олон мянган цэгийг GPU-аар хурдан зурна |
| **Blender (4.2+)** | Эцсийн рэндэр (Cycles, эзлэхүүн, glow) | Кино чанарын гэрэл/материал; extension-аар `.h5`-г шууд уншина |
| **astropy, pynvml, psutil, pyyaml, matplotlib** | Нэгж/тогтмол, GPU илрүүлэлт, түвшин тохиргоо, диагностик зураг | Туслах |

**Орчин:** Python 3.10 `.venv`; RTX 3070 (CUDA) дээр хөгжүүлсэн, гэхдээ нягтрал
config-оор тохируулагддаг — нэг машинд түгжээгүй. `run.bat` venv-ийг анхны
ажиллуулалтад автоматаар үүсгэнэ.

---

## 3. Архитектур — модулиуд давхаргаар

```
┌── gui/  (PyQt6 — зөвхөн харагдац, физикт хүрэхгүй)
│     main.py          цонх + панел + QTimer-ийн амьд гогцоо
│     viewport.py      3D scatter + 3D brush-ийн хулганы заалт
│     timeline.py      видео-editor маягийн доод зурвас (scrub/play)
│     scene_builder.py объект-композицийн диалог (галактик нэм, мөргөлдүүл)
│     physics_dialog.py engine-ийн физик параметрийн панел (gravity/cooling/SF)
│     brush.py         projection математик (цэг сонгох/нэмэх)
│     blender_launcher.py  Blender-ийг олж GUI/headless горимоор нээх
│
├── core/  (GUI-гүй "тархи" — бүгд headless тестлэгддэг)
│     sim_controller.py  ГАНЦ удирдагч: build/run/step/snapshot/resume/edit
│     scenarios.py       бүртгэл: нэр → (IC, engine, dt, softening, params)
│     scene_spec.py      зөөвөрлөгддөг "scene тодорхойлолт" (объект жагсаалт+тохиргоо)
│     objects.py         object template + композиц (олон объект нэгтгэх)
│     presets.py         бэлэн тохиргоонууд (сайхан харагдах эхлэл)
│     state.py           бүх төлөв + HDF5 хадгалалт (pos/vel/mass/ptype/u/rho/age…)
│     units.py           код нэгж (kpc, km/s, 1e10 Msun) ба G
│     engine.py          цэвэр-таталцлын N-body engine (direct эсвэл BH)
│     living_galaxy_engine.py  хий+од+DM+SF+SN+cooling (SPH + таталцал)
│     gas_disk_engine.py SPH хийн диск (аналитик потенциалд)
│     sim_display.py     State → өнгөт цэгүүд (од=blackbody, хий=нягт, DM=бүдэг)
│     solvers/  gravity.py (direct N²), barnes_hut.py (treecode),
│               sph.py (SPH spline), neighbor_grid.py (uniform grid),
│               test_particle.py (хөнгөн preview)
│     integrators/ leapfrog.py (KDK интегратор)
│     ic/        disk, plummer, merger, cosmo, gas_disk, impact, full_galaxy
│
├── export/  (snapshot → рэндэрийн актив)
│     to_pointcloud.py  одод → өнгөт PLY
│     to_blender.py     төрөл бүрээр PLY + атрибут (temp/density/age)
│     blackbody.py      температур [K] → sRGB өнгө
│
├── blender_addon/  (Blender 4.2+ extension)
│     loader.py         .h5-г шууд уншиж анимэйшн болгон ачаална
│     __init__.py       N-sidebar панел + Import оператор
│     blender_manifest.toml, fetch_wheels.py, README.md
│
└── cli.py, run.bat/ps1, verify_*.py, docs/
```

**Гол зарчим:** `gui/` нь зөвхөн харагдац; бүх логик `SimController`-т төвлөрсөн тул
GUI-гүйгээр (`verify_*.py`) бүрэн тестлэгддэг.

---

## 4. Физик ба тоон аргууд (техникүүд)

### 4.1 Нэгжийн систем
Бүх зүйл **код нэгжээр**: урт = kpc, хурд = km/s, масс = 10¹⁰ M☉. Ийм учраас
snapshot физик утгатай, аль ч машин/нягтралд адил.

### 4.2 N-body таталцал
- **Direct N²** (`solvers/gravity.py`): бөөм бүр бусад бүгдтэй харилцана. Яг нарийн
  боловч O(N²). RTX 3070 дээр ~хэдэн арван мянган бөөмд тохиромжтой.
- **Barnes-Hut treecode** (`solvers/barnes_hut.py`): O(N log N).
  - **Morton (Z-order) код** — бөөмсийг орон зайн дагуу эрэмбэлнэ (numpy).
  - **LBVH (Karras radix tree)** — GPU дээр binary tree барина.
  - **Bottom-up COM** — node бүрийн масс/массын төв/хайрцгийг **race-гүй atomic
    хуримтлалаар** тооцно (энэ session-д stale-read race-ийг засаж, детерминист
    болгосон).
  - **Stack traversal** — алс/том node-ийг ганц массын төвөөр ойролцоолно
    (`size² < θ²·r²`). θ бага = нарийн, удаан.
  - Хэмжсэн: θ=0.5-д хүчний алдаа ~0.5%, энерги хадгалалт ~4e-5; хурд ×3.8 (20k),
    ×7.5 (50k), ×13.7 (100k). Galaxy engine дотор ×5.7 (30k), ×9.7 (60k).
- **Softening** (ε): ойрын бөөмсийн хуурамч хязгааргүй хүчийг зөөлрүүлнэ
  (`1/(r²+ε²)^1.5`).
- **Интегратор:** leapfrog KDK (kick-drift-kick) — симплектик, урт хугацааны энерги
  хадгалалт сайн (`integrators/leapfrog.py`).

### 4.3 SPH гидродинамик (хий)
- **Smoothed-Particle Hydrodynamics**: хийг бөөмсээр, cubic-spline kernel-аар
  нягтрал, даралт, шок, artificial viscosity тооцно (`solvers/sph.py`).
- **Uniform-grid neighbour search** (`solvers/neighbor_grid.py`): SPH-ийн O(N²)
  хөршийн хайлтыг **сетка** (cell = хайлтын радиус) болгож, 3×3×3 stencil-ээр O(N).
  Host дээр counting-sort, GPU дээр traversal. N² замтай machine precision хүртэл
  таарсан.
- **Variable smoothing length h**: нягтралд тааруулж дасан зохицдог.

### 4.4 "Амьд" галактикийн процессууд (`living_galaxy_engine.py`)
- **Од үүсэлт**: нягт хий магадлалаар одод болж хувирна (`sf_prob`).
- **Одны нас → өнгө**: залуу = хөх/халуун, хөгшин = улаан/хүйтэн.
- **Supernova feedback**: од төрсний дараа эргэн тойрны хийг халаана (`du_sn`) →
  хийн усан оргилуур, өөрийгөө зохицуулсан од үүсэлт.
- **Radiative cooling**: хий `t_cool` хугацаанд `u_floor` хүртэл хөрнө — диск нимгэн,
  спираль үүсэх нөхцөл (энэ session-д нэмэгдсэн).

### 4.5 Бүрэлдэхүүн ба домэйн
- **Бүрэн галактик** (`galaxy` scenario): од + **live DM халуун (бодит бөөмс)** + хий
  бүгд хамт self-gravity-тэй. `LivingGalaxyEngine`-д DM = species `2`; бүх
  SPH/SF/SN kernel түүнийг алгасч, зөвхөн таталцалд оролцуулна.
- **Тоос (dust):** тусдаа бөөм биш — рэндэрийн үед хийн нягтаас гаргадаг атрибут.
- **Домэйн тусгаарлалт (editor):** Galaxy mode (макро) ба Planetary mode (гараг)
  — нэг scene-д хольдоггүй.

### 4.6 DM-ийн илэрхийлэл (scenario бүрээр)
| Scenario | DM / halo |
|----------|-----------|
| disk, merger, cosmo, plummer, **galaxy** | **Бодит бөөмс** (live) |
| gas, living | **Аналитик потенциал** (бодит DM бөөмгүй) |
| impact | Halo байхгүй |

---

## 5. Өгөгдлийн урсгал

```
Scenario/Preset эсвэл Scene builder
        │  (SceneSpec: объект жагсаалт + engine тохиргоо)
        ▼
build_scenario / build_scene ──► Engine (GPU талбарууд дүүрнэ)
        │  engine.step(dt) × N
        ▼
engine.to_state() ──► State (pos/vel/mass/ptype/u/rho/age…)
        ├─► sim_display.state_to_display ──► Viewport (амьд 3D)
        └─► State.save() ──► snap_0000.h5, snap_0001.h5, …  (meta-д SceneSpec JSON)
                                   │
                ┌──────────────────┼─────────────────────┐
                ▼                  ▼                     ▼
        Timeline playback    open_folder→resume     export → PLY / blender_addon
                                                          ▼
                                                    Blender (Cycles рэндэр)
```

**Resume:** snapshot-ийн `meta["scene"]`-д бүтэн SceneSpec (объект, dt, softening,
gravity_mode, theta) хадгалагддаг тул аль ч машинд, аль ч цагт сүүлийн фрэймээс
үргэлжилнэ. Хуучин файлд `meta["scenario"]` руу fallback.

---

## 6. Юуг ЧАДНА (одоогийн чадвар)

- ✅ Диск галактик, **2 галактикийн мөргөлдөөн** (tidal tail/bridge), орчлонгийн
  **cosmic web**, Plummer тэнцвэр, **хийн диск (SPH)**, **амьд галактик** (од үүсэлт+SN),
  **гараг мөргөлдөх (giant impact)**, мөн **бүрэн галактик** (од+live DM+хий).
- ✅ **Barnes-Hut** таталцал — олон зуун мянган бөөмд масштаблана (cosmo/merger/galaxy).
- ✅ **Object-композиц editor** — слайдераар галактик/гараг нэмж, байршил/хурд/spin
  өгч мөргөлдүүлэх; **3D brush**-аар бөөм нэмэх/арилгах.
- ✅ **Snapshot + resume** (зөөвөрлөгддөг), **timeline playback**, **progress bar**,
  threaded рэндэр.
- ✅ **Blender extension** — бүх фрэймийг анимэйшн болгож timeline дээр тоглуулах.
- ✅ **Cooling, presets, physics dialog** — сайхан үр дүнгийн бэлэн тохиргоо.
- ✅ Бүрэн **headless тестлэгддэг** (`verify_*.py`), CUDA дээр баталгаажсан.

---

## 7. Юуг ЧАДАХГҮЙ (одоогийн хязгаарлалт)

- ❌ **Сая+ бөөм бодит цагт биш**: tree-ийн bbox/Morton numpy дээр, sort host дээр,
  SPH grid bin host дээр — host↔GPU хуулалт сая бөөмд саад. (GPU build хэрэгтэй.)
- ❌ **Холимог домэйн биш**: stellar галактик + SPH гараг нэг scene-д ажиллахгүй
  (нэгдсэн solver хэрэгтэй).
- ❌ **GasDiskEngine-д BH алга** (`gas`, `impact`) — тэдгээр SPH self-gravity нь N².
- ❌ **Тогтмол timestep** — бүх бөөмд нэг dt; нягт бүсэд (мөргөлдөөний цөм) илүү
  жижиг алхам автоматаар авдаггүй.
- ❌ **Энгийн cooling** — жинхэнэ химийн/металличностийн cooling хүснэгт биш,
  нэг хугацааны масштабтай хялбар загвар.
- ❌ **Solver-ийг ажил дунд сольж болохгүй** — direct↔bh солих бүрд Build дахин.
- ❌ **Периодик хүрээ (cosmology) хязгаарлагдмал** — Zel'dovich хайрцаг бий ч жинхэнэ
  периодик таталцал (Ewald/PM) алга; изоляц домэйнд тохирсон.
- ❌ **Тоос/нар орчмын эмиссийн шугам бодит RT биш** — нягтад суурилсан glow.
- ❌ **Multi-GPU / RAM-аас том өгөгдөл** биш — нэг GPU-ийн санах ойд багтах ёстой.
- ⚠️ **Бодит интерактив GUI турших** (scene-builder/brush/timeline-ийг хулганаар)
  хараахан гар аар хийгдээгүй (логик нь тестлэгдсэн).
- ⚠️ **Blender extension** жинхэнэ Blender дотор турших шаардлагатай (код бэлэн).

---

## 8. Ирээдүйн хувьсал — шинэчлэлт бүрийн trade-off

> Формат: **Хийвэл → олж авна** / **гэхдээ → хохирно/эрсдэл**.

1. **GPU дээр tree build + SPH bin** (Morton/sort/bin-ийг GPU atomic histogram +
   prefix-sum болгох)
   - ✅ Сая–арван сая бөөм; host↔GPU саад арилна; жинхэнэ том масштаб.
   - ❌ Хэрэгжүүлэхэд төвөгтэй (GPU sort/scan), дибаг хүнд; код нэмэгдэнэ.

2. **Tree-д quadrupole момент** (одоо зөвхөн monopole=COM)
   - ✅ Ижил θ-д хүч илүү нарийн → θ-г томруулж улам хурдан.
   - ❌ Node бүрд 6 нэмэлт тоо (санах ой), traversal-ийн тооцоол нэмэгдэнэ.

3. **Adaptive / per-particle timestep** (block timestep)
   - ✅ Нягт бүс (импакт цөм, галактикийн төв) нарийвчлал↑, нийт хурд↑.
   - ❌ Snapshot-ийн детерминизм/энгийн байдал алдагдана; хэрэгжүүлэлт нарийсна.

4. **BH-г GasDiskEngine-д холбох**
   - ✅ `gas`/`impact` ч том N-д масштаблана.
   - ❌ SPH+BH-ийн нэгдэл нэмэлт ажил; одоо эдгээр голдуу жижиг тул ач холбогдол бага.

5. **Particle-Mesh / TreePM нэмэх (cosmology)**
   - ✅ Жинхэнэ периодик орчлонгийн симуляц (FFT, O(N log N), маш том N).
   - ❌ Изоляц галактикт эвгүй (zero-padding); FFT/Green's function нэмэлт нарийн.

6. **Жинхэнэ cooling/heating (хүснэгт, металличность, UV background)**
   - ✅ Физик илүү бодит — олон фазын ISM, бодит од үүсэлт.
   - ❌ Хурд↓ (хайлт/интерполяц), өгөгдлийн хүснэгт хэрэгтэй, тохируулга нарийсна.

7. **Холимог домэйн / нэгдсэн solver** (нэг scene-д collisionless + SPH чөлөөтэй)
   - ✅ "Хийтэй галактик + гараг" гэх мэт дурын найрлага.
   - ❌ Engine-ийн нэгдсэн дахин дизайн; цаг алхам/softening-ийн зөрчил шийдэх.

8. **Solver hot-switch (rebuild-гүй)**
   - ✅ UX: ажил дунд direct↔bh сэлгэх.
   - ❌ `_bh`-г динамикаар хуваарилах/чөлөөлөх setter; алдааны эрсдэл бага зэрэг.

9. **Single precision (f32) сонголт**
   - ✅ Санах ой 2 дахин бага, GPU дээр 2× хурдан.
   - ❌ Урт N-body-д энерги хадгалалт муудна (одоо f64 зориуд сонгосон).

10. **Илүү сайн рэндэр** (бодит тоосны RT, эмиссийн шугам, HII муж)
    - ✅ Фото-реалистик чанар↑.
    - ❌ Рэндэрийн хугацаа↑; Blender материал/нягтралын нэмэлт ажил.

11. **Multi-GPU / out-of-core**
    - ✅ GPU санах ойноос том симуляц.
    - ❌ Домэйн задаргаа, харилцаа — маш том инженерийн ажил.

12. **Бодит интерактив туршилт + Blender extension баталгаажуулалт**
    - ✅ Сүүлчийн GUI/рэндэрийн цоорхой хаагдана (бодит хэрэглээ батлагдана).
    - ❌ Хүний/Blender орчин шаардана; автоматжуулахад хязгаартай.

---

## 9. Гүйцэтгэлийн зураглал (RTX 3070, ойролцоо)

| Бөөмийн тоо | Direct N² (gravity) | Barnes-Hut |
|-------------|---------------------|------------|
| 20,000 | ~55 ms/алхам | ~14 ms (×3.8) |
| 50,000 | ~306 ms | ~41 ms (×7.5) |
| 100,000 | ~1,170 ms | ~85 ms (×13.7) |
| 1,000,000 | бараг боломжгүй (~100 с) | ~1 с зэрэгцээ (GPU build хийвэл) |

SPH (хий) одоогоор сетка-аар O(N); хийн хувь нийт N-ийн багахан хэсэг тул галактик
дотор гол масштабын хүчин зүйл нь таталцал (BH).

---

## 10. Хаанаас эхлэх (хөгжүүлэгчид)

- Эхлүүлэх: `run.bat` (эсвэл `.venv/Scripts/python.exe -m gui.main`).
- Headless шалгах: `verify_phase1.py`, `verify_gui_g1.py`, `verify_bh_galaxy.py`,
  `verify_grid_sph.py`.
- Шинэ Taichi-kernel файлд **`from __future__ import annotations` бичиж болохгүй**
  (annotation-ийг string болгож Taichi-г эвддэг).
- Гол өргөтгөх цэгүүд: шинэ IC → `core/ic/` + `scenarios.py`; шинэ engine →
  `to_state()/step(dt)` интерфэйс; шинэ solver → `core/solvers/`.
