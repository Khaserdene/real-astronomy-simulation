# Хэрэглэгчийн заавар — Real Astronomy Simulator

GPU дээр астрономын симуляц хийж, фрэйм бүрээр хадгалж, Blender-т рэндэрлэх систем.
Энэ заавар суулгахаас эхлээд галактик/мөргөлдөөн/орчлон симуляц хийж, рэндэрлэх
хүртэлх бүх алхмыг хамарна.

> Командуудыг **төслийн үндсэн хавтаснаас** (энэ файл байгаа хавтас) ажиллуул.
> Бүх Python команд venv доторх Python-ыг (`.venv/Scripts/python.exe`) ашиглана.

---

## 1. Шаардлага

- **Windows** + **NVIDIA GPU** (CUDA). GPU байхгүй бол командад `--arch cpu` нэм (удаан).
- **Python 3.10** (Taichi 1.7.x-д хэрэгтэй).
- **Blender 4.x** (рэндэрт; тусдаа суулгана) — жишээ зам:
  `C:/Program Files/Blender Foundation/Blender 4.5/blender.exe`

## 2. Суулгац

```bash
py -3.10 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

GPU-г олж байгааг шалга:
```bash
.venv/Scripts/python.exe cli.py --suggest
```
Гаралт: GPU нэр, VRAM, RAM, санал болгох түвшин (жишээ нь `high`).

## 3. Зөв ажиллаж буйг батлах (verify)

```bash
.venv/Scripts/python.exe verify_phase1.py   # энерги хадгалалт, диск
.venv/Scripts/python.exe verify_phase2.py   # level, checkpoint, resume
.venv/Scripts/python.exe verify_phase3.py   # editor controller
.venv/Scripts/python.exe verify_phase5.py   # SPH (Sod shock tube)
```
Бүгд `PASSED` гарвал бэлэн.

---

## 4. Симуляц хийх (CLI)

Үндсэн команд: `cli.py`. Гаралт нь `--out` хавтсанд **фрэйм бүрээр** HDF5 snapshot
(`snap_0000.h5`, ...) болж хадгалагдана. Snapshot бүр нь resume-д ашиглах checkpoint.

### Боломжтой дүр зураг (`--ic`)

| `--ic` | Юу |
|--------|-----|
| `disk` | Эргэлдэх диск галактик (анхдагч) |
| `merger` | Хоёр галактик мөргөлдөх |
| `cosmo` | Орчлонгийн бүтэц үүсэх (cosmic web) |
| `plummer` | Plummer бөмбөрцөг (тест/эталон) |

### Нарийвчлалын түвшин (`--level`)
`preview` · `low` · `medium` · `high` · `ultra` — партиклын тоо, dt, нарийвчлалыг тохируулна
(`configs/levels.yaml`). `--n`, `--dt`, `--softening`-ээр гараар дарж болно.

### Жишээнүүд

```bash
# Диск галактик (40k партикл, 2000 алхам, 20 алхам тутамд snapshot)
.venv/Scripts/python.exe cli.py --ic disk --seed 7 --level medium \
    --steps 2000 --snap-every 100 --out output/galaxy

# Галактик мөргөлдөөн
.venv/Scripts/python.exe cli.py --ic merger --seed 5 --level medium \
    --steps 1600 --snap-every 100 --out output/merger_run

# Орчлонгийн бүтэц (cosmic web)
.venv/Scripts/python.exe cli.py --ic cosmo --seed 2 --level medium \
    --n 20000 --dt 5e-5 --softening 0.22 --steps 1400 --snap-every 100 \
    --out output/cosmo_run
```

### Preview → жинхэнэ (production)
Эхлээд хурдан preview харж IC-ээ батлаад, **ижил seed-ээр** өндөр нарийвчлалаар ажиллуул:
```bash
.venv/Scripts/python.exe cli.py --ic disk --seed 7 --level preview --steps 1500 --out output/prev
.venv/Scripts/python.exe cli.py --ic disk --seed 7 --level high    --steps 4000 --out output/prod
```

### Завсарлаад үргэлжлүүлэх (checkpoint / resume)
Симуляцыг дундаас, бүр **өөр компьютер дээр** үргэлжлүүлж болно (snapshot нь зөөвөрлөгддөг):
```bash
.venv/Scripts/python.exe cli.py --resume output/galaxy/snap_0040.h5 \
    --steps 4000 --snap-every 100 --out output/galaxy
```

---

## 5. График интерфэйс (үндсэн арга) — PyQt6 апп

```bash
.venv/Scripts/python.exe -m gui.main
```

Бүх ажил нэг цонхонд — CLI шаардахгүй:
- **Scene** — дүр зураг сонгох (disk / merger / cosmic web / plummer / gas /
  living / impact), seed, партиклын тоо.
- **Run** — **Build** → **Run/Pause** → **Step**, steps/frame.
- **View** — live 3D viewport (хулганаар орбит/зум), одод/хий физик өнгөөр.
- **Output** — auto-snapshot, Save snapshot, **Load checkpoint** (file dialog).
- **Timeline** — хадгалсан snapshot-уудыг слайдераар гүйлгэж үзэх (playback).
- **Render (Blender)**:
  - **Open in Blender (interactive)** — Blender GUI нээж, **төрлөөр нь объект+
    материалтай** scene ачаална (одод emissive, хий volume, DM бүдэг). Та эндээс
    материал/гэрэлтүүлэг/камераа засаж, өөрөө рэндэрлэнэ.
  - **Quick render (headless)** — одоогийн фрэймийг автоматаар рэндэрлэж PNG нээнэ.

> Том симуляц (High 40k) live ажиллахад GUI түр удааширч магадгүй — interactive-д
> Preview/Low/Medium ашиглаж, том бодолтыг CLI-аар (§4) фонд хийгээрэй.
> (Хуучин Taichi GGUI editor: `python -m editor.app` — хөнгөн хувилбар.)

Blender-ийг олохын тулд апп нь `C:/Program Files/Blender Foundation/Blender */`-ийг
автоматаар хайна; өөр байвал `BLENDER` орчны хувьсагчид `blender.exe`-ийн замаа тавь.

---

## 6. Blender-т рэндэрлэх

Хоёр алхам: (1) snapshot → өнгөт PLY, (2) Blender headless рэндэр.

```bash
# 1) snapshots -> coloured PLY (<run>/ply/ дотор)
.venv/Scripts/python.exe -m export.to_pointcloud output/galaxy

# 2) PLY -> PNG (Cycles GPU + bloom). BL хувьсагчид Blender-ийн замаа тавь.
BL="/c/Program Files/Blender Foundation/Blender 4.5/blender.exe"
"$BL" --background --python blender/render_frames.py -- \
    --ply output/galaxy/ply --out output/galaxy/render \
    --res 1280 --samples 48 --radius 0.09 --emission 1.3 --glare 5 --cam-elev 72
```

Гол сонголтууд:
- `--cam-elev` — харах өнцөг (90 = дээрээс face-on, 0 = хажуугаас edge-on)
- `--emission` — оддын гэрэлтэлт (өнгө цайрвал багасга)
- `--radius` — цэгийн хэмжээ, `--glare` — gerel сарних (0 = унтраах)
- `--res`, `--samples` — нягт ба чанар

Гаралт нь `output/galaxy/render/snap_XXXX.png`. Эдгээрийг видео болгож нийлүүлж болно
(жишээ нь ffmpeg-ээр).

---

## 7. Хий, амьд галактик, гараг мөргөлдөөн (жишээ скриптүүд)

Эдгээр нь SPH-суурьтай тул тусдаа жишээ скриптээр ажиллана (CLI биш).

```bash
# Хийн диск -> volume рэндэр
.venv/Scripts/python.exe examples/run_gas_disk.py --steps 400
"$BL" --background --python blender/render_gas.py -- \
    --ply output/gas_demo/gas.ply --out output/gas_demo/render.png \
    --voxel 0.22 --radius 0.35 --density 0.12 --emission 0.13 --cam-dist 85 --cam-elev 32

# Амьд галактик: од үүсэх + хувьсал + supernova feedback
.venv/Scripts/python.exe examples/run_living_galaxy.py --steps 750
"$BL" --background --python blender/render_frames.py -- \
    --ply output/living_demo --out output/living_demo/render \
    --radius 0.08 --emission 1.6 --glare 5 --cam-elev 62

# Гараг мөргөлдөөн (giant impact): өнгө = дотоод энерги (халуун = гэрэлтэх)
.venv/Scripts/python.exe examples/run_impact.py --steps 520
"$BL" --background --python blender/render_frames.py -- \
    --ply output/impact_demo --out output/impact_demo/render \
    --radius 0.06 --emission 1.6 --glare 4 --cam-dist 16 --cam-elev 35
```

---

## 8. Research-grade (GADGET-4 / GIZMO)

Илүү өндөр нарийвчлал/баталгаажсан физик хэрэгтэй бол GADGET-4/GIZMO-г WSL2 дээр build
хийж, snapshot-ыг нь манай pipeline руу оруул. Дэлгэрэнгүй: [research/README.md](research/README.md).
```bash
.venv/Scripts/python.exe -m research.gadget_to_state /path/snap.hdf5 output/gadget/snap_0000.h5
# дараа нь export.to_pointcloud + render_frames (ердийн pipeline)
```

---

## 9. Нэгжийн систем

Бүх дотоод тоо **галактикийн код нэгжээр**: урт `kpc`, хурд `km/s`, масс `1e10 Msun`,
хугацаа `~0.978 Gyr`. Snapshot эдгээр физик нэгжээр хадгалагддаг тул өөр комп/түвшин дээр
ачаалахад өөрчлөгддөггүй ([core/units.py](core/units.py)).

## 10. Гарын алдаа засах

| Асуудал | Шийдэл |
|---------|--------|
| `No module named 'core'` | Командыг **төслийн үндсэн хавтаснаас** ажиллуул |
| `ti.init` алдаа / GPU олдохгүй | `--arch cpu` (удаан) эсвэл CUDA драйвер шалга |
| Рэндэр хар гарна | `--emission` ихэсгэх, `--cam-dist` тааруулах |
| Рэндэр хэт цагаан | `--emission` багасгах |
| Blender PNG буруу хавтсанд | Зам **absolute** эсэхийг шалга (скрипт автоматаар хөрвүүлдэг) |
| Санах ой хүрэхгүй | `--level`/`--n` багасга; `--suggest`-ийн саналыг дага |

Дэлгэрэнгүй техник мэдээлэл: [README.md](README.md). Зам зураг: `.claude/plans/`.
