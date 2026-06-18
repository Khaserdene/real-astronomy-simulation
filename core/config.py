"""Level presets, hardware detection, and resolution scaling.

This module decouples *what* you simulate (a Scene) from *how finely* you
simulate it (a level).  It also makes the simulator hardware-aware without being
hardware-*locked*: it detects available VRAM/RAM, suggests a sensible level, and
caps or warns on particle counts that won't fit -- but every choice is
overridable.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional

import yaml

_LEVELS_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "levels.yaml")


@dataclass
class LevelConfig:
    """Resolved numerical settings for one run."""
    name: str
    solver: str          # "test_particle" | "direct" | "tree"
    n: int               # total particle count
    dt: float
    softening: float
    note: str = ""


@dataclass
class Hardware:
    gpu_name: str
    vram_mb: int
    ram_gb: float
    has_cuda: bool


# --------------------------------------------------------------------- presets
def load_levels(path: str = _LEVELS_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_level(name: str, overrides: Optional[dict] = None,
              path: str = _LEVELS_PATH) -> LevelConfig:
    """Resolve a named level into a LevelConfig, applying optional overrides.

    ``overrides`` may set any of ``n``, ``dt``, ``softening``, ``solver`` --
    e.g. to push particle count beyond a preset on a bigger machine.
    """
    data = load_levels(path)
    if name not in data["levels"]:
        raise KeyError(f"unknown level '{name}'. "
                       f"Available: {list(data['levels'])}")
    spec = dict(data["levels"][name])
    if overrides:
        spec.update({k: v for k, v in overrides.items() if v is not None})
    return LevelConfig(
        name=name,
        solver=spec["solver"],
        n=int(spec["n"]),
        dt=float(spec["dt"]),
        softening=float(spec["softening"]),
        note=spec.get("note", "").strip(),
    )


# ------------------------------------------------------------------- hardware
def detect_hardware() -> Hardware:
    """Best-effort detection of GPU VRAM and system RAM (never raises)."""
    gpu_name, vram_mb, has_cuda = "unknown", 0, False
    try:
        import pynvml
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(h)
            gpu_name = name.decode() if isinstance(name, bytes) else name
            vram_mb = int(pynvml.nvmlDeviceGetMemoryInfo(h).total / 1024 ** 2)
            has_cuda = True
            pynvml.nvmlShutdown()
    except Exception:
        pass

    ram_gb = 0.0
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:
        pass

    return Hardware(gpu_name, vram_mb, ram_gb, has_cuda)


def max_particles(hw: Hardware, path: str = _LEVELS_PATH) -> int:
    """How many particles fit in the budgeted fraction of VRAM."""
    data = load_levels(path)
    if hw.vram_mb <= 0:
        return 0
    budget = hw.vram_mb * 1024 ** 2 * data["vram_budget_fraction"]
    return int(budget / data["bytes_per_particle"])


def suggest_level(hw: Hardware, path: str = _LEVELS_PATH) -> str:
    """Suggest the highest *practical* self-gravitating level for this GPU.

    Bounded by both the VRAM budget and a compute-based ceiling: direct N^2 cost
    grows as N^2, so for the ``direct`` solver the realistic limit is compute,
    not memory.  Users can always override to push past the suggestion.
    """
    data = load_levels(path)
    mem_cap = max_particles(hw, path)
    comfortable = int(data.get("direct_comfortable_n", 150000))
    order = ["ultra", "high", "medium", "low", "preview"]
    for name in order:
        if name not in data["levels"]:
            continue
        spec = data["levels"][name]
        cap = mem_cap if spec["solver"] != "direct" else min(mem_cap, comfortable)
        if spec["n"] <= cap:
            return name
    return "low"  # fall back if even low doesn't "fit" the estimate


def check_fit(level: LevelConfig, hw: Hardware, path: str = _LEVELS_PATH) -> None:
    """Emit a warning if a level's particle count likely exceeds VRAM."""
    cap = max_particles(hw, path)
    if cap and level.n > cap:
        warnings.warn(
            f"Level '{level.name}' wants {level.n:,} particles but the detected "
            f"GPU ({hw.gpu_name}, {hw.vram_mb} MiB) comfortably fits ~{cap:,}. "
            f"Reduce --n or expect out-of-memory / swapping.",
            stacklevel=2,
        )
