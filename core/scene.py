"""A Scene describes *what* to simulate, independent of resolution.

The same Scene (IC type, physical parameters, RNG seed) can be realised at any
level -- a 60k-tracer preview or a 400k self-gravitating production run -- and,
because the seed is fixed, both are "the same galaxy, sampled differently".  This
is the backbone of the preview -> production workflow.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.state import State, PTYPE_STAR


# Default physical parameters per IC type (code units: kpc, km/s, 1e10 Msun).
_DEFAULTS = {
    "disk": dict(disk_mass=5.0, disk_scale=3.0, disk_height=0.25,
                 halo_mass=12.0, halo_scale=8.0, disk_fraction=0.7),
    "plummer": dict(total_mass=10.0, scale_radius=1.0),
    "merger": dict(separation=50.0, impact_param=16.0, v_approach=130.0,
                   inclination=60.0, mass_ratio=1.0),
    "cosmo": dict(box_size=24.0, total_mass=120.0, amplitude=1.4,
                  n_modes=14, hubble=42.0, vel_factor=18.0),
}


@dataclass
class Scene:
    """The resolution-independent definition of an experiment."""
    ic: str = "disk"
    seed: int = 0
    params: dict = field(default_factory=dict)

    def resolved_params(self) -> dict:
        if self.ic not in _DEFAULTS:
            raise KeyError(f"unknown IC '{self.ic}'. Available: {list(_DEFAULTS)}")
        p = dict(_DEFAULTS[self.ic])
        p.update(self.params)
        return p


def build_state(scene: Scene, n: int, solver: str) -> State:
    """Realise a Scene as an initial State at particle count ``n``.

    For ``solver == "test_particle"`` only the (massless) stellar tracers are
    produced -- they move in the analytic potential defined by the same params.
    For self-gravitating solvers the full live system (e.g. disk + halo) is built.
    """
    p = scene.resolved_params()

    if scene.ic == "plummer":
        from core.ic.plummer import make_plummer
        return make_plummer(n=n, total_mass=p["total_mass"],
                            scale_radius=p["scale_radius"], seed=scene.seed)

    if scene.ic == "disk":
        from core.ic.disk import make_disk_galaxy
        if solver == "test_particle":
            # Tracers only: build the disk, keep the stars as massless points.
            full = make_disk_galaxy(n_disk=n, n_halo=1, seed=scene.seed,
                                    disk_mass=p["disk_mass"],
                                    disk_scale=p["disk_scale"],
                                    disk_height=p["disk_height"],
                                    halo_mass=p["halo_mass"],
                                    halo_scale=p["halo_scale"])
            stars = full.ptype == PTYPE_STAR
            return State(
                pos=full.pos[stars], vel=full.vel[stars],
                mass=np.zeros(int(stars.sum())),
                ptype=full.ptype[stars], ids=full.ids[stars],
                seed=scene.seed, meta={"ic": "disk_tracers"},
            )
        n_disk = max(int(n * p["disk_fraction"]), 1)
        n_halo = max(n - n_disk, 1)
        return make_disk_galaxy(n_disk=n_disk, n_halo=n_halo, seed=scene.seed,
                                disk_mass=p["disk_mass"],
                                disk_scale=p["disk_scale"],
                                disk_height=p["disk_height"],
                                halo_mass=p["halo_mass"],
                                halo_scale=p["halo_scale"])

    if scene.ic == "cosmo":
        from core.ic.cosmo import make_cosmo_box
        n_side = max(int(round(n ** (1.0 / 3.0))), 4)
        return make_cosmo_box(n_side=n_side, box_size=p["box_size"],
                              total_mass=p["total_mass"], amplitude=p["amplitude"],
                              n_modes=p["n_modes"], hubble=p["hubble"],
                              vel_factor=p["vel_factor"], seed=scene.seed)

    if scene.ic == "merger":
        from core.ic.merger import make_merger
        return make_merger(n_each=max(n // 2, 1), seed=scene.seed,
                           separation=p["separation"],
                           impact_param=p["impact_param"],
                           v_approach=p["v_approach"],
                           inclination=p["inclination"],
                           mass_ratio=p["mass_ratio"])

    raise KeyError(f"unknown IC '{scene.ic}'")
