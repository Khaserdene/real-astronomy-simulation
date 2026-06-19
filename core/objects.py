"""Object templates + scene composition for the interactive editor.

The editor builds a scene by *placing objects*: drop a galaxy, set its position
and approach velocity (and optional spin), drop another, and collide them -- or
two planets, in the planetary domain.  Each :class:`ObjectTemplate` knows how to
generate one object's particles at the origin, at rest, with sensible default
parameters; :func:`compose_arrays` then offsets each object to its position, adds
its bulk velocity and spin, and concatenates everything into one particle set.

This is the multi-object generalisation that :class:`core.scene_spec.SceneSpec`
was designed for (today's single-scenario runs are just the one-object case).

Composition is *within a domain and engine kind* (v1): all placed objects must
share one engine kind (e.g. two gravity galaxies, or two SPH planets).  Mixing a
collisionless galaxy with an SPH body needs a unified solver and is deferred.

All quantities are code units (kpc, km/s, 1e10 Msun); ``u`` is (km/s)^2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.state import PTYPE_GAS


@dataclass
class ObjectTemplate:
    name: str
    label: str
    domain: str        # "galaxy" | "planetary"
    kind: str          # engine kind: "gravity" | "gas" | "impact"
    dt: float
    softening: float
    defaults: dict = field(default_factory=dict)
    description: str = ""


OBJECTS: dict[str, ObjectTemplate] = {
    "disk_galaxy": ObjectTemplate(
        "disk_galaxy", "Disk galaxy", "galaxy", "gravity",
        dt=1e-4, softening=0.1,
        defaults=dict(disk_mass=5.0, disk_scale=3.0, disk_height=0.25,
                      halo_mass=12.0, halo_scale=8.0, disk_fraction=0.7,
                      sigma_frac=0.1),
        description="Stellar exponential disk in a live Plummer DM halo."),
    "plummer_blob": ObjectTemplate(
        "plummer_blob", "Plummer blob", "galaxy", "gravity",
        dt=3e-5, softening=0.05,
        defaults=dict(total_mass=10.0, scale_radius=1.0),
        description="Equilibrium spherical blob of collisionless particles."),
    "gas_disk": ObjectTemplate(
        "gas_disk", "Gas disk", "galaxy", "gas",
        dt=5e-4, softening=0.2,
        defaults=dict(gas_mass=1.0, disk_scale=3.5, disk_height=0.2,
                      sound_speed=12.0),
        description="Rotating SPH gas disk."),
    "planet": ObjectTemplate(
        "planet", "Planet / body", "planetary", "impact",
        dt=5e-4, softening=0.1,
        defaults=dict(body_mass=0.02, body_radius=1.2, cs_frac=0.7),
        description="Self-gravitating SPH sphere (planetary body)."),
}


def templates_for(domain: str) -> dict[str, ObjectTemplate]:
    return {k: v for k, v in OBJECTS.items() if v.domain == domain}


def compose_spec(object_specs, domain: str = "galaxy"):
    """Wrap a list of placed :class:`~core.scene_spec.ObjectSpec` into a
    :class:`~core.scene_spec.SceneSpec`, deriving the shared engine kind plus a
    safe dt (smallest) and softening (largest) from the objects' templates.
    """
    from core.scene_spec import SceneSpec

    if not object_specs:
        raise ValueError("a scene needs at least one object")
    used = [OBJECTS[o.template] for o in object_specs]
    kinds = {t.kind for t in used}
    if len(kinds) > 1:
        raise ValueError(
            f"cannot compose mixed engine kinds {sorted(kinds)} yet (v1).")
    return SceneSpec(engine_kind=kinds.pop(),
                     dt=min(t.dt for t in used),
                     softening=max(t.softening for t in used),
                     objects=list(object_specs), domain=domain)


# --------------------------------------------------------------- single object
def build_object(template: str, n: int, seed: int, params: dict | None = None):
    """Build one object's particles at the origin, at rest.

    Returns a dict with ``pos, vel, mass, ptype`` and (for SPH kinds) ``u``.
    """
    if template not in OBJECTS:
        raise KeyError(f"unknown object '{template}'. Have: {list(OBJECTS)}")
    tmpl = OBJECTS[template]
    p = dict(tmpl.defaults)
    if params:
        p.update({k: v for k, v in params.items() if v is not None})

    if template == "disk_galaxy":
        from core.ic.disk import make_disk_galaxy
        n_disk = max(int(n * p["disk_fraction"]), 1)
        n_halo = max(n - n_disk, 1)
        st = make_disk_galaxy(n_disk=n_disk, n_halo=n_halo, seed=seed,
                              disk_mass=p["disk_mass"], disk_scale=p["disk_scale"],
                              disk_height=p["disk_height"],
                              halo_mass=p["halo_mass"], halo_scale=p["halo_scale"],
                              sigma_frac=p.get("sigma_frac", 0.1))
        return dict(pos=st.pos, vel=st.vel, mass=st.mass, ptype=st.ptype, u=None)

    if template == "plummer_blob":
        from core.ic.plummer import make_plummer
        st = make_plummer(n=n, total_mass=p["total_mass"],
                          scale_radius=p["scale_radius"], seed=seed)
        return dict(pos=st.pos, vel=st.vel, mass=st.mass, ptype=st.ptype, u=None)

    if template == "gas_disk":
        from core.ic.gas_disk import make_gas_disk
        pos, vel, mass, u = make_gas_disk(
            n=n, gas_mass=p["gas_mass"], disk_scale=p["disk_scale"],
            disk_height=p["disk_height"], sound_speed=p["sound_speed"], seed=seed)
        ptype = np.full(n, PTYPE_GAS, dtype=np.int32)
        return dict(pos=pos, vel=vel, mass=mass, ptype=ptype, u=u)

    if template == "planet":
        from core.ic.impact import make_body
        pos, vel, mass, u = make_body(
            n=n, body_mass=p["body_mass"], body_radius=p["body_radius"],
            cs_frac=p["cs_frac"], seed=seed)
        ptype = np.full(n, PTYPE_GAS, dtype=np.int32)
        return dict(pos=pos, vel=vel, mass=mass, ptype=ptype, u=u)

    raise ValueError(f"unhandled object template: {template}")


# ------------------------------------------------------------- scene composition
def compose_arrays(spec) -> dict:
    """Build + place every object in ``spec`` and concatenate them.

    Pure numpy (no Taichi): returns ``pos, vel, mass, ptype, u, kind``.  ``u`` is
    ``None`` for gravity-only scenes.  Raises if objects mix engine kinds (v1).
    """
    if not spec.objects:
        raise ValueError("scene has no objects")
    kinds = {OBJECTS[o.template].kind for o in spec.objects}
    if len(kinds) > 1:
        raise ValueError(
            f"cannot compose objects of mixed engine kinds {sorted(kinds)} yet; "
            "all objects in a scene must share one kind (v1).")
    kind = kinds.pop()
    needs_u = kind in ("gas", "impact")

    pos_l, vel_l, mass_l, ptype_l, u_l = [], [], [], [], []
    for obj in spec.objects:
        d = build_object(obj.template, obj.n, obj.seed, obj.params)
        pos = np.asarray(d["pos"], dtype=np.float64)
        vel = np.asarray(d["vel"], dtype=np.float64)

        # spin: add solid-body rotation about z (omega in (km/s)/kpc).
        if obj.spin:
            vel = vel + obj.spin * np.column_stack(
                (-pos[:, 1], pos[:, 0], np.zeros(len(pos))))
        # place: position offset + bulk drift velocity.
        pos = pos + np.asarray(obj.position, dtype=np.float64)
        vel = vel + np.asarray(obj.velocity, dtype=np.float64)

        pos_l.append(pos)
        vel_l.append(vel)
        mass_l.append(np.asarray(d["mass"], dtype=np.float64))
        ptype_l.append(np.asarray(d["ptype"], dtype=np.int32))
        if needs_u:
            u = d["u"]
            u_l.append(np.asarray(u if u is not None else
                                  np.full(len(pos), 200.0), dtype=np.float64))

    return dict(
        pos=np.vstack(pos_l), vel=np.vstack(vel_l),
        mass=np.concatenate(mass_l), ptype=np.concatenate(ptype_l),
        u=(np.concatenate(u_l) if needs_u else None), kind=kind)


def build_scene(spec):
    """Compose ``spec`` and return a ready engine + dt (constructs Taichi)."""
    a = compose_arrays(spec)
    kind, dt, soft = a["kind"], spec.dt, spec.softening

    if kind == "gravity":
        import numpy as np
        from core.engine import Engine
        from core.state import State
        ids = np.arange(len(a["pos"]), dtype=np.int64)
        state = State(pos=a["pos"], vel=a["vel"], mass=a["mass"],
                      ptype=a["ptype"], ids=ids)
        eng = Engine(softening=soft)
        eng.load_state(state)
        return eng, dt

    if kind in ("gas", "impact"):
        from core.gas_disk_engine import GasDiskEngine
        pot = (dict(M_d=0.0, a=1.0, b=1.0, M_h=0.0, a_h=1.0)
               if kind == "impact" else None)
        # Planetary impacts keep their shock heat (no cooling); gas disks cool.
        eng = GasDiskEngine(pot=pot, softening=soft, cooling=(kind != "impact"))
        eng.setup(a["pos"], a["vel"], a["mass"], a["u"])
        return eng, dt

    raise ValueError(f"unhandled scene kind: {kind}")
