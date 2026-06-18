"""Scene specification -- a portable, forward-compatible description of *what*
to build, separate from the resulting particle :class:`~core.state.State`.

Today a scene is a single scenario (one implicit object centred at the origin).
The schema already carries a *list* of placed objects with position / velocity /
spin so the upcoming interactive editor (object-composition: drop a galaxy, set
its position and approach velocity, drop another, collide them) can grow into it
without changing the on-disk format.

The spec is serialised to JSON and stored in a snapshot's ``meta["scene"]`` so a
run can be resumed -- or rebuilt from scratch at a different resolution -- on any
machine, with the engine configuration (kind, dt, softening) and every tuned
parameter preserved (which the bare scenario name alone would lose).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

SCENE_SCHEMA_VERSION = 1


@dataclass
class ObjectSpec:
    """One placed object in a scene.

    ``template`` is a scenario/object name (e.g. ``"disk"``).  ``position`` and
    ``velocity`` are the bulk offset / drift applied to the object's particles;
    ``spin`` is reserved for the editor (extra angular momentum about ``z``).
    Today there is exactly one object and ``position``/``velocity``/``spin`` are
    left at their no-op defaults.
    """

    template: str
    n: int = 20000
    seed: int = 0
    position: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    spin: float | None = None
    params: dict = field(default_factory=dict)


@dataclass
class SceneSpec:
    """A full scene: engine configuration plus the list of objects to build."""

    engine_kind: str                 # "gravity" | "gas" | "living" | "impact"
    dt: float
    softening: float
    objects: list[ObjectSpec] = field(default_factory=list)
    domain: str = "galaxy"           # "galaxy" | "planetary" (editor domains)
    gravity_mode: str = "direct"     # "direct" (N^2) | "bh" (Barnes-Hut)
    theta: float = 0.6               # Barnes-Hut opening angle
    version: int = SCENE_SCHEMA_VERSION

    # ------------------------------------------------------------ convenience
    @property
    def primary(self) -> ObjectSpec:
        """The first object -- the only one in single-scenario scenes today."""
        return self.objects[0]

    @classmethod
    def single(cls, scenario_name: str, n: int, seed: int,
               overrides: dict | None = None) -> "SceneSpec":
        """Build a one-object spec from a registered scenario + overrides."""
        from core.scenarios import SCENARIOS

        sc = SCENARIOS[scenario_name]
        params = dict(sc.params)
        if overrides:
            params.update({k: v for k, v in overrides.items() if v is not None})
        return cls(
            engine_kind=sc.kind, dt=sc.dt, softening=sc.softening,
            objects=[ObjectSpec(template=scenario_name, n=n, seed=seed,
                                params=params)],
        )

    # -------------------------------------------------------------------- I/O
    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw) -> "SceneSpec":
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        d = json.loads(raw)
        # Drop unknown keys defensively so newer files don't crash older code.
        obj_known = set(ObjectSpec.__dataclass_fields__)
        objs = [ObjectSpec(**{k: v for k, v in o.items() if k in obj_known})
                for o in d.get("objects", [])]
        scene_known = set(cls.__dataclass_fields__)
        d = {k: v for k, v in d.items() if k in scene_known}
        d["objects"] = objs
        return cls(**d)
