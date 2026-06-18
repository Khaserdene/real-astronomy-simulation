"""Scenario registry: one place that maps a scenario name to a ready engine.

Every scenario -- gravity-only (disk, merger, cosmo, plummer) or SPH-based (gas
disk, living galaxy, giant impact) -- is built through :func:`build_scenario`,
which returns an engine already loaded with initial conditions plus a sensible
timestep.  All returned engines satisfy the same small interface used by the
editor/GUI:

    engine.step(dt)         advance one step
    engine.to_state()       -> core.state.State  (for display / snapshots)
    engine.n, engine.time, engine.step_count

This unifies the previously separate engine entry points (the gravity ``Engine``
and the SPH ``GasDiskEngine`` / ``LivingGalaxyEngine``) behind one call, so the
GUI can run any scenario the same way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Scenario:
    name: str
    label: str
    kind: str                 # "gravity" | "gas" | "living" | "impact"
    dt: float
    softening: float
    description: str = ""
    params: dict = field(default_factory=dict)   # GUI-tunable defaults


# Registry. ``params`` are the handful of knobs worth exposing in the GUI.
SCENARIOS: dict[str, Scenario] = {
    "disk": Scenario(
        "disk", "Disk galaxy", "gravity", dt=1e-4, softening=0.1,
        description="A rotating stellar disk in a dark-matter halo.",
        params={}),
    "merger": Scenario(
        "merger", "Galaxy merger", "gravity", dt=1e-4, softening=0.15,
        description="Two disk galaxies collide: tidal tails and bridges.",
        params={"separation": 50.0, "impact_param": 16.0,
                "v_approach": 130.0, "inclination": 60.0}),
    "cosmo": Scenario(
        "cosmo", "Cosmic web", "gravity", dt=5e-5, softening=0.22,
        description="A perturbed box collapses into filaments and clumps.",
        params={"hubble": 42.0, "amplitude": 1.4}),
    "plummer": Scenario(
        "plummer", "Plummer sphere", "gravity", dt=3e-5, softening=0.05,
        description="Equilibrium test sphere (energy/virial check).",
        params={"total_mass": 10.0, "scale_radius": 1.0}),
    "gas": Scenario(
        "gas", "Gas disk (SPH)", "gas", dt=5e-4, softening=0.2,
        description="An SPH gas disk in a galaxy potential.",
        params={"sound_speed": 12.0}),
    "living": Scenario(
        "living", "Living galaxy", "living", dt=5e-4, softening=0.2,
        description="Gas forms stars; stars age and explode (SN feedback).",
        params={"sf_prob": 0.03, "du_sn": 500.0}),
    "impact": Scenario(
        "impact", "Giant impact", "impact", dt=5e-4, softening=0.1,
        description="Two self-gravitating bodies collide (shock heating).",
        params={"v_approach": 30.0, "impact_param": 1.6}),
}

# Zero-potential params reused by the giant-impact SPH engine.
_NO_POTENTIAL = dict(M_d=0.0, a=1.0, b=1.0, M_h=0.0, a_h=1.0)


def build_scenario(name: str, n: int = 20000, seed: int = 0,
                   overrides: dict | None = None):
    """Build a scenario's engine (ICs loaded) and return (engine, dt)."""
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario '{name}'. Have: {list(SCENARIOS)}")
    sc = SCENARIOS[name]
    p = dict(sc.params)
    if overrides:
        p.update({k: v for k, v in overrides.items() if v is not None})

    if sc.kind == "gravity":
        from core.scene import Scene, build_state
        from core.engine import Engine
        state = build_state(Scene(ic=name, seed=seed, params=p), n, "direct")
        eng = Engine(softening=sc.softening)
        eng.load_state(state)
        return eng, sc.dt

    if sc.kind == "gas":
        from core.ic.gas_disk import make_gas_disk
        from core.gas_disk_engine import GasDiskEngine
        pos, vel, mass, u = make_gas_disk(
            n=n, seed=seed, sound_speed=p.get("sound_speed", 12.0))
        eng = GasDiskEngine(softening=sc.softening)
        eng.setup(pos, vel, mass, u)
        return eng, sc.dt

    if sc.kind == "living":
        from core.ic.gas_disk import make_gas_disk
        from core.living_galaxy_engine import LivingGalaxyEngine
        pos, vel, mass, u = make_gas_disk(n=n, seed=seed)
        eng = LivingGalaxyEngine(softening=sc.softening,
                                 sf_prob=p.get("sf_prob", 0.03),
                                 du_sn=p.get("du_sn", 500.0))
        eng.setup(pos, vel, mass, u)
        return eng, sc.dt

    if sc.kind == "impact":
        from core.ic.impact import make_impact
        from core.gas_disk_engine import GasDiskEngine
        pos, vel, mass, u = make_impact(
            n_each=max(n // 2, 1), seed=seed,
            v_approach=p.get("v_approach", 30.0),
            impact_param=p.get("impact_param", 1.6))
        eng = GasDiskEngine(pot=dict(_NO_POTENTIAL), softening=sc.softening)
        eng.setup(pos, vel, mass, u)
        return eng, sc.dt

    raise ValueError(f"unhandled scenario kind: {sc.kind}")


def resume_scenario(name: str, state):
    """Rebuild a scenario's engine from a saved State and return (engine, dt).

    Gravity and gas/impact resume exactly; living-galaxy resume restores the
    star flags and ages but re-derives SPH state (a close, not bit-exact, resume).
    """
    sc = SCENARIOS.get(name)
    if sc is None or sc.kind == "gravity":
        from core.engine import Engine
        eng = Engine(softening=(sc.softening if sc else 0.1))
        eng.load_state(state)
        return eng, (sc.dt if sc else 1e-4)

    if sc.kind in ("gas", "impact"):
        from core.gas_disk_engine import GasDiskEngine
        pot = dict(_NO_POTENTIAL) if sc.kind == "impact" else None
        eng = GasDiskEngine(pot=pot, softening=sc.softening)
        eng.setup(state.pos, state.vel, state.mass,
                  state.u if state.u is not None else _default_u(state.n))
        eng.time, eng.step_count = state.time, state.step
        return eng, sc.dt

    if sc.kind == "living":
        import numpy as np
        from core.living_galaxy_engine import LivingGalaxyEngine
        from core.state import PTYPE_STAR
        eng = LivingGalaxyEngine(softening=sc.softening)
        eng.setup(state.pos, state.vel, state.mass,
                  state.u if state.u is not None else _default_u(state.n))
        eng.time, eng.step_count = state.time, state.step
        # Restore which particles are stars and their birth times.
        is_star = (state.ptype == PTYPE_STAR).astype(np.int32)
        eng._f["is_star"].from_numpy(is_star)
        if state.age is not None:
            birth = np.where(state.age >= 0.0, state.time - state.age, 0.0)
            eng._f["birth"].from_numpy(birth.astype(np.float64))
            eng._f["sn_done"].from_numpy(is_star)  # assume past SN already fired
        return eng, sc.dt

    raise ValueError(f"cannot resume scenario kind: {sc.kind}")


def _default_u(n):
    import numpy as np
    return np.full(n, 200.0)
