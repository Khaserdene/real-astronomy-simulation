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
    description: str = ""      # one-line summary (scenario menu)
    details: str = ""         # multi-line: physics, what happens, params
    params: dict = field(default_factory=dict)   # GUI-tunable defaults


# Registry. ``params`` are the handful of knobs worth exposing in the GUI.
SCENARIOS: dict[str, Scenario] = {
    "empty": Scenario(
        "empty", "Empty (brush canvas)", "gravity", dt=1e-4, softening=0.1,
        description="A blank canvas — build it, then paint particles with the "
                    "3D brush (Add mode) on the work plane.",
        details=(
            "PHYSICS  Collisionless gravity over whatever you paint. Starts with a "
            "single faint seed particle; use the Brush panel (Enable → Add) to "
            "drop stars/DM on the work plane, then Run to watch them evolve.\n"
            "WHAT YOU SEE  Nothing until you paint. Great for sketching custom "
            "configurations by hand.\n"
            "PARAMS  none — set the brush type/count/radius and work-plane depth "
            "in the Brush panel."),
        params={}),
    "disk": Scenario(
        "disk", "Disk galaxy", "gravity", dt=1e-4, softening=0.1,
        description="A rotating stellar disk in a dark-matter halo.",
        details=(
            "PHYSICS  Collisionless N-body gravity. Stars start on near-circular "
            "orbits in an exponential disk, held together by a fixed analytic "
            "dark-matter halo + bulge potential.\n"
            "WHAT YOU SEE  A flat rotating spiral-like disk that winds up over "
            "time; differential rotation shears any initial clumps into arms.\n"
            "PARAMS  none exposed (set N and seed above)."),
        params={}),
    "merger": Scenario(
        "merger", "Galaxy merger", "gravity", dt=1e-4, softening=0.15,
        description="Two disk galaxies collide: tidal tails and bridges.",
        details=(
            "PHYSICS  Two collisionless disks on an approaching orbit, each in "
            "its own moving halo potential. Tides do the rest.\n"
            "WHAT YOU SEE  Long tidal tails, a connecting bridge, then a "
            "disrupted merger remnant.\n"
            "PARAMS  separation (kpc), impact_param (kpc), v_approach (km/s), "
            "inclination (deg of the second disk)."),
        params={"separation": 50.0, "impact_param": 16.0,
                "v_approach": 130.0, "inclination": 60.0}),
    "cosmo": Scenario(
        "cosmo", "Cosmic web", "gravity", dt=5e-5, softening=0.22,
        description="A perturbed box collapses into filaments and clumps.",
        details=(
            "PHYSICS  Zel'dovich-perturbed particle box under self-gravity "
            "(gravity-only). Small density ripples grow by gravitational "
            "instability.\n"
            "WHAT YOU SEE  Matter drains out of voids into sheets, then "
            "filaments, then knots — the cosmic web.\n"
            "PARAMS  hubble (expansion rate proxy), amplitude (initial "
            "perturbation strength)."),
        params={"hubble": 42.0, "amplitude": 1.4}),
    "plummer": Scenario(
        "plummer", "Plummer sphere", "gravity", dt=3e-5, softening=0.05,
        description="Equilibrium test sphere (energy/virial check).",
        details=(
            "PHYSICS  A self-gravitating Plummer sphere in equilibrium — the "
            "standard validation case (energy conservation, virial ratio).\n"
            "WHAT YOU SEE  A steady round cluster that should barely change; "
            "drift here means the integrator/softening is off.\n"
            "PARAMS  total_mass (1e10 Msun), scale_radius (kpc)."),
        params={"total_mass": 10.0, "scale_radius": 1.0}),
    "gas": Scenario(
        "gas", "Gas disk (SPH)", "gas", dt=5e-4, softening=0.2,
        description="An SPH gas disk in a galaxy potential.",
        details=(
            "PHYSICS  Smoothed-particle hydrodynamics (pressure, shocks, "
            "artificial viscosity) for a gas disk orbiting in a fixed galaxy "
            "potential, with self-gravity.\n"
            "WHAT YOU SEE  A rotating gaseous disk that develops pressure-"
            "supported structure, rings and shocks.\n"
            "PARAMS  sound_speed (km/s) — sets gas 'stiffness'/temperature."),
        params={"sound_speed": 12.0}),
    "living": Scenario(
        "living", "Living galaxy", "living", dt=5e-4, softening=0.2,
        description="Gas forms stars; stars age and explode (SN feedback).",
        details=(
            "PHYSICS  SPH gas disk + star formation (dense gas -> stars) + "
            "stellar ageing (age->colour) + supernova feedback that reheats "
            "gas. Self-regulating star formation.\n"
            "WHAT YOU SEE  Young blue stars lighting up spiral features, gas "
            "fountains from SN, the disk slowly converting gas to stars.\n"
            "PARAMS  sf_prob (per-step SF probability of eligible gas), du_sn "
            "(energy injected per supernova)."),
        params={"sf_prob": 0.03, "du_sn": 500.0}),
    "realistic_galaxy": Scenario(
        "realistic_galaxy", "Realistic Galaxy", "galaxy", dt=5e-4, softening=0.2,
        description="A mathematically balanced realistic galaxy (SMBH, DM, Bulge, Disk, Gas).",
        details=(
            "PHYSICS  Uses Hernquist and Toomre Q dispersion profiles to ensure equilibrium.\n"
            "WHAT YOU SEE  A disk that naturally forms spirals and bars.\n"
            "PARAMS  live_halo (True/False to use N-body vs Analytic DM)."
        ),
        params={"live_halo": True, "toomre_q": 1.5, "sf_prob": 0.03, "du_sn": 400.0}
    ),
    "proto_galaxy_collapse": Scenario(
        "proto_galaxy_collapse", "Proto Galaxy Collapse", "galaxy", dt=5e-4, softening=0.2,
        description="A warm gas cloud that collapses into a disk over time.",
        details=(
            "PHYSICS  Gas cools and settles into a disk.\n"
            "WHAT YOU SEE  Formation of a disk.\n"
            "PARAMS  live_halo."
        ),
        params={"live_halo": True, "sf_prob": 0.04, "du_sn": 350.0}
    ),
    "galaxy": Scenario(
        "galaxy", "Full galaxy (stars+DM+gas)", "galaxy", dt=5e-4, softening=0.2,
        description="A complete galaxy: stars, a live dark-matter halo and SPH "
                    "gas, all self-gravitating together.",
        details=(
            "PHYSICS  Live self-gravity over ALL components — stars + a live "
            "dark-matter halo + gas (no analytic potential) — with SPH hydro on "
            "the gas, star formation and supernova feedback. The most physically "
            "complete galaxy here.\n"
            "WHAT YOU SEE  A self-consistent galaxy: a rotating stellar disk in a "
            "responsive DM halo, cold gas forming young blue stars, supernovae "
            "driving fountains. Dust glow is added at render from gas density.\n"
            "PARAMS  sf_prob (star-formation rate), du_sn (SN energy). Gas/star/"
            "DM split scales with N.\n"
            "NOTE  Uses direct N² gravity — heavier than the split scenarios; a "
            "Barnes-Hut tree is the planned speed-up for large N."),
        params={"sf_prob": 0.03, "du_sn": 400.0}),
    "galaxy_merger": Scenario(
        "galaxy_merger", "Galaxy merger (live, gas+stars+DM)", "galaxy",
        dt=5e-4, softening=0.2,
        description="Two full galaxies (stars + live DM + gas) collide: tidal "
                    "tails plus gas shocks and triggered star formation.",
        details=(
            "PHYSICS  Two complete galaxies — each stars + a live dark-matter "
            "halo + SPH gas with star formation & SN feedback — placed on an "
            "approaching orbit; everything self-gravitates (live, no analytic "
            "potential). The richest merger here.\n"
            "WHAT YOU SEE  Tidal tails and bridges (collisionless), shocked gas "
            "and a starburst of young blue stars where the gas piles up, then a "
            "merger remnant. Use Fast gravity (BH) for large N.\n"
            "PARAMS  separation (kpc), v_approach (km/s), impact_param (kpc), "
            "inclination (deg of the 2nd disk), sf_prob, du_sn."),
        params={"separation": 60.0, "v_approach": 120.0, "impact_param": 15.0,
                "inclination": 30.0, "sf_prob": 0.03, "du_sn": 400.0}),
    "cosmos": Scenario(
        "cosmos", "Living cosmos (web + gas + stars)", "galaxy",
        dt=5e-5, softening=0.22,
        description="Cosmological box with baryons: dark matter + gas that cools "
                    "and forms stars in the cosmic-web knots.",
        details=(
            "PHYSICS  A Zel'dovich-perturbed box (cosmic web) where a fraction of "
            "the matter is gas. Live self-gravity over DM + gas, SPH hydro, "
            "radiative cooling, star formation & feedback — the 'living' version "
            "of the cosmic-web scenario.\n"
            "WHAT YOU SEE  Matter drains into filaments and knots; gas cools at "
            "the nodes and lights up as the first stars/galaxies.\n"
            "PARAMS  sf_prob, du_sn (cooling via the Physics panel). Use Fast "
            "gravity (BH) — many particles.\n"
            "NOTE  Isolated (non-periodic) box, like cosmo."),
        params={"sf_prob": 0.03, "du_sn": 350.0}),
    "formation": Scenario(
        "formation", "Galaxy formation (gas → disk → stars)", "galaxy",
        dt=5e-4, softening=0.2,
        description="A warm rotating gas cloud in a DM halo cools, settles into a "
                    "disk, and turns into stars — composition evolves over time.",
        details=(
            "PHYSICS  A live dark-matter halo + a warm, slowly-rotating gas cloud. "
            "With cooling on, the gas radiates energy, collapses along its spin "
            "axis into a rotationally-supported disk, and forms stars — so the "
            "baryon mix evolves from gas-dominated to star-dominated.\n"
            "WHAT YOU SEE  A diffuse gas blob spinning up, flattening into a disk, "
            "then growing a young stellar disk from the inside out.\n"
            "PARAMS  sf_prob, du_sn; cooling (Physics panel) is essential here."),
        params={"sf_prob": 0.04, "du_sn": 350.0}),
    "impact": Scenario(
        "impact", "Giant impact", "impact", dt=5e-4, softening=0.1,
        description="Two self-gravitating bodies collide (shock heating).",
        details=(
            "PHYSICS  Two self-gravitating SPH bodies (no external potential) "
            "collide; the contact shock heats material (coloured by internal "
            "energy). Planetary-scale, not galactic.\n"
            "WHAT YOU SEE  A giant impact: shock-heated ejecta, a debris disk, "
            "possible re-accretion — Moon-forming-impact style.\n"
            "PARAMS  v_approach (km/s), impact_param (offset of the second "
            "body; 0 = head-on)."),
        params={"v_approach": 30.0, "impact_param": 1.6}),
}

# Zero-potential params reused by the giant-impact SPH engine.
_NO_POTENTIAL = dict(M_d=0.0, a=1.0, b=1.0, M_h=0.0, a_h=1.0)


def build_scenario(name: str, n: int = 20000, seed: int = 0,
                   overrides: dict | None = None,
                   gravity_mode: str = "direct", theta: float = 0.6,
                   engine_params: dict | None = None):
    """Build a scenario's engine (ICs loaded) and return (engine, dt).

    ``gravity_mode`` (``"direct"`` | ``"bh"``) selects the gravity solver for
    gravity-kind scenarios; ``"bh"`` is the Barnes-Hut treecode for large N.
    ``engine_params`` carries the physics knobs (cooling/u_floor/t_cool,
    sf_prob, du_sn) edited in the GUI's Physics panel.
    """
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario '{name}'. Have: {list(SCENARIOS)}")
    sc = SCENARIOS[name]
    p = dict(sc.params)
    if overrides:
        p.update({k: v for k, v in overrides.items() if v is not None})
    ep = engine_params or {}

    if name == "empty":
        import numpy as np
        from core.engine import Engine
        from core.state import State, PTYPE_STAR
        # A single faint seed particle at the origin; paint the rest with the brush.
        state = State(pos=np.zeros((1, 3)), vel=np.zeros((1, 3)),
                      mass=np.full(1, 1e-6), ptype=np.full(1, PTYPE_STAR, np.int32),
                      ids=np.arange(1, dtype=np.int64))
        eng = Engine(softening=sc.softening, gravity_mode=gravity_mode, theta=theta)
        eng.load_state(state)
        return eng, sc.dt

    if sc.kind == "gravity":
        from core.scene import Scene, build_state
        from core.engine import Engine
        state = build_state(Scene(ic=name, seed=seed, params=p), n, "direct")
        eng = Engine(softening=sc.softening, gravity_mode=gravity_mode,
                     theta=theta)
        eng.load_state(state)
        return eng, sc.dt

    if sc.kind == "gas":
        from core.ic.gas_disk import make_gas_disk
        from core.gas_disk_engine import GasDiskEngine
        pos, vel, mass, u = make_gas_disk(
            n=n, seed=seed, sound_speed=p.get("sound_speed", 12.0))
        eng = GasDiskEngine(softening=sc.softening, gravity_mode=gravity_mode,
                            theta=theta, **_cool_kw(ep))
        eng.setup(pos, vel, mass, u)
        return eng, sc.dt

    if sc.kind == "living":
        from core.ic.gas_disk import make_gas_disk
        from core.living_galaxy_engine import LivingGalaxyEngine
        pos, vel, mass, u = make_gas_disk(n=n, seed=seed)
        eng = LivingGalaxyEngine(softening=sc.softening,
                                 gravity_mode=gravity_mode, theta=theta,
                                 sf_prob=ep.get("sf_prob", p.get("sf_prob", 0.03)),
                                 du_sn=ep.get("du_sn", p.get("du_sn", 500.0)),
                                 **_cool_kw(ep))
        eng.setup(pos, vel, mass, u)
        return eng, sc.dt

    if sc.kind == "galaxy":
        from core.living_galaxy_engine import LivingGalaxyEngine
        if name == "galaxy_merger":
            from core.ic.full_galaxy import make_galaxy_merger
            pos, vel, mass, u, species = make_galaxy_merger(
                n=n, seed=seed,
                separation=p.get("separation", 60.0),
                v_approach=p.get("v_approach", 120.0),
                impact_param=p.get("impact_param", 15.0),
                inclination=p.get("inclination", 30.0))
            pot = dict(_NO_POTENTIAL)
        elif name == "cosmos":
            from core.ic.full_galaxy import make_cosmo_baryon
            pos, vel, mass, u, species = make_cosmo_baryon(n=n, seed=seed)
            pot = dict(_NO_POTENTIAL)
        elif name == "formation":
            from core.ic.full_galaxy import make_protogalaxy
            pos, vel, mass, u, species = make_protogalaxy(n=n, seed=seed)
            pot = dict(_NO_POTENTIAL)
        elif name in ("realistic_galaxy", "proto_galaxy_collapse"):
            from core.ic.realistic_galaxy import make_realistic_galaxy
            live_halo = p.get("live_halo", True)
            pos, vel, mass, u, species = make_realistic_galaxy(
                n=n, seed=seed, live_halo=live_halo,
                toomre_q=p.get("toomre_q", 1.5), proto=(name == "proto_galaxy_collapse"))
            pot = dict(_NO_POTENTIAL) if live_halo else dict(
                M_d=0.0, a=1.0, b=1.0, M_h=40.0, a_h=15.0, is_hernquist=1, smbh_mass=0.1)
        else:
            from core.ic.full_galaxy import make_full_galaxy
            pos, vel, mass, u, species = make_full_galaxy(n=n, seed=seed)
            pot = dict(_NO_POTENTIAL)
        eng = LivingGalaxyEngine(
            pot=pot, softening=sc.softening,
            gravity_mode=gravity_mode, theta=theta,
            sf_prob=ep.get("sf_prob", p.get("sf_prob", 0.03)),
            du_sn=ep.get("du_sn", p.get("du_sn", 400.0)), **_cool_kw(ep))
        eng.setup(pos, vel, mass, u, species=species)
        return eng, sc.dt

    if sc.kind == "impact":
        from core.ic.impact import make_impact
        from core.gas_disk_engine import GasDiskEngine
        pos, vel, mass, u = make_impact(
            n_each=max(n // 2, 1), seed=seed,
            v_approach=p.get("v_approach", 30.0),
            impact_param=p.get("impact_param", 1.6))
        # Planetary impact: no radiative cooling -- keep the shock heat (glow).
        eng = GasDiskEngine(pot=dict(_NO_POTENTIAL), softening=sc.softening,
                            cooling=False, gravity_mode=gravity_mode, theta=theta)
        eng.setup(pos, vel, mass, u)
        return eng, sc.dt

    raise ValueError(f"unhandled scenario kind: {sc.kind}")


def _cool_kw(ep: dict) -> dict:
    """Cooling keyword args for the SPH engines, from engine_params."""
    return dict(cooling=ep.get("cooling", True),
                u_floor=ep.get("u_floor", 60.0),
                t_cool=ep.get("t_cool", 0.02))


def resume_scenario(name: str, state, spec=None):
    """Rebuild a scenario's engine from a saved State and return (engine, dt).

    Engine configuration (kind, dt, softening) and tuned parameters come from
    ``spec`` (a :class:`core.scene_spec.SceneSpec`) when available -- so a resumed
    run keeps any values the user changed -- and otherwise fall back to the
    scenario registry defaults (older snapshots that predate the scene spec).

    Gravity and gas/impact resume exactly; living-galaxy resume restores the
    star flags and ages but re-derives SPH state (a close, not bit-exact, resume).
    """
    sc = SCENARIOS.get(name)
    if spec is not None:
        kind = spec.engine_kind
        dt = spec.dt
        softening = spec.softening
        params = dict(spec.primary.params) if spec.objects else {}
    else:
        kind = sc.kind if sc else "gravity"
        dt = sc.dt if sc else 1e-4
        softening = sc.softening if sc else 0.1
        params = {}

    # Self-gravity solver carried by the scene spec (falls back for old files).
    gmode = spec.gravity_mode if spec is not None else "direct"
    gtheta = spec.theta if spec is not None else 0.6

    if kind == "gravity":
        from core.engine import Engine
        eng = Engine(softening=softening, gravity_mode=gmode, theta=gtheta)
        eng.load_state(state)
        return eng, dt

    if kind in ("gas", "impact"):
        from core.gas_disk_engine import GasDiskEngine
        pot = dict(_NO_POTENTIAL) if kind == "impact" else None
        eng = GasDiskEngine(pot=pot, softening=softening,
                            gravity_mode=gmode, theta=gtheta,
                            cooling=(kind != "impact"))
        eng.setup(state.pos, state.vel, state.mass,
                  state.u if state.u is not None else _default_u(state.n))
        eng.time, eng.step_count = state.time, state.step
        return eng, dt

    if kind == "galaxy":
        import numpy as np
        from core.living_galaxy_engine import LivingGalaxyEngine
        from core.state import PTYPE_STAR, PTYPE_DM
        eng = LivingGalaxyEngine(
            pot=dict(_NO_POTENTIAL), softening=softening,
            gravity_mode=gmode, theta=gtheta,
            sf_prob=params.get("sf_prob", 0.03),
            du_sn=params.get("du_sn", 400.0))
        species = np.zeros(state.n, np.int32)
        species[state.ptype == PTYPE_STAR] = 1
        species[state.ptype == PTYPE_DM] = 2
        eng.setup(state.pos, state.vel, state.mass,
                  state.u if state.u is not None else _default_u(state.n),
                  species=species)
        eng.time, eng.step_count = state.time, state.step
        if state.age is not None:
            birth = np.where(state.age >= 0.0, state.time - state.age, 0.0)
            eng._f["birth"].from_numpy(birth.astype(np.float64))
        return eng, dt

    if kind == "living":
        import numpy as np
        from core.living_galaxy_engine import LivingGalaxyEngine
        from core.state import PTYPE_STAR
        eng = LivingGalaxyEngine(softening=softening,
                                 gravity_mode=gmode, theta=gtheta,
                                 sf_prob=params.get("sf_prob", 0.03),
                                 du_sn=params.get("du_sn", 500.0))
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
        return eng, dt

    raise ValueError(f"cannot resume scenario kind: {kind}")


def _default_u(n):
    import numpy as np
    return np.full(n, 200.0)
