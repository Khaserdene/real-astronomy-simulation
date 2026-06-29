"""Curated presets -- tuned configurations that lead to good-looking results.

Each preset is a set of controller settings (scenario, particle count, gravity
solver, and the physics knobs) chosen to produce a recognisable, "looks like a
real galaxy/merger/web" outcome without hand-tuning.  Selecting a preset in the
GUI fills those settings in; the user can still tweak afterwards.
"""
from __future__ import annotations

# Keys map onto core.sim_controller.Settings fields.
PRESETS: dict[str, dict] = {
    "Realistic spiral galaxy": dict(
        scenario="galaxy", n=40000, gravity_mode="bh",
        cooling=True, u_floor=45.0, t_cool=0.015,
        eps_ff=0.01, du_sn=350.0, v_sn=50.0,
        t_sn_max=0.05, r_fb=0.6, f_return=0.4),
    "Living galaxy (stars from gas)": dict(
        scenario="living", n=30000, gravity_mode="direct",
        cooling=True, u_floor=55.0, t_cool=0.02,
        eps_ff=0.015, du_sn=500.0, v_sn=50.0,
        t_sn_max=0.04, r_fb=0.6, f_return=0.35),
    "Gas-rich disk (cool, thin)": dict(
        scenario="gas", n=30000, gravity_mode="direct",
        cooling=True, u_floor=40.0, t_cool=0.01),
    "Galaxy merger (tidal tails)": dict(
        scenario="merger", n=40000, gravity_mode="bh"),
    "Cosmic web": dict(
        scenario="cosmo", n=30000, gravity_mode="bh"),
    "Giant impact (planetary)": dict(
        scenario="impact", n=16000, gravity_mode="direct", cooling=False),
    "Realistic Galaxy (real)": dict(
        scenario="realistic_galaxy", n=40000, gravity_mode="bh",
        cooling=True, u_floor=45.0, t_cool=0.015,
        eps_ff=0.01, du_sn=400.0, v_sn=60.0,
        t_sn_max=0.05, r_fb=0.6, f_return=0.4),
    "Proto Galaxy Collapse (real)": dict(
        scenario="proto_galaxy_collapse", n=40000, gravity_mode="bh",
        cooling=True, u_floor=50.0, t_cool=0.025,
        # Feedback strong enough that supernovae are actually visible (heated gas
        # glows magenta/white) and drive outflows; was 50/10 which was invisible.
        eps_ff=0.008, du_sn=300.0, v_sn=40.0,
        sf_density_factor=12.0,
        t_sn_max=0.05, r_fb=0.8, f_return=0.35),
}


def apply_preset(settings, name: str) -> None:
    """Write a preset's values onto a Settings instance (in place)."""
    cfg = PRESETS.get(name)
    if not cfg:
        return
    for key, value in cfg.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
