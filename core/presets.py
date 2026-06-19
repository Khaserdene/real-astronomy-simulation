"""Curated presets -- tuned configurations that lead to good-looking results.

Each preset is a set of controller settings (scenario, particle count, gravity
solver, and the physics knobs) chosen to produce a recognisable, "looks like a
real galaxy/merger/web" outcome without hand-tuning.  Selecting a preset in the
GUI fills those settings in; the user can still tweak afterwards.

(A future option is automatic tuning -- searching parameters against a target --
but that needs an objective metric and many runs; these hand-tuned presets are
the practical starting point.)
"""
from __future__ import annotations

# Keys map onto core.sim_controller.Settings fields.
PRESETS: dict[str, dict] = {
    "Realistic spiral galaxy": dict(
        scenario="galaxy", n=40000, gravity_mode="bh",
        cooling=True, u_floor=45.0, t_cool=0.015, sf_prob=0.02, du_sn=350.0),
    "Living galaxy (stars from gas)": dict(
        scenario="living", n=30000, gravity_mode="direct",
        cooling=True, u_floor=55.0, t_cool=0.02, sf_prob=0.03, du_sn=500.0),
    "Gas-rich disk (cool, thin)": dict(
        scenario="gas", n=30000, gravity_mode="direct",
        cooling=True, u_floor=40.0, t_cool=0.01),
    "Galaxy merger (tidal tails)": dict(
        scenario="merger", n=40000, gravity_mode="bh"),
    "Cosmic web": dict(
        scenario="cosmo", n=30000, gravity_mode="bh"),
    "Giant impact (planetary)": dict(
        scenario="impact", n=16000, gravity_mode="direct", cooling=False),
}


def apply_preset(settings, name: str) -> None:
    """Write a preset's values onto a Settings instance (in place)."""
    cfg = PRESETS.get(name)
    if not cfg:
        return
    for key, value in cfg.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
