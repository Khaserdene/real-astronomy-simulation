"""Galactic unit system used throughout the engine.

We work in a self-consistent set of units common in galactic dynamics so that
numbers stay O(1)-ish and the gravitational constant is a fixed scalar:

    length   : kpc
    velocity : km / s
    mass     : 1e10 solar masses          (Msun10)
    time     : kpc / (km/s) ~= 0.9778 Gyr  (derived)

In these units Newton's constant is

    G = 4.30091e-6 kpc (km/s)^2 / Msun  *  1e10 Msun / Msun10
      = 43009.1  kpc (km/s)^2 / Msun10

All state stored on disk (snapshots/checkpoints) is in *these physical units*,
which is what makes a run portable between machines and resolution levels.
"""

# Gravitational constant in (kpc, km/s, 1e10 Msun) units.
G = 43009.1725

# Handy conversions for reporting / I/O.
KPC_PER_KM = 3.240779289e-17          # 1 km in kpc
GYR_PER_TIMEUNIT = 0.97781            # 1 (kpc/(km/s)) in Gyr
MSUN_PER_MASSUNIT = 1.0e10            # 1 mass unit in solar masses

# Human-readable unit labels (for plots / logs).
LENGTH_UNIT = "kpc"
VELOCITY_UNIT = "km/s"
MASS_UNIT = "1e10 Msun"
TIME_UNIT = "kpc/(km/s) ~= 0.978 Gyr"


def time_to_gyr(t: float) -> float:
    """Convert a simulation time (code units) to Gyr."""
    return t * GYR_PER_TIMEUNIT
