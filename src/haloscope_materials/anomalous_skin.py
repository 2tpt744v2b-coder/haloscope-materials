"""Surface resistance of a normal metal in the EXTREME anomalous skin-effect limit.

Formula (real part of the surface impedance when mean free path >> skin depth):

    R_s = eta0 * [ (sqrt(3) * v_F) / (16 * pi * c) * (omega / omega_p)**2 ]**(1/3)

Units (SI):
    frequency_hz                      -> Hz
    fermi_velocity_m_per_s            -> m/s
    plasma_angular_frequency_rad_per_s-> rad/s
    return value                      -> ohms

SOURCES (both read via machine-extracted text, so UNVERIFIED until you check
the equations in the PDFs yourself):
    * Lawler, Bosco, Rosenzweig, arXiv:2310.11578, Eq. (3)
    * arXiv:1603.01657, Eq. (9) (real part of the surface impedance)

LIMITS OF VALIDITY (UNVERIFIED, from the same sources):
    * Only the extreme limit (mean free path much longer than the skin depth).
      It does NOT cover the crossover regime between classical and extreme
      behaviour; the papers say that regime is poorly predicted.
    * Assumes a smooth, bulk, single-material wall. No thin-film or surface-
      roughness effects.
    * Material constants (v_F, omega_p) are inputs; this module contains none.
      Every value you pass in must have a cited source in the database.
"""

import math

from scipy.constants import c as SPEED_OF_LIGHT  # m/s
from scipy.constants import mu_0 as MU_0  # H/m

ETA_0 = MU_0 * SPEED_OF_LIGHT  # impedance of free space, ohms


def surface_resistance_extreme_anomalous(
    frequency_hz: float,
    fermi_velocity_m_per_s: float,
    plasma_angular_frequency_rad_per_s: float,
) -> float:
    """Extreme anomalous-limit surface resistance (ohms). See module docstring."""
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    if fermi_velocity_m_per_s <= 0:
        raise ValueError("fermi_velocity_m_per_s must be positive")
    if plasma_angular_frequency_rad_per_s <= 0:
        raise ValueError("plasma_angular_frequency_rad_per_s must be positive")
    omega = 2.0 * math.pi * frequency_hz
    inner = (
        math.sqrt(3.0)
        * fermi_velocity_m_per_s
        / (16.0 * math.pi * SPEED_OF_LIGHT)
        * (omega / plasma_angular_frequency_rad_per_s) ** 2
    )
    return ETA_0 * inner ** (1.0 / 3.0)
