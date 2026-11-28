"""Free-electron-model constants for a metal, derived from its crystal structure.

Units (SI): number density -> m^-3, Fermi velocity -> m/s, plasma frequency -> rad/s.

MODEL (UNVERIFIED for real copper): treats conduction electrons as a free,
isotropic gas. Real metals have non-spherical Fermi surfaces, so these are
estimates, not measurements. No uncertainty is attached; do not invent one.
"""

import math

from scipy.constants import e as ELEMENTARY_CHARGE  # C
from scipy.constants import epsilon_0 as EPSILON_0  # F/m
from scipy.constants import hbar as HBAR  # J s
from scipy.constants import m_e as ELECTRON_MASS  # kg

ATOMS_PER_FCC_CELL = 4  # standard geometry of a face-centred cubic unit cell


def fcc_electron_number_density(lattice_constant_m: float, electrons_per_atom: float = 1.0) -> float:
    """Conduction-electron density (m^-3) = electrons_per_atom * 4 / a^3."""
    if lattice_constant_m <= 0:
        raise ValueError("lattice_constant_m must be positive")
    if electrons_per_atom <= 0:
        raise ValueError("electrons_per_atom must be positive")
    return electrons_per_atom * ATOMS_PER_FCC_CELL / lattice_constant_m**3


def free_electron_fermi_velocity(number_density_m3: float) -> float:
    """v_F = hbar * (3 * pi^2 * n)^(1/3) / m_e, in m/s."""
    if number_density_m3 <= 0:
        raise ValueError("number_density_m3 must be positive")
    k_fermi = (3.0 * math.pi**2 * number_density_m3) ** (1.0 / 3.0)
    return HBAR * k_fermi / ELECTRON_MASS


def free_electron_plasma_frequency(number_density_m3: float) -> float:
    """omega_p = sqrt(n e^2 / (eps0 m_e)), in rad/s (NOT Hz)."""
    if number_density_m3 <= 0:
        raise ValueError("number_density_m3 must be positive")
    return math.sqrt(
        number_density_m3 * ELEMENTARY_CHARGE**2 / (EPSILON_0 * ELECTRON_MASS)
    )
