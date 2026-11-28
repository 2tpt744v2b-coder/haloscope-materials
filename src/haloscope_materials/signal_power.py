"""Axion-to-photon signal power in a haloscope cavity.

    P = g^2 * (rho_a / m_a) * B^2 * V * C * Q_L       (natural units: hbar = c = 1)

Natural units are converted to SI inside this module. Inputs and outputs are SI,
except where a name says otherwise (GeV, cm^3).

Units:
    frequency_hz -> Hz          b_tesla -> T          volume_m3 -> m^3
    form_factor, loaded_q -> dimensionless
    dm_density_gev_per_cm3 -> GeV/cm^3
    mass_coefficient_ev -> eV   (see below)
    return value -> watts

SOURCE FOR THE FORMULA'S SHAPE AND NORMALISATION (UNVERIFIED until you check):
    ADMX Collaboration, Phys. Rev. Lett. 127, 261803 (2021), Eq. (1):
    7.7e-23 W at V = 136 L, B = 7.5 T, C = 0.4, g_gamma = 0.36,
    rho = 0.45 GeV/cm^3, f = 1 GHz, Q_L = 80,000.
    That paper lists g_gamma = -0.97 for KSVZ axions and 0.36 for DFSZ axions.

CONVENTION WARNING:
    The axion mass is tied to its decay constant f_a by
        m_a = mass_coefficient * (1e12 GeV / f_a).
    The coefficient is passed in explicitly because the right choice is not
    settled here. 5.7e-6 eV is a commonly quoted value (UNVERIFIED). With it we
    get about 18 percent MORE power than the paper. 6.2e-6 eV reproduces the
    paper's number, but that value was reverse-engineered, NOT taken from a
    source. Treat it as a hypothesis to check.
"""

import math

from scipy.constants import alpha as FINE_STRUCTURE
from scipy.constants import c as SPEED_OF_LIGHT
from scipy.constants import e as ELEMENTARY_CHARGE
from scipy.constants import h as PLANCK
from scipy.constants import hbar as HBAR
from scipy.constants import mu_0 as MU_0

GEV_IN_JOULE = 1.0e9 * ELEMENTARY_CHARGE  # J per GeV
HBAR_C_J_M = HBAR * SPEED_OF_LIGHT  # J m
HBAR_C_GEV_M = HBAR_C_J_M / GEV_IN_JOULE  # GeV m


def magnetic_field_to_natural(b_tesla: float) -> float:
    """B in tesla -> B in natural units (GeV^2), Heaviside-Lorentz convention.

    Uses energy density u = B^2 / mu0 (J/m^3) converted to GeV^4.
    """
    if b_tesla < 0:
        raise ValueError("b_tesla must not be negative")
    u_si = b_tesla**2 / MU_0  # J/m^3
    return math.sqrt(u_si * HBAR_C_J_M**3 / GEV_IN_JOULE**4)


def volume_to_natural(volume_m3: float) -> float:
    """Volume in m^3 -> GeV^-3."""
    if volume_m3 <= 0:
        raise ValueError("volume_m3 must be positive")
    return volume_m3 / HBAR_C_GEV_M**3


def density_to_natural(density_gev_per_cm3: float) -> float:
    """Dark matter density in GeV/cm^3 -> GeV^4."""
    if density_gev_per_cm3 <= 0:
        raise ValueError("density must be positive")
    return density_gev_per_cm3 * 1.0e6 * HBAR_C_GEV_M**3


def axion_mass_ev(frequency_hz: float) -> float:
    """Axion rest energy in eV for a photon at frequency_hz (m c^2 = h f)."""
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    return PLANCK * frequency_hz / ELEMENTARY_CHARGE


def decay_constant_gev(mass_ev: float, mass_coefficient_ev: float) -> float:
    """f_a in GeV from m_a = mass_coefficient * (1e12 GeV / f_a)."""
    if mass_ev <= 0 or mass_coefficient_ev <= 0:
        raise ValueError("mass_ev and mass_coefficient_ev must be positive")
    return mass_coefficient_ev / mass_ev * 1.0e12


def axion_photon_coupling_per_gev(g_gamma: float, f_a_gev: float) -> float:
    """g_a_gamma_gamma = alpha * g_gamma / (pi * f_a), in 1/GeV."""
    if f_a_gev <= 0:
        raise ValueError("f_a_gev must be positive")
    return FINE_STRUCTURE * g_gamma / (math.pi * f_a_gev)


def axion_signal_power_w(
    frequency_hz: float,
    b_tesla: float,
    volume_m3: float,
    form_factor: float,
    loaded_q: float,
    g_gamma: float,
    mass_coefficient_ev: float,
    dm_density_gev_per_cm3: float = 0.45,
) -> float:
    """Axion signal power (watts) in the cavity. See module docstring for caveats."""
    if form_factor <= 0 or loaded_q <= 0:
        raise ValueError("form_factor and loaded_q must be positive")
    m_ev = axion_mass_ev(frequency_hz)
    m_gev = m_ev * 1.0e-9
    f_a = decay_constant_gev(m_ev, mass_coefficient_ev)
    g = axion_photon_coupling_per_gev(g_gamma, f_a)
    power_gev2 = (
        g**2
        * (density_to_natural(dm_density_gev_per_cm3) / m_gev)
        * magnetic_field_to_natural(b_tesla) ** 2
        * volume_to_natural(volume_m3)
        * form_factor
        * loaded_q
    )
    # GeV^2 -> watts: (GeV in joules) * (GeV in joules / hbar)
    return power_gev2 * GEV_IN_JOULE * (GEV_IN_JOULE / HBAR)
