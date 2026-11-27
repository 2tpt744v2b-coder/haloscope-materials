"""Unit helpers. The whole project uses SI units internally.

SI base units used here:
    frequency -> hertz (Hz)
    angular frequency -> radians per second (rad/s)
    temperature -> kelvin (K)
    magnetic field -> tesla (T)
    length -> metre (m)
"""

from scipy.constants import e as ELEMENTARY_CHARGE  # C (exact in the 2019 SI)
from scipy.constants import hbar as HBAR  # J s

GHZ_TO_HZ = 1.0e9  # hertz per gigahertz (exact by definition of the prefix)


def ghz_to_hz(frequency_ghz: float) -> float:
    """Convert a frequency from gigahertz (GHz) to hertz (Hz).

    Args:
        frequency_ghz: frequency in GHz. Must not be negative.

    Returns:
        The frequency in Hz.
    """
    if frequency_ghz < 0:
        raise ValueError("frequency_ghz must not be negative")
    return frequency_ghz * GHZ_TO_HZ


def ev_to_angular_frequency(energy_ev: float) -> float:
    """Convert a photon energy hbar*omega given in electronvolts to omega in rad/s.

    Use this for plasma frequencies quoted as 'hbar * omega_p (eV)'.

    Args:
        energy_ev: energy in eV. Must be positive.

    Returns:
        Angular frequency in rad/s (NOT Hz).
    """
    if energy_ev <= 0:
        raise ValueError("energy_ev must be positive")
    return energy_ev * ELEMENTARY_CHARGE / HBAR
