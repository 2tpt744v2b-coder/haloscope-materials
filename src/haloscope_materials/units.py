"""Unit helpers. The whole project uses SI units internally.

SI base units used here:
    frequency -> hertz (Hz)
    temperature -> kelvin (K)
    magnetic field -> tesla (T)
    length -> metre (m)
"""

GHZ_TO_HZ = 1.0e9  # hertz per gigahertz (exact by definition of the prefix)


def ghz_to_hz(frequency_ghz: float) -> float:
    """Convert a frequency from gigahertz (GHz) to hertz (Hz).

    Args:
        frequency_ghz: frequency in GHz. Must not be negative.

    Returns:
        The frequency in Hz.
    """
    if frequency_ghz < 0:
        raise ValueError("frequency must not be negative")
    return frequency_ghz * GHZ_TO_HZ
