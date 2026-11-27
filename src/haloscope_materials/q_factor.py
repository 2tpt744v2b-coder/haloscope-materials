"""Cavity quality factor (Q) building blocks.

All quantities are SI unless stated otherwise:
    frequency          -> hertz (Hz)
    length / radius    -> metres (m)
    conductivity       -> siemens per metre (S/m)
    surface resistance -> ohms (Ohm)
    geometry factor    -> ohms (Ohm)
    Q, tan_delta, fill -> dimensionless

SCOPE OF THIS MODULE (Step 3A):
    * Cylindrical cavity, TM010 mode only.
    * Perfectly smooth walls, no coupling ports, no tuning rod.
    * CLASSICAL (normal) skin effect only. This is NOT valid for pure metals
      at cryogenic temperatures, where the anomalous skin effect applies
      (Step 3B). See classical_regime_ratio().

UNVERIFIED: formulas below were derived by hand and must be confirmed against
a textbook (e.g. Pozar, Microwave Engineering; Jackson, Classical
Electrodynamics) before being trusted for real design decisions.
"""

import math
from typing import Sequence

from scipy.constants import c as SPEED_OF_LIGHT  # m/s
from scipy.constants import mu_0 as MU_0  # H/m (vacuum permeability)

# First zero of the Bessel function J0. Mathematical constant (dimensionless).
X01 = 2.404825557695773

# Impedance of free space, in ohms (= mu_0 * c).
ETA_0 = MU_0 * SPEED_OF_LIGHT


def tm010_radius(frequency_hz: float) -> float:
    """Radius (m) of a cylindrical cavity whose TM010 mode resonates at frequency_hz.

    TM010 resonance depends only on radius: f = x01 * c / (2 * pi * a).
    """
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    return X01 * SPEED_OF_LIGHT / (2.0 * math.pi * frequency_hz)


def tm010_geometry_factor(radius_m: float, length_m: float) -> float:
    """Geometry factor G (ohms) of a cylindrical cavity in TM010: G = x01*eta0*d / (2*(a+d))."""
    if radius_m <= 0 or length_m <= 0:
        raise ValueError("radius_m and length_m must be positive")
    return X01 * ETA_0 * length_m / (2.0 * (radius_m + length_m))


def skin_depth_classical(frequency_hz: float, conductivity_s_per_m: float) -> float:
    """Classical skin depth (m): delta = sqrt(2 / (omega * mu0 * sigma)).

    Valid only when the electron mean free path is much shorter than delta.
    """
    if frequency_hz <= 0 or conductivity_s_per_m <= 0:
        raise ValueError("frequency_hz and conductivity_s_per_m must be positive")
    omega = 2.0 * math.pi * frequency_hz
    return math.sqrt(2.0 / (omega * MU_0 * conductivity_s_per_m))


def surface_resistance_classical(frequency_hz: float, conductivity_s_per_m: float) -> float:
    """Classical surface resistance (ohms): R_s = sqrt(omega * mu0 / (2 * sigma))."""
    if frequency_hz <= 0 or conductivity_s_per_m <= 0:
        raise ValueError("frequency_hz and conductivity_s_per_m must be positive")
    omega = 2.0 * math.pi * frequency_hz
    return math.sqrt(omega * MU_0 / (2.0 * conductivity_s_per_m))


def classical_regime_ratio(mean_free_path_m: float, skin_depth_m: float) -> float:
    """Return mean_free_path / skin_depth (dimensionless).

    The classical skin-effect formulas are trustworthy only when this ratio is
    much smaller than 1. A value near or above 1 means the anomalous skin
    effect matters and the classical results will be too optimistic.
    (The exact threshold is a rule of thumb: UNVERIFIED.)
    """
    if mean_free_path_m <= 0 or skin_depth_m <= 0:
        raise ValueError("mean_free_path_m and skin_depth_m must be positive")
    return mean_free_path_m / skin_depth_m


def q_wall(geometry_factor_ohm: float, surface_resistance_ohm: float) -> float:
    """Wall-limited quality factor: Q_wall = G / R_s (dimensionless)."""
    if geometry_factor_ohm <= 0 or surface_resistance_ohm <= 0:
        raise ValueError("geometry_factor_ohm and surface_resistance_ohm must be positive")
    return geometry_factor_ohm / surface_resistance_ohm


def q_dielectric(loss_tangent: float, fill_factor: float) -> float:
    """Dielectric-limited quality factor: Q_d = 1 / (p * tan_delta).

    Args:
        loss_tangent: tan(delta) of the dielectric, dimensionless, > 0.
        fill_factor: p, fraction of electric energy stored in the dielectric,
            between 0 (exclusive) and 1 (inclusive).
    """
    if loss_tangent <= 0:
        raise ValueError("loss_tangent must be positive")
    if not 0 < fill_factor <= 1:
        raise ValueError("fill_factor must be in (0, 1]")
    return 1.0 / (fill_factor * loss_tangent)


def q_combined(q_values: Sequence[float]) -> float:
    """Combine independent loss channels: 1/Q_total = sum(1/Q_i)."""
    if len(q_values) == 0:
        raise ValueError("q_values must not be empty")
    if any(q <= 0 for q in q_values):
        raise ValueError("every Q must be positive")
    return 1.0 / sum(1.0 / q for q in q_values)


def q_wall_with_uncertainty(
    geometry_factor_ohm: float,
    surface_resistance_ohm: float,
    surface_resistance_uncertainty_ohm: float,
) -> tuple[float, float]:
    """Return (Q_wall, uncertainty in Q_wall).

    Assumes the geometry factor is exact and only R_s is uncertain. Since
    Q = G / R_s, the RELATIVE uncertainty of Q equals that of R_s
    (first-order approximation, fine for uncertainties below about 20 percent).
    """
    if surface_resistance_uncertainty_ohm < 0:
        raise ValueError("uncertainty must not be negative")
    q = q_wall(geometry_factor_ohm, surface_resistance_ohm)
    relative = surface_resistance_uncertainty_ohm / surface_resistance_ohm
    return q, q * relative


def q_combined_with_uncertainty(
    q_values: Sequence[float], q_uncertainties: Sequence[float]
) -> tuple[float, float]:
    """Return (Q_total, uncertainty) for independent channels.

    Works on 1/Q: if x_i = 1/Q_i then dx_i = dQ_i / Q_i**2. For independent
    errors, d(sum x) = sqrt(sum dx_i**2), and then dQ_total = Q_total**2 * d(sum x).
    Assumes the uncertainties are independent and small (first order).
    """
    if len(q_values) != len(q_uncertainties):
        raise ValueError("q_values and q_uncertainties must have the same length")
    if any(u < 0 for u in q_uncertainties):
        raise ValueError("uncertainties must not be negative")
    q_total = q_combined(q_values)
    d_inverse_sum = math.sqrt(sum((u / q**2) ** 2 for q, u in zip(q_values, q_uncertainties)))
    return q_total, q_total**2 * d_inverse_sum
