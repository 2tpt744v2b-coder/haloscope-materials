"""Tests for q_factor.py.

The 'illustrative' conductivity below is NOT a database value; it is a round
number used only so the arithmetic can be checked by hand.
"""

import math

import pytest
from scipy.integrate import quad
from scipy.special import j1, jn_zeros

from haloscope_materials import q_factor as qf

ILLUSTRATIVE_SIGMA = 5.8e7  # S/m, illustrative only
F_5GHZ = 5.0e9  # Hz


def test_x01_is_first_zero_of_bessel_j0():
    assert qf.X01 == pytest.approx(jn_zeros(0, 1)[0], rel=1e-12)


def test_tm010_radius_at_5ghz():
    # Hand check: 2.4048 * 2.998e8 / (2*pi*5e9) = 0.02295 m
    assert qf.tm010_radius(F_5GHZ) == pytest.approx(0.02295, rel=5e-3)


def test_geometry_factor_equal_radius_and_length():
    # Hand check for d = a: G = 2.4048 * 376.73 / 4 = 226.5 ohm
    assert qf.tm010_geometry_factor(0.02, 0.02) == pytest.approx(226.5, rel=1e-3)


def test_geometry_factor_long_cavity_limit():
    # d >> a: G -> x01 * eta0 / 2 = 453 ohm
    assert qf.tm010_geometry_factor(0.02, 1000.0) == pytest.approx(453.0, rel=2e-3)


def test_surface_resistance_classical_hand_check():
    # Hand check at 5 GHz with sigma = 5.8e7: R_s = 0.0184 ohm
    assert qf.surface_resistance_classical(F_5GHZ, ILLUSTRATIVE_SIGMA) == pytest.approx(
        0.01845, rel=5e-3
    )


def test_skin_depth_hand_check():
    # Hand check: about 0.93 micrometres
    assert qf.skin_depth_classical(F_5GHZ, ILLUSTRATIVE_SIGMA) == pytest.approx(
        0.93e-6, rel=1e-2
    )


def test_rs_and_skin_depth_are_consistent():
    # R_s = 1 / (sigma * delta)
    delta = qf.skin_depth_classical(F_5GHZ, ILLUSTRATIVE_SIGMA)
    rs = qf.surface_resistance_classical(F_5GHZ, ILLUSTRATIVE_SIGMA)
    assert rs == pytest.approx(1.0 / (ILLUSTRATIVE_SIGMA * delta), rel=1e-9)


def test_q_wall_matches_textbook_closed_form():
    # Consistency check only (same derivation as G / R_s, NOT independent):
    # Q = (a/delta) / (1 + a/d) for TM010.
    a = qf.tm010_radius(F_5GHZ)
    d = 0.10
    delta = qf.skin_depth_classical(F_5GHZ, ILLUSTRATIVE_SIGMA)
    expected = (a / delta) / (1.0 + a / d)
    g = qf.tm010_geometry_factor(a, d)
    rs = qf.surface_resistance_classical(F_5GHZ, ILLUSTRATIVE_SIGMA)
    assert qf.q_wall(g, rs) == pytest.approx(expected, rel=1e-6)
    # Hand check: about 2.0e4
    assert qf.q_wall(g, rs) == pytest.approx(2.0e4, rel=2e-2)


def test_q_dielectric_known_value():
    assert qf.q_dielectric(1e-4, 0.5) == pytest.approx(2.0e4, rel=1e-12)


def test_q_combined_equal_channels_halves_q():
    assert qf.q_combined([1.0e5, 1.0e5]) == pytest.approx(5.0e4, rel=1e-12)


def test_q_combined_dominated_by_worst_channel():
    assert qf.q_combined([1.0e3, 1.0e9]) == pytest.approx(1.0e3, rel=1e-5)


def test_q_wall_with_uncertainty_relative_error_matches_rs():
    q, dq = qf.q_wall_with_uncertainty(300.0, 0.02, 0.002)  # 10 percent on R_s
    assert q == pytest.approx(15000.0)
    assert dq / q == pytest.approx(0.10, rel=1e-12)


def test_q_combined_with_uncertainty_equal_channels():
    # Two equal channels, each Q=1e5 +/- 1e4. By hand:
    # dx_i = 1e4/1e10 = 1e-6; d(sum) = sqrt(2)*1e-6; Q_tot = 5e4
    # dQ_tot = (5e4)^2 * sqrt(2)*1e-6 = 3535.5
    q, dq = qf.q_combined_with_uncertainty([1e5, 1e5], [1e4, 1e4])
    assert q == pytest.approx(5e4)
    assert dq == pytest.approx(2.5e9 * math.sqrt(2) * 1e-6, rel=1e-9)


def test_classical_regime_ratio():
    assert qf.classical_regime_ratio(1e-7, 1e-6) == pytest.approx(0.1)


@pytest.mark.parametrize(
    "call",
    [
        lambda: qf.tm010_radius(0.0),
        lambda: qf.tm010_geometry_factor(-1.0, 1.0),
        lambda: qf.surface_resistance_classical(1e9, 0.0),
        lambda: qf.q_wall(0.0, 1.0),
        lambda: qf.q_dielectric(1e-4, 1.5),
        lambda: qf.q_dielectric(-1e-4, 0.5),
        lambda: qf.q_combined([]),
        lambda: qf.q_combined([1e5, -1.0]),
    ],
)
def test_invalid_inputs_raise(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize("radius,length", [(0.02, 0.02), (0.02, 0.10), (0.0229, 0.30)])
def test_geometry_factor_matches_field_integrals(radius, length):
    """Independent check: G = omega*mu0 * int|H|^2 dV / int|H_t|^2 dS.

    TM010 has H_phi proportional to J1(k r), k = x01/a, uniform in z. Common
    amplitude factors cancel. Integrals done numerically over the volume,
    the sidewall and both endplates.
    """
    k = qf.X01 / radius
    omega = qf.X01 * qf.SPEED_OF_LIGHT / radius
    radial = quad(lambda r: j1(k * r) ** 2 * r, 0.0, radius)[0]
    volume = length * 2.0 * math.pi * radial
    sidewall = j1(k * radius) ** 2 * 2.0 * math.pi * radius * length
    endplates = 2.0 * 2.0 * math.pi * radial
    g_integral = omega * qf.MU_0 * volume / (sidewall + endplates)
    assert qf.tm010_geometry_factor(radius, length) == pytest.approx(g_integral, rel=1e-8)
