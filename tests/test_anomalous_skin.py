"""Tests for anomalous_skin.py.

The inputs below (v_F = 1.0e6 m/s, omega_p = 1.0e16 rad/s) are ROUND ILLUSTRATIVE
numbers chosen only so the arithmetic can be checked by hand. They are NOT
values for any real metal.
"""

import pytest

from haloscope_materials import anomalous_skin as ask

VF = 1.0e6  # m/s, illustrative
WP = 1.0e16  # rad/s, illustrative


def test_hand_check_at_5ghz():
    # By hand: inner = sqrt(3)*1e6/(16*pi*2.998e8) * (2*pi*5e9/1e16)**2
    #                = 1.1494e-4 * 9.8696e-12 = 1.1344e-15
    # cube root = 1.0426e-5 ; times eta0 (376.73 ohm) = 3.929e-3 ohm
    assert ask.surface_resistance_extreme_anomalous(5.0e9, VF, WP) == pytest.approx(
        3.929e-3, rel=1e-3
    )


def test_frequency_scaling_is_two_thirds_power():
    r1 = ask.surface_resistance_extreme_anomalous(2.0e9, VF, WP)
    r2 = ask.surface_resistance_extreme_anomalous(4.0e9, VF, WP)
    assert r2 / r1 == pytest.approx(2.0 ** (2.0 / 3.0), rel=1e-12)


def test_fermi_velocity_scaling_is_one_third_power():
    r1 = ask.surface_resistance_extreme_anomalous(5.0e9, VF, WP)
    r2 = ask.surface_resistance_extreme_anomalous(5.0e9, 8.0 * VF, WP)
    assert r2 / r1 == pytest.approx(2.0, rel=1e-12)


def test_plasma_frequency_scaling_is_minus_two_thirds_power():
    r1 = ask.surface_resistance_extreme_anomalous(5.0e9, VF, WP)
    r2 = ask.surface_resistance_extreme_anomalous(5.0e9, VF, 8.0 * WP)
    assert r2 / r1 == pytest.approx(8.0 ** (-2.0 / 3.0), rel=1e-12)


@pytest.mark.parametrize(
    "args",
    [(0.0, VF, WP), (5e9, 0.0, WP), (5e9, VF, -1.0)],
)
def test_invalid_inputs_raise(args):
    with pytest.raises(ValueError):
        ask.surface_resistance_extreme_anomalous(*args)
