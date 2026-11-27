import pytest

from haloscope_materials.units import ev_to_angular_frequency, ghz_to_hz


def test_ghz_to_hz():
    assert ghz_to_hz(5.0) == pytest.approx(5.0e9)


def test_ev_to_angular_frequency_hand_check():
    # By hand: 10.83 eV * 1.602177e-19 J/eV / 1.054572e-34 J s = 1.6454e16 rad/s
    assert ev_to_angular_frequency(10.83) == pytest.approx(1.6454e16, rel=1e-4)


def test_ev_to_angular_frequency_is_linear():
    assert ev_to_angular_frequency(2.0) == pytest.approx(2 * ev_to_angular_frequency(1.0))


def test_invalid_energy_raises():
    with pytest.raises(ValueError):
        ev_to_angular_frequency(0.0)
