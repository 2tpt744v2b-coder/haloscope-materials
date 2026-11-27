"""Smoke tests: prove the project is wired up correctly."""

import pytest

import haloscope_materials
from haloscope_materials.units import ghz_to_hz


def test_package_has_version():
    assert isinstance(haloscope_materials.__version__, str)


def test_ghz_to_hz_known_values():
    assert ghz_to_hz(1.0) == pytest.approx(1.0e9)
    assert ghz_to_hz(5.0) == pytest.approx(5.0e9)


def test_ghz_to_hz_rejects_negative():
    with pytest.raises(ValueError):
        ghz_to_hz(-1.0)
