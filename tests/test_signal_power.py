import pytest

from haloscope_materials import signal_power as sp

# Operating values quoted in ADMX PRL 127, 261803 (2021), Eq. (1) normalisation.
KW = dict(
    frequency_hz=1.0e9,
    b_tesla=7.5,
    volume_m3=0.136,
    form_factor=0.4,
    loaded_q=80_000.0,
    g_gamma=0.36,
)
PUBLISHED_W = 7.7e-23


def test_regression_with_5p7_microev_coefficient():
    # Regression only: guards against accidental code changes.
    assert sp.axion_signal_power_w(mass_coefficient_ev=5.7e-6, **KW) == pytest.approx(
        9.108e-23, rel=2e-3
    )


def test_reproduces_published_number_with_hypothesised_coefficient():
    # The 6.2e-6 eV coefficient was REVERSE-ENGINEERED to match. Passing this
    # test shows the unit chain is consistent; it does NOT prove the coefficient.
    p = sp.axion_signal_power_w(mass_coefficient_ev=6.2e-6, **KW)
    assert p == pytest.approx(PUBLISHED_W, rel=1e-2)


def test_KNOWN_DISCREPANCY_standard_coefficient_is_about_18_percent_high():
    p = sp.axion_signal_power_w(mass_coefficient_ev=5.7e-6, **KW)
    assert 1.10 < p / PUBLISHED_W < 1.25


@pytest.mark.parametrize(
    "name,factor,expected",
    [
        ("b_tesla", 2.0, 4.0),  # power goes as B squared
        ("volume_m3", 2.0, 2.0),
        ("form_factor", 2.0, 2.0),
        ("loaded_q", 2.0, 2.0),
        ("frequency_hz", 2.0, 2.0),  # g^2 / m_a goes as m_a, so power goes as f
    ],
)
def test_scaling_laws(name, factor, expected):
    base = sp.axion_signal_power_w(mass_coefficient_ev=5.7e-6, **KW)
    kw = dict(KW)
    kw[name] = KW[name] * factor
    scaled = sp.axion_signal_power_w(mass_coefficient_ev=5.7e-6, **kw)
    assert scaled / base == pytest.approx(expected, rel=1e-9)


def test_axion_mass_at_1ghz_is_about_4p1_microev():
    # By hand: h f / e = 6.626e-34 * 1e9 / 1.602e-19 = 4.136e-6 eV
    assert sp.axion_mass_ev(1.0e9) == pytest.approx(4.136e-6, rel=1e-3)


@pytest.mark.parametrize(
    "call",
    [
        lambda: sp.axion_mass_ev(0.0),
        lambda: sp.volume_to_natural(-1.0),
        lambda: sp.decay_constant_gev(1e-6, 0.0),
        lambda: sp.axion_signal_power_w(mass_coefficient_ev=5.7e-6, **{**KW, "loaded_q": 0.0}),
    ],
)
def test_invalid_inputs_raise(call):
    with pytest.raises(ValueError):
        call()
