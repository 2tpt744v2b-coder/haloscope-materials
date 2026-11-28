import pytest

from haloscope_materials import reference_experiments as ref

RADIUS = ref.ADMX_CAVITY_DIAMETER_M / 2.0
LENGTH = ref.ADMX_CAVITY_LENGTH_M


def test_frequency_hand_check():
    # By hand: 2.4048 * 2.998e8 / (2 * pi * 0.2) = 0.5737 GHz
    assert ref.tm010_frequency_hz(RADIUS) == pytest.approx(0.5737e9, rel=1e-3)


def test_geometry_factor_hand_check():
    # By hand: 2.4048 * 376.73 * 1.0 / (2 * 1.2) = 377.5 ohm
    r = ref.predict_rodless_copper_q(RADIUS, LENGTH)
    assert r["geometry_factor_ohm"] == pytest.approx(377.5, rel=1e-3)


def test_q_equals_g_over_rs():
    r = ref.predict_rodless_copper_q(RADIUS, LENGTH)
    assert r["q_unloaded"] == pytest.approx(
        r["geometry_factor_ohm"] / r["surface_resistance_ohm"], rel=1e-12
    )


def test_regression_predicted_q():
    # Regression only: guards against accidental code changes.
    r = ref.predict_rodless_copper_q(RADIUS, LENGTH)
    assert r["q_unloaded"] == pytest.approx(4.869e5, rel=1e-2)


def test_KNOWN_DISCREPANCY_model_is_about_3x_optimistic():
    """KNOWN DISCREPANCY, documented on purpose.

    The idealised empty-cylinder prediction is about 3 times the measured
    unloaded Q of about 160,000. If you improve the model and this test fails,
    that may be GOOD news: update the numbers deliberately and write down why.
    """
    ratio = ref.predict_rodless_copper_q(RADIUS, LENGTH)["q_unloaded"] / ref.MEASURED_UNLOADED_Q_COLD
    assert 2.0 < ratio < 4.0


def test_prediction_not_hypersensitive_to_approximate_dimensions():
    base = ref.predict_rodless_copper_q(RADIUS, LENGTH)["q_unloaded"]
    low = ref.predict_rodless_copper_q(RADIUS * 0.95, LENGTH * 0.9)["q_unloaded"]
    high = ref.predict_rodless_copper_q(RADIUS * 1.05, LENGTH * 1.1)["q_unloaded"]
    assert abs(low / base - 1.0) < 0.10
    assert abs(high / base - 1.0) < 0.10
