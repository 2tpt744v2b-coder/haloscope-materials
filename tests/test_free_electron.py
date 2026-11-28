import pytest

from haloscope_materials import free_electron as fe
from haloscope_materials.units import ev_to_angular_frequency

# Copper lattice constant as listed on WebElements (cites Straumanis & Yu,
# Acta Cryst. 1969, 25A, 676). Input is sourced; derived values are model output.
A_CU = 361.49e-12  # m


def test_number_density_hand_check():
    # By hand: a^3 = 4.7237e-29 m^3 ; n = 4 / a^3 = 8.468e28 m^-3
    assert fe.fcc_electron_number_density(A_CU) == pytest.approx(8.468e28, rel=1e-3)


def test_fermi_velocity_hand_check():
    n = fe.fcc_electron_number_density(A_CU)
    assert fe.free_electron_fermi_velocity(n) == pytest.approx(1.573e6, rel=1e-3)


def test_plasma_frequency_hand_check():
    n = fe.fcc_electron_number_density(A_CU)
    assert fe.free_electron_plasma_frequency(n) == pytest.approx(1.6416e16, rel=1e-3)


def test_matches_values_in_nanocage_paper_table():
    # Consistency check only: Shao et al., Nanomaterials 14(19):1603 (2024),
    # Table 1 lists v_F = 1.57e6 m/s and hbar*omega_p = 10.83 eV for Cu.
    n = fe.fcc_electron_number_density(A_CU)
    assert fe.free_electron_fermi_velocity(n) == pytest.approx(1.57e6, rel=5e-3)
    assert fe.free_electron_plasma_frequency(n) == pytest.approx(
        ev_to_angular_frequency(10.83), rel=5e-3
    )


def test_scaling_laws():
    n = 1.0e28
    assert fe.free_electron_fermi_velocity(8 * n) / fe.free_electron_fermi_velocity(n) == pytest.approx(2.0)
    assert fe.free_electron_plasma_frequency(4 * n) / fe.free_electron_plasma_frequency(n) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: fe.fcc_electron_number_density(0.0),
        lambda: fe.fcc_electron_number_density(1e-10, electrons_per_atom=0.0),
        lambda: fe.free_electron_fermi_velocity(-1.0),
        lambda: fe.free_electron_plasma_frequency(0.0),
    ],
)
def test_invalid_inputs_raise(call):
    with pytest.raises(ValueError):
        call()
