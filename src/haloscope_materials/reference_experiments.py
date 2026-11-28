"""Reference experiment for validation: ADMX copper cavity (Step 4).

Every number below has a source. Values marked APPROXIMATE are quoted by the
source as approximate. Nothing here is a fitted or invented value.

Units (SI): lengths in metres, frequency in Hz, resistance in ohms, Q dimensionless.

SOURCES
    [A] Khatiwada et al., "Axion Dark Matter eXperiment: Detailed Design and
        Operations", arXiv:2010.00169, Sec. II.B (Cavity).
        Cavity: copper-plated stainless steel cylinder, APPROXIMATELY 1 m long
        and 0.4 m diameter; two copper tuning rods (these lower Q); loaded Q
        typically 40,000 to 80,000 in Runs 1A and 1B.
    [B] ADMX Run 1A Analysis Details, arXiv:2312.16668.
        Unloaded Q of the TM010-like mode: 20,000 at room temperature up to
        160,000 at cryogenic temperatures.
    [C] arXiv:2208.11799: Q0 usually around 160,000 for a 1 GHz annealed
        copper-plated cavity.
    [D] WebElements (winter.group.shef.ac.uk/webelements/copper), citing
        Straumanis and Yu, Acta Cryst. 1969, 25A, 676: copper lattice constant.

STATUS: UNVERIFIED until you read the sources yourself and an expert agrees.
"""

import math

from haloscope_materials import anomalous_skin as ask
from haloscope_materials import free_electron as fe
from haloscope_materials import q_factor as qf

CU_LATTICE_CONSTANT_M = 361.49e-12  # [D]

ADMX_CAVITY_DIAMETER_M = 0.4  # [A] APPROXIMATE
ADMX_CAVITY_LENGTH_M = 1.0  # [A] APPROXIMATE

MEASURED_UNLOADED_Q_COLD = 160_000.0  # [B], [C]  (no uncertainty given in the text I read)
MEASURED_LOADED_Q_RANGE = (40_000.0, 80_000.0)  # [A]


def tm010_frequency_hz(radius_m: float) -> float:
    """TM010 resonant frequency (Hz) of an empty cylinder: f = x01 * c / (2 * pi * a)."""
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    return qf.X01 * qf.SPEED_OF_LIGHT / (2.0 * math.pi * radius_m)


def predict_rodless_copper_q(
    radius_m: float,
    length_m: float,
    lattice_constant_m: float = CU_LATTICE_CONSTANT_M,
) -> dict:
    """Idealised Q of an EMPTY copper cylinder in the extreme cold limit.

    Chain: lattice constant -> electron density -> v_F and omega_p ->
    extreme-limit R_s -> Q = G / R_s.

    Ignores tuning rods, ports, roughness, plating imperfections, and the
    crossover between classical and extreme regimes. Expect it to be optimistic.
    """
    f_hz = tm010_frequency_hz(radius_m)
    n = fe.fcc_electron_number_density(lattice_constant_m)
    v_f = fe.free_electron_fermi_velocity(n)
    omega_p = fe.free_electron_plasma_frequency(n)
    r_s = ask.surface_resistance_extreme_anomalous(f_hz, v_f, omega_p)
    g = qf.tm010_geometry_factor(radius_m, length_m)
    return {
        "frequency_hz": f_hz,
        "geometry_factor_ohm": g,
        "surface_resistance_ohm": r_s,
        "q_unloaded": qf.q_wall(g, r_s),
    }
