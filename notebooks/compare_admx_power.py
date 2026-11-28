"""Compare our first-principles signal power with the published normalisation."""

from haloscope_materials import signal_power as sp

kw = dict(frequency_hz=1.0e9, b_tesla=7.5, volume_m3=0.136, form_factor=0.4,
          loaded_q=80_000.0, g_gamma=0.36)
published = 7.7e-23
print(f"Published (ADMX PRL 127, 261803, Eq. 1): {published:.3e} W")
for coef in (5.7e-6, 6.2e-6):
    p = sp.axion_signal_power_w(mass_coefficient_ev=coef, **kw)
    print(f"  coefficient {coef * 1e6:.1f} microeV -> {p:.3e} W  "
          f"({(p / published - 1) * 100:+.1f} percent)")
print("6.2 microeV was reverse-engineered, not sourced: UNVERIFIED hypothesis.")
