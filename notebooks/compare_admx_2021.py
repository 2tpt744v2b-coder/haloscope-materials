"""Print our idealised prediction next to the published ADMX numbers."""

from haloscope_materials import reference_experiments as ref

radius = ref.ADMX_CAVITY_DIAMETER_M / 2.0
result = ref.predict_rodless_copper_q(radius, ref.ADMX_CAVITY_LENGTH_M)

print("OUR MODEL: empty copper cylinder, extreme cold limit")
print(f"  TM010 frequency      : {result['frequency_hz'] / 1e9:.4f} GHz")
print(f"  geometry factor G    : {result['geometry_factor_ohm']:.1f} ohm")
print(f"  surface resistance Rs: {result['surface_resistance_ohm'] * 1e3:.3f} milliohm")
print(f"  predicted unloaded Q : {result['q_unloaded']:,.0f}")
print()
print("PUBLISHED (ADMX, with tuning rods, about 1 GHz, cold)")
print(f"  measured unloaded Q  : about {ref.MEASURED_UNLOADED_Q_COLD:,.0f}  (arXiv:2312.16668, arXiv:2208.11799)")
print(f"  typical loaded Q     : {ref.MEASURED_LOADED_Q_RANGE[0]:,.0f} to {ref.MEASURED_LOADED_Q_RANGE[1]:,.0f}  (arXiv:2010.00169)")
print()
ratio = result["q_unloaded"] / ref.MEASURED_UNLOADED_Q_COLD
print(f"OUR PREDICTION IS ABOUT {ratio:.1f} TIMES THE MEASURED UNLOADED Q.")
print("Known reasons (UNVERIFIED): tuning rods, different frequency, plated copper,")
print("surface roughness and joints, idealised cold-limit formula.")
