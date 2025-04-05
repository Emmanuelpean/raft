"""constants module"""

# ----------------------------------------------------- QUANTITIES -----------------------------------------------------

absorbance_qt = "absorbance"
absorbed_qt = "absorbed"
absorptance_qt = "absorptance"
area_qt = "area"
concentration_qt = "concentration"
contribution_qt = "contribution"
crystals_mean_size_qt = "domain mean size"
current_density_qt = "current density"
current_qt = "current"
derivative_qt = "delta"
distance_qt = "distance"
em_wavelength_qt = "emission wavelength"
energy_qt = "energy"
exc_wavelength_qt = "excitation wavelength"
frequency_qt = "frequency"
fwhm_qt = "fwhm"
gamma_qt = "gamma"
horizontal_distance_qt = "horizontal distance"
humidity_qt = "humidity"
integrated_qt = "integrated"
intensity_qt = "intensity"
max_qt = "max."
min_qt = "min."
norm_intensity_0_qt = "I(t)/I(0)"
pixel_qt = 'pixel'
posx_qt = "x"
posy_qt = "y"
power_qt = "power"
pulse_qt = "pulse"
r2_qt = "R^2"
ratio_qt = "ratio"
reflectance_qt = "reflectance"
rel_abs_qt = "relative absorbance"
rel_change_qt = "relative change"
repeats_qt = "repeats"
residual_qt = "residual"
spacing_qt = "spacing"
speed_qt = "speed"
surface_coverage_qt = "surface coverage"
tau_qt = "tau"
temperature_qt = "temperature"
theta_qt = "theta"
threshold_qt = "threshold"
time_qt = "time"
transmittance_qt = "transmittance"
trmc_qt = "I_TRMC"
trpl_qt = "I_TRPL"
two_theta_qt = "2 theta"
unknown_qt = "unknown quantity"
vertical_distance_qt = "vertical distance"
voc_qt = "V_OC"
voltage_qt = "voltage"
wavelength_qt = "wavelength"
wavenumber_qt = "wavenumber"
x_0_qt = "x_0"

# Labels (string used to display the quantities)
quantities_label = {
    derivative_qt: "Δ",
    fwhm_qt: "FWHM",
    gamma_qt: "γ",
    norm_intensity_0_qt: "I(t) / I(0)",
    rel_change_qt: "rel. change",
    spacing_qt: "d",
    tau_qt: "τ",
    theta_qt: "θ",
    two_theta_qt: "2θ",
    x_0_qt: r"x<sub>0<\sub>",
}

# -------------------------------------------------------- UNITS -------------------------------------------------------

# Vols, Amperes and Watts
amps_unit = "A"
ma_cm2_unit = "mA/cm^2"
ma_cm_unit = "mA/cm"
ma_unit = "mA"
mw_unit = "mW"
volt_unit = "V"
watt_unit = "W"

# Length units
angstrom_unit = "angstrom"
cm_1_unit = "cm^-1"
cm_3_unit = "cm^-3"
meter_unit = "m"
micrometer_unit = "um"
mum2_unit = "um^2"
nm_unit = "nm"

# Time units
day_unit = "days"
femtosecond_unit = "fs"
hour_unit = "h"
microsecond_unit = "us"
millisecond_unit = "ms"
minute_unit = "min"
nanosecond_unit = "ns"
persec_unit = "/s"
picosecond_unit = "ps"
second_unit = "s"

# Energy units
ev_unit = "eV"
hertz_unit = "Hz"
joule_unit = "J"
kelvin_unit = "K"
onesun_unit = "Sun"

# Other units
au_unit = "a.u."
celsius_unit = "deg C"
counts_unit = "counts"
deg_unit = "deg"
micromolar_unit = "uM"
microwatt_cm2_unit = "uW/cm^2"
microwatt_unit = "uW"
molar_unit = "M"
nm_min_unit = "nm/min"
none_unit = ""
nonlinear_unit = "non linear scale"
norm_unit = "normalised"
percent_unit = "%"
perpixel_unit = "/pixel"
relhum_unit = "% RH"
unknown_unit = "unknown"

# Labels (string used to display the units, use | to for whitespace)
units_label = {
    angstrom_unit: "Å",
    celsius_unit: "°C",
    deg_unit: "deg",
    ev_unit: "eV",
    ma_unit: "mA",
    micrometer_unit: "μm",
    micromolar_unit: "μM",
    microsecond_unit: "μs",
    microwatt_cm2_unit: "μW/cm^2",
    microwatt_unit: "μW",
    mw_unit: "mW",
    norm_unit: "norm.",
    relhum_unit: "%RH",
    volt_unit: "V",
}

# ------------------------------------------------------ PYDA IDS ------------------------------------------------------

elapsed_time_id = "elapsed_time"
exc_wl_id = "excitation_wl"
filename_id = "filename"
int_time_id = "IntegrationTime"
measure_time_id = "measure_time"
posx_id = "pos-x"
posy_id = "pos-y"
pulse_period_id = "pulse_period"
pyda_id = "pyda:"
scan_speed_id = "scan_speed"
slitwidth_id = "slitwidth"
timestamp_id = "TimeStamp"
wl_id = "wavelength"
x_quantity_id = "x_quantity"
x_unit_id = "x_unit"
y_quantity_id = "y_quantity"
y_unit_id = "y_unit"
