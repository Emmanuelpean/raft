""" constants module """


# ----------------------------------------------------- QUANTITIES -----------------------------------------------------

intensity_qt = 'intensity'
wavelength_qt = 'wavelength'
energy_qt = 'energy'
frequency_qt = 'frequency'
absorbance_qt = 'absorbance'
transmittance_qt = 'transmittance'
reflectance_qt = 'reflectance'
unknown_qt = 'unknown quantity'
time_qt = 'time'
temperature_qt = 'temperature'
humidity_qt = 'humidity'
exc_wavelength_qt = 'excitation wavelength'
em_wavelength_qt = 'emission wavelength'
area_qt = 'area'
fwhm_qt = 'fwhm'
horizontal_distance_qt = 'horizontal distance'
vertical_distance_qt = 'vertical distance'
max_qt = 'max.'
surface_coverage_qt = 'surface coverage'
integrated_qt = 'integrated'
derivative_qt = 'delta'
threshold_qt = 'threshold'
x_0_qt = 'x_0'
wavenumber_qt = 'wavenumber'
voltage_qt = 'voltage'
current_qt = 'current'
current_density_qt = 'current density'
power_qt = 'power'
posx_qt = 'x'
posy_qt = 'y'
pixel_qt = 'pixel'
rel_abs_qt = 'relative absorbance'
repeats_qt = 'repeats'
min_qt = 'min.'
two_theta_qt = '2 theta'
theta_qt = 'theta'
rel_change_qt = 'relative change'
ratio_qt = 'ratio'
spacing_qt = 'spacing'
crystals_mean_size_qt = 'domain mean size'
norm_intensity_0_qt = 'I(t)/I(0)'
tau_qt = 'tau'
gamma_qt = 'gamma'
absorbed_qt = 'absorbed'
absorptance_qt = 'absorptance'
voc_qt = 'V_OC'
pulse_qt = 'pulse'
trpl_qt = 'I_TRPL'
trmc_qt = 'I_TRMC'
distance_qt = 'distance'
speed_qt = 'speed'
r2_qt = 'R^2'
residual_qt = 'residual'
contribution_qt = 'contribution'
concentration_qt = 'concentration'


# Labels (string used to display the quantities)
quantities_label = {x_0_qt: 'x<sub>0<\sub>',
                    fwhm_qt: 'FWHM',
                    theta_qt: 'θ',
                    two_theta_qt: '2θ',
                    max_qt + ' ' + wavelength_qt: 'λ_{max}',
                    max_qt + ' ' + intensity_qt: 'I_{max}',
                    derivative_qt: 'Δ',
                    spacing_qt: 'd',
                    rel_change_qt: 'rel. change',
                    norm_intensity_0_qt: 'I(t) / I(0)',
                    tau_qt: 'τ',
                    gamma_qt: 'γ'}

# -------------------------------------------------------- UNITS -------------------------------------------------------

# Vols, Amperes and Watts
volt_unit = 'V'
mw_unit = 'mW'
ma_unit = 'mA'
ma_cm_unit = 'mA/cm'
ma_cm2_unit = 'mA/cm^2'
watt_unit = 'W'
amps_unit = 'A'


# Length units
nm_unit = 'nm'
micrometer_unit = 'um'
mum2_unit = 'um^2'
angstrom_unit = 'angstrom'
cm_1_unit = 'cm^-1'
cm_3_unit = 'cm^-3'
meter_unit = 'm'

distance_factors = {'nm': 1e-9, 'um': 1e-6, 'mm': 1e-3, 'cm': 1e-2, 'dm': 1e-1, 'm': 1, 'km': 1e3}

# Time units
persec_unit = '/s'
day_unit = 'days'
hour_unit = 'h'
minute_unit = 'min'
second_unit = 's'
millisecond_unit = 'ms'
microsecond_unit = 'us'
nanosecond_unit = 'ns'
picosecond_unit = 'ps'
femtosecond_unit = 'fs'

# Energy units
ev_unit = 'eV'
hertz_unit = 'Hz'
joule_unit = 'J'
kelvin_unit = 'K'
onesun_unit = 'Sun'

# Other units
none_unit = ''
au_unit = 'a.u.'
unknown_unit = 'unknown'
counts_unit = 'counts'
percent_unit = '%'
celsius_unit = 'deg C'
relhum_unit = '% RH'
norm_unit = 'normalised'
perpixel_unit = '/pixel'
deg_unit = 'deg'
micromolar_unit = 'uM'
nonlinear_unit = 'non linear scale'
molar_unit = 'M'
nm_min_unit = 'nm/min'
microwatt_unit = 'uW'
microwatt_cm2_unit = 'uW/cm^2'


# Labels (string used to display the units, use | to for whitespace)
units_label = {ev_unit: 'eV',
               celsius_unit: '°C',
               relhum_unit: '%RH',
               volt_unit: 'V',
               mw_unit: 'mW',
               ma_unit: 'mA',
               norm_unit: 'norm.',
               deg_unit: 'deg',
               angstrom_unit: 'Å',
               micrometer_unit: 'μm',
               micromolar_unit: 'μM',
               microsecond_unit: 'μs',
               microwatt_unit: 'μW',
               microwatt_cm2_unit: 'μW/cm^2'}

# ------------------------------------------------------ PYDA IDS ------------------------------------------------------

posx_id = 'pos-x'
posy_id = 'pos-y'
timestamp_id = 'TimeStamp'
pyda_id = 'pyda:'
int_time_id = 'IntegrationTime'
wl_id = 'wavelength'
measure_time_id = 'measure_time'
exc_wl_id = 'excitation_wl'
x_unit_id = 'x_unit'
y_unit_id = 'y_unit'
x_quantity_id = 'x_quantity'
y_quantity_id = 'y_quantity'
filename_id = 'filename'
elapsed_time_id = 'elapsed_time'
pulse_period_id = 'pulse_period'
scan_speed_id = 'scan_speed'
slitwidth_id = 'slitwidth'
