"""constants module"""

# ----------------------------------------------------- QUANTITIES -----------------------------------------------------

ABSORBANCE_QT = "absorbance"
ABSORPTANCE_QT = "absorptance"
AREA_QT = "area"
CURRENT_DENSITY_QT = "current density"
CURRENT_QT = "current"
DISTANCE_QT = "distance"
EM_WAVELENGTH_QT = "emission wavelength"
EXC_WAVELENGTH_QT = "excitation wavelength"
FWHM_QT = "fwhm"
HORIZONTAL_DISTANCE_QT = "horizontal distance"
HUMIDITY_QT = "humidity"
INTENSITY_QT = "intensity"
MAX_QT = "max."
MIN_QT = "min."
POWER_QT = "power"
REFLECTANCE_QT = "reflectance"
REL_ABS_QT = "relative absorbance"
REPEATS_QT = "repeats"
SPEED_QT = "speed"
TEMPERATURE_QT = "temperature"
THETA_QT = "theta"
TIME_QT = "time"
TRANSMITTANCE_QT = "transmittance"
TWO_THETA_QT = "2 theta"
VERTICAL_DISTANCE_QT = "vertical distance"
VOLTAGE_QT = "voltage"
WAVELENGTH_QT = "wavelength"
WAVENUMBER_QT = "wavenumber"

# Quantity labels
QUANTITIES_LABEL = {
    FWHM_QT: "FWHM",
    THETA_QT: "θ",
    TWO_THETA_QT: "2θ",
}

# -------------------------------------------------------- UNITS -------------------------------------------------------

# Vols, Amperes and Watts
MA_CM_2_UNIT = "mA/cm^2"
MA_UNIT = "mA"
MW_UNIT = "mW"
VOLT_UNIT = "V"

# Length units
ANGSTROM_UNIT = "angstrom"
CM_1_UNIT = "cm^-1"
MICROMETER_UNIT = "um"
NM_UNIT = "nm"

# Time units
NANOSECOND_UNIT = "ns"
SECOND_UNIT = "s"

# Other units
AU_UNIT = "a.u."
CELSIUS_UNIT = "deg C"
COUNTS_UNIT = "counts"
DEG_UNIT = "deg"
NM_MIN_UNIT = "nm/min"
PERCENT_UNIT = "%"

# Unit labels (string used to display the units, use | to for whitespace)
UNITS_LABEL = {
    ANGSTROM_UNIT: "Å",
    CELSIUS_UNIT: "°C",
    DEG_UNIT: "deg",
    MA_UNIT: "mA",
    MICROMETER_UNIT: "μm",
    MW_UNIT: "mW",
    VOLT_UNIT: "V",
}

# ------------------------------------------------------ PYDA IDS ------------------------------------------------------

EXCITATION_WAVELENGTH_ID = "Excitation Wavelength"
INTEGRATION_TIME_ID = "Integration Time"
MEASUREMENT_TIME_ID = "Measurement Time"
PYDA_ID = "pyda:"
SCAN_SPEED_ID = "Scan Speed"
TIMESTAMP_ID = "Date & Time"
WAVELENGTH_ID = "Wavelength"
