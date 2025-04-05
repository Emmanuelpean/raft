from os import path

resources_path = path.join(path.dirname(path.dirname(__file__)), "resources")

# ------------------------------------------------------- IMAGES -------------------------------------------------------


logo_text_filename = path.join(resources_path, "medias/logo_text.svg")
logo_filename = path.join(resources_path, "medias/logo.svg")


# ----------------------------------------------------- TEST FILES -----------------------------------------------------

# BeamPro
beampro = path.join(resources_path, "test_files/BeamPro/BeamPro.txt")

# Dektak
dektak = path.join(resources_path, "test_files/Dektak/Dektak.csv")

# Diffrac
diffrac_brml = path.join(resources_path, "test_files/Diffrac/Diffrac.brml")
diffrac_timelapse = path.join(resources_path, "test_files/Diffrac/Diffrac_multiple.brml")
diffrac_brml_psd = path.join(resources_path, "test_files/Diffrac/Diffrac_PSD.brml")

# EasyLog
easylog_file = path.join(resources_path, "test_files/EasyLog/Easylog.txt")

# Edinst
f980_irf = path.join(resources_path, "test_files/Edinst/F980_IRF.txt")
f980_irts_comma = path.join(resources_path, "test_files/Edinst/F980_IRF_comma.txt")
f980_multi_irts = path.join(resources_path, "test_files/Edinst/F980_IRF_multiple.txt")
f980_emscan = path.join(resources_path, "test_files/Edinst/F980_EmScan.txt")
fluoracle_absorbance = path.join(resources_path, "test_files/Edinst/Fluoracle_absorbance.txt")
fluoracle_emission = path.join(resources_path, "test_files/Edinst/Fluoracle_emission.txt")
fluoracle_emission_multiple = path.join(resources_path, "test_files/Edinst/Fluoracle_emission_multiple.txt")
fluoracle_missing = path.join(resources_path, "test_files/Edinst/Fluoracle_missing.csv")

# FluorEssence
fluoressence_allcol = path.join(resources_path, "test_files/FluorEssence/FluorEssence_allcol.txt")
fluoressence_file = path.join(resources_path, "test_files/FluorEssence/FluorEssence.txt")
fluoressence_multiple = path.join(resources_path, "test_files/FluorEssence/FluorEssence_multiple.txt")

# FL Winlab
flwinlab_file = path.join(resources_path, "test_files/FLWinlab/FLWinlab")

# LambdaSPX
lambdaspx_absorbance = path.join(resources_path, "test_files/LambdaSPX/LambdaSPX_absorbance.dsp")
lambdaspx_transmittance = path.join(resources_path, "test_files/LambdaSPX/LambdaSPX_transmittance.dsp")
lambdaspx_reflectance = path.join(resources_path, "test_files/LambdaSPX/LambdaSPX_reflectance.dsp")
lambdaspx_absorbance2 = path.join(resources_path, "test_files/LambdaSPX/LambdaSPX_absorbance_other-date-format.dsp")

# Pro-Data
prodata_pll_12wl_1prop = path.join(resources_path, "test_files/Pro-Data/prodata-pll_12wl_1prop.csv")
prodata_tas_3prop = path.join(resources_path, "test_files/Pro-Data/prodata-tas_3prop.csv")

# SBTPS
sbtps_seq1 = path.join(resources_path, "test_files/SBTPS/SBTPS_SEQ1.SEQ")
sbtps_seq2 = path.join(resources_path, "test_files/SBTPS/SBTPS_SEQ2.SEQ")
sbtps_seq3 = path.join(resources_path, "test_files/SBTPS/SBTPS_SEQ3.SEQ")
sbtps_iv1 = path.join(resources_path, "test_files/SBTPS/SBTPS_IV1.IV")
sbtps_iv2 = path.join(resources_path, "test_files/SBTPS/SBTPS_IV2.IV")
sbtps_iv3 = path.join(resources_path, "test_files/SBTPS/SBTPS_IV3.IV")

# Simple
simple_semicolon = path.join(resources_path, "test_files/Simple/Simple_semicolon.csv")
simple_tab = path.join(resources_path, "test_files/Simple/Simple_tab.txt")

# SpectraSuite
spectrasuite_header = path.join(resources_path, "test_files/SpectraSuite/SpectraSuite_Header.txt")
spectrasuite_header_bst = path.join(resources_path, "test_files/SpectraSuite/SpectraSuite_Header_BST.txt")

# Spectrum 10
spectrum_file = path.join(resources_path, "test_files/Spectrum/spectrum.csv")
spectrum_multiple = path.join(resources_path, "test_files/Spectrum/spectrum_multiple.csv")

# UV Winlab
uvwinlab_csv = path.join(resources_path, "test_files/UVWinlab/UVWinlab.csv")
uvwinlab_ascii = path.join(resources_path, "test_files/UVWinlab/UVWinlab_ASCII.ASC")

# Vesta
vesta_diffraction = path.join(resources_path, "test_files/Vesta/Vesta.xy")

# Wire44
wire_wdf1 = path.join(resources_path, "test_files/WiRE/WiRE_532.wdf")
wire_wdf2 = path.join(resources_path, "test_files/WiRE/WiRE_785.wdf")

# Zem3
zem3 = path.join(resources_path, "test_files/Zem3/Zem3")
zem3_txt = path.join(resources_path, "test_files/Zem3/Zem3.txt")

all_files = (
    beampro,
    dektak,
    diffrac_brml,
    diffrac_timelapse,
    diffrac_brml_psd,
    easylog_file,
    f980_irf,
    f980_emscan,
    f980_irts_comma,
    f980_multi_irts,
    fluoracle_absorbance,
    fluoracle_emission,
    fluoracle_emission_multiple,
    fluoressence_file,
    fluoressence_allcol,
    fluoressence_multiple,
    flwinlab_file,
    lambdaspx_absorbance,
    lambdaspx_reflectance,
    lambdaspx_absorbance2,
    lambdaspx_transmittance,
    prodata_tas_3prop,
    prodata_pll_12wl_1prop,
    sbtps_iv1,
    sbtps_iv2,
    sbtps_iv3,
    sbtps_seq1,
    sbtps_seq2,
    sbtps_seq3,
    simple_tab,
    simple_semicolon,
    spectrasuite_header,
    spectrasuite_header_bst,
    spectrum_file,
    spectrum_multiple,
    uvwinlab_csv,
    uvwinlab_ascii,
    vesta_diffraction,
    wire_wdf2,
    wire_wdf1,
    zem3,
    zem3_txt,
)
