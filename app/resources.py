"""Paths to images, CSS files and test files"""

from os import path

resources_path = path.join(path.dirname(path.dirname(__file__)), "resources")
CSS_STYLE_PATH = path.join(resources_path, "style.css")

# ------------------------------------------------------- IMAGES -------------------------------------------------------


LOGO_TEXT_PATH = path.join(resources_path, "medias/logo_text.svg")
LOGO_PATH = path.join(resources_path, "medias/logo.svg")
ICON_PATH = path.join(resources_path, "medias/icon.png")
DATA_PROCESSING_PATH = path.join(resources_path, "medias/data_processing.svg")


# ----------------------------------------------------- TEST FILES -----------------------------------------------------

# BeamPro
BEAMPRO_PATH = path.join(resources_path, "test_files/BeamPro/BeamPro.txt")

# Dektak
DEKTAK_PATH = path.join(resources_path, "test_files/Dektak/Dektak.csv")

# Diffrac
DIFFRAC_PATH = path.join(resources_path, "test_files/Diffrac/diffrac.brml")
DIFFRAC_TIMELAPSE_PATH = path.join(resources_path, "test_files/Diffrac/Diffrac_multiple.brml")
DIFFRAC_PSD_PATH = path.join(resources_path, "test_files/Diffrac/Diffrac_PSD.brml")

# EasyLog
EASYLOG_PATH = path.join(resources_path, "test_files/EasyLog/Easylog.txt")

# Edinst
F980_IRF_PATH = path.join(resources_path, "test_files/Edinst/F980_IRF.txt")
F980_IRF_COMMA_PATH = path.join(resources_path, "test_files/Edinst/F980_IRF_comma.txt")
F980_IRF_MULTIPLE_PATH = path.join(resources_path, "test_files/Edinst/F980_IRF_multiple.txt")
F980_EMSCAN_PATH = path.join(resources_path, "test_files/Edinst/F980_EmScan.txt")
FLUORACLE_ABSORPTANCE_PATH = path.join(resources_path, "test_files/Edinst/Fluoracle_absorbance.txt")
FLUORACLE_EMISSION_PATH = path.join(resources_path, "test_files/Edinst/Fluoracle_emission.txt")
FLUORACLE_MULTIPLE_EMISSION_PATH = path.join(resources_path, "test_files/Edinst/Fluoracle_emission_multiple.txt")

# FluorEssence
FLUORESSENCE_PATH = path.join(resources_path, "test_files/FluorEssence/FluorEssence.txt")
FLUORESSENCE_ALLCOL_PATH = path.join(resources_path, "test_files/FluorEssence/FluorEssence_allcol.txt")
FLUORESSENCE_MULTIPLE_PATH = path.join(resources_path, "test_files/FluorEssence/FluorEssence_multiple.txt")

# FL Winlab
FLWINLAB_PATH = path.join(resources_path, "test_files/FLWinlab/FLWinlab")

# LambdaSPX
LAMBDASPX_ABSORBANCE_PATH = path.join(resources_path, "test_files/LambdaSPX/LambdaSPX_absorbance.dsp")
LAMBDASPX_TRANSMITTANCE_PATH = path.join(resources_path, "test_files/LambdaSPX/LambdaSPX_transmittance.dsp")
LAMBDASPX_REFLECTANCE_PATH = path.join(resources_path, "test_files/LambdaSPX/LambdaSPX_reflectance.dsp")
LAMBDASPX_ABSORBANCE2_PATH = path.join(
    resources_path, "test_files/LambdaSPX/LambdaSPX_absorbance_other-date-format.dsp"
)

# Pro-Data
PRODATA_PLL_12WL_1PROP_PATH = path.join(resources_path, "test_files/Pro-Data/prodata-pll_12wl_1prop.csv")
PRODATA_TAS_3PROP_PATH = path.join(resources_path, "test_files/Pro-Data/prodata-tas_3prop.csv")

# SBTPS
SBTPS_SEQ1_PATH = path.join(resources_path, "test_files/SBTPS/SBTPS_SEQ1.SEQ")
SBTPS_SEQ2_PATH = path.join(resources_path, "test_files/SBTPS/SBTPS_SEQ2.SEQ")
SBTPS_SEQ3_PATH = path.join(resources_path, "test_files/SBTPS/SBTPS_SEQ3.SEQ")
SBTPS_IV1_PATH = path.join(resources_path, "test_files/SBTPS/SBTPS_IV1.IV")
SBTPS_IV2_PATH = path.join(resources_path, "test_files/SBTPS/SBTPS_IV2.IV")
SBTPS_IV3_PATH = path.join(resources_path, "test_files/SBTPS/SBTPS_IV3.IV")

# Simple
SIMPLE_SEMICOLON_PATH = path.join(resources_path, "test_files/Simple/Simple_semicolon.csv")
SIMPLE_TAB_PATH = path.join(resources_path, "test_files/Simple/Simple_tab.txt")

# SpectraSuite
SPECTRASUITE_HEADER_PATH = path.join(resources_path, "test_files/SpectraSuite/SpectraSuite_Header.txt")
SPECTRASUITE_HEADER_BST_PATH = path.join(resources_path, "test_files/SpectraSuite/SpectraSuite_Header_BST.txt")

# Spectrum 10
SPECTRUM_PATH = path.join(resources_path, "test_files/Spectrum/spectrum.csv")
SPECTRUM_MULTIPLE_PATH = path.join(resources_path, "test_files/Spectrum/spectrum_multiple.csv")

# UV Winlab
UVWINLAB_PATH = path.join(resources_path, "test_files/UVWinlab/UVWinlab.csv")
UVWINLAB_ASCII_PATH = path.join(resources_path, "test_files/UVWinlab/UVWinlab_ASCII.ASC")

# Vesta
VESTA_PATH = path.join(resources_path, "test_files/Vesta/Vesta.xy")

# Wire44
WIRE1_PATH = path.join(resources_path, "test_files/WiRE/WiRE_532.wdf")
WIRE2_PATH = path.join(resources_path, "test_files/WiRE/WiRE_785.wdf")

# Zem3
ZEM3_PATH = path.join(resources_path, "test_files/Zem3/Zem3")
ZEM3_TXT_PATH = path.join(resources_path, "test_files/Zem3/Zem3.txt")


FILE_TYPE_DICT = {
    BEAMPRO_PATH: "Beampro (.txt)",
    DEKTAK_PATH: "Dektak (.csv)",
    DIFFRAC_PATH: "Diffrac (.brml)",
    DIFFRAC_TIMELAPSE_PATH: "Diffrac (.brml)",
    DIFFRAC_PSD_PATH: "Diffrac (.brml)",
    EASYLOG_PATH: "EasyLog (.txt)",
    F980_IRF_PATH: "F980/Fluoracle (.txt, tab)",
    F980_EMSCAN_PATH: "F980/Fluoracle (.txt, tab)",
    F980_IRF_MULTIPLE_PATH: "F980/Fluoracle (.txt, tab)",
    F980_IRF_COMMA_PATH: "F980/Fluoracle (.txt, comma)",
    FLUORACLE_EMISSION_PATH: "F980/Fluoracle (.txt, tab)",
    FLUORACLE_ABSORPTANCE_PATH: "F980/Fluoracle (.txt, tab)",
    FLUORACLE_MULTIPLE_EMISSION_PATH: "F980/Fluoracle (.txt, tab)",
    FLUORESSENCE_PATH: "FluorEssence (.txt)",
    FLUORESSENCE_MULTIPLE_PATH: "FluorEssence (.txt)",
    FLUORESSENCE_ALLCOL_PATH: "FluorEssence (.txt)",
    FLWINLAB_PATH: "FlWinlab",
    LAMBDASPX_REFLECTANCE_PATH: "LambdaSpx (.dsp)",
    LAMBDASPX_TRANSMITTANCE_PATH: "LambdaSpx (.dsp)",
    LAMBDASPX_ABSORBANCE_PATH: "LambdaSpx (.dsp)",
    LAMBDASPX_ABSORBANCE2_PATH: "LambdaSpx (.dsp)",
    PRODATA_TAS_3PROP_PATH: "ProData (.csv)",
    PRODATA_PLL_12WL_1PROP_PATH: "ProData (.csv)",
    SBTPS_IV1_PATH: "SBTPS (.IV)",
    SBTPS_IV2_PATH: "SBTPS (.IV)",
    SBTPS_IV3_PATH: "SBTPS (.IV)",
    SBTPS_SEQ1_PATH: "SBTPS (.SEQ)",
    SBTPS_SEQ2_PATH: "SBTPS (.SEQ)",
    SBTPS_SEQ3_PATH: "SBTPS (.SEQ)",
    SIMPLE_TAB_PATH: "Simple (tab)",
    SIMPLE_SEMICOLON_PATH: "Simple (semicolon)",
    SPECTRASUITE_HEADER_PATH: "SpectraSuite (.txt)",
    SPECTRASUITE_HEADER_BST_PATH: "SpectraSuite (.txt)",
    SPECTRUM_PATH: "UvWinlab/Spectrum (.csv)",
    SPECTRUM_MULTIPLE_PATH: "UvWinlab/Spectrum (.csv)",
    UVWINLAB_PATH: "UvWinlab/Spectrum (.csv)",
    UVWINLAB_ASCII_PATH: "UVWinLab (.asc)",
    VESTA_PATH: "Vesta (.xy)",
    WIRE1_PATH: "WiRE (.wdf)",
    WIRE2_PATH: "WiRE (.wdf)",
    ZEM3_PATH: "Zem3 (tab)",
    ZEM3_TXT_PATH: "Zem3 (tab)",
}
