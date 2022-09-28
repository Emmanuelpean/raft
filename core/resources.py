from os import path

basepath = path.dirname(__file__)


# Images
logo_text_filename = path.join(basepath, '../resources/medias/logo_text.svg')
logo_filename = path.join(basepath, '../resources/medias/logo.svg')


# ----------------------------------------------------- TEST FILES -----------------------------------------------------

# FluorEssence
fluoressence_single_meas = path.join(basepath, '../resources/test_files/FluorEssence/Single_Measure.txt')
fluoressence_single_meas_exp = path.join(basepath, '../resources/test_files/FluorEssence/Single_Measure_exp.txt')
fluoressence_single_meas_s1 = path.join(basepath, '../resources/test_files/FluorEssence/Singlet_Measure_S1.txt')
fluoressence_timelapse = path.join(basepath, '../resources/test_files/FluorEssence/Timelapse.txt')

# LambdaSPX
lambdaspx_absorbance = path.join(basepath, '../resources/test_files/LambdaSPX/Absorbance.dsp')
lambdaspx_transmittance = path.join(basepath, '../resources/test_files/LambdaSPX/Transmittance.dsp')
lambdaspx_reflectance = path.join(basepath, '../resources/test_files/LambdaSPX/Reflectance.dsp')
lambdaspx_absorbance_fr = path.join(basepath, '../resources/test_files/LambdaSPX/Absorbance_other-date-format.dsp')

# SpectraSuite
spectrasuite_noheader = path.join(basepath, '../resources/test_files/SpectraSuite/SpectraSuite.txt')
spectrasuite_header = path.join(basepath, '../resources/test_files/SpectraSuite/SpectraSuite_Header.txt')
spectrasuite_header_bst = path.join(basepath, '../resources/test_files/SpectraSuite/SpectraSuite_Header_BST.txt')

# EasyLog
easylog_file = path.join(basepath, '../resources/test_files/EasyLog/Easylog.txt')

# Pro-Data
prodata_pll_12wl_1prop = path.join(basepath, '../resources/test_files/Pro-Data/prodata-pll_12wl_1prop.csv')
prodata_tas_3prop = path.join(basepath, '../resources/test_files/Pro-Data/prodata-tas_3prop.csv')

# CheckTr
checktr_measure = path.join(basepath, '../resources/test_files/CheckTr/Timelapse/MAPI_3__S1_1_.txt')
checktr_reference = path.join(basepath, '../resources/test_files/CheckTr/reference.txt')
checktr_timelapse = path.join(basepath, '../resources/test_files/CheckTr/Timelapse')

# Dektak
dektak_file = path.join(basepath, '../resources/test_files/Dektak/Dektak_file.csv')

# Wire44
wire44_pl_file = path.join(basepath, '../resources/test_files/Wire/PL.txt')
wire44_raman_file = path.join(basepath, '../resources/test_files/Wire/Raman.txt')
wire_wdf = path.join(basepath, '../resources/test_files/Wire/532.wdf')

# SBTPS
sbtps_seq_file = path.join(basepath, '../resources/test_files/SBTPS/SEQ.SEQ')
sbtps_iv_file = path.join(basepath, '../resources/test_files/SBTPS/IV.IV')
sbtps_current_file = path.join(basepath, '../resources/test_files/SBTPS/Current.IV')
sbtps_seq_file2 = path.join(basepath, '../resources/test_files/SBTPS/SEQ2.SEQ')
sbtps_seq_file3 = path.join(basepath, '../resources/test_files/SBTPS/T3-P0 2022-08-03 155906.SEQ')
sbtps_iv_file3 = path.join(basepath, '../resources/test_files/SBTPS/T3-P0-000-2022-03-08 155839.IV')

# Diffrac
diffrac_brml = path.join(basepath, '../resources/test_files/Diffrac/diffrac.brml')
diffrac_timelapse = path.join(basepath, '../resources/test_files/Diffrac/Timelapse')
diffrac_brml_timelapse = path.join(basepath, '../resources/test_files/Diffrac/TL.brml')
diffrac_brml_2d_timelapse = path.join(basepath, '../resources/test_files/Diffrac/4D_XRD_TL.brml')
diffrac_brml_psd = path.join(basepath, '../resources/test_files/Diffrac/PSD.brml')

# UV Winlab
uvwinlab_csv = path.join(basepath, '../resources/test_files/UVWinlab/UVWinlab.csv')
uvwinlab_csv_3d = path.join(basepath, '../resources/test_files/UVWinlab/3D')
uvwinlab_ascii = path.join(basepath, '../resources/test_files/UVWinlab/UVWinlab_ASCII.ASC')

# FL Winlab
flwinlab_csv = path.join(basepath, '../resources/test_files/FLWinlab/spectrum')

# Spectrum 10
spectrum_file = path.join(basepath, '../resources/test_files/Spectrum/spectrum.csv')
spectrum_timelapse = path.join(basepath, '../resources/test_files/Spectrum/Timelapse.csv')

# Vesta
vesta_diffraction = path.join(basepath, '../resources/test_files/Vesta/Diffraction_patter.xy')

# Csv
csv_ftir = path.join(basepath, '../resources/test_files/Csv/FTIR.csv')
csv_diffraction = path.join(basepath, '../resources/test_files/Csv/Diffraction.csv')

# F980
f980_irts = path.join(basepath, '../resources/test_files/F980/IRTS.txt')
f980_irts_comma = path.join(basepath, '../resources/test_files/F980/IRTS_comma.txt')
f980_emscan = path.join(basepath, '../resources/test_files/F980/EmScan.txt')
f980_multi_irts = path.join(basepath, '../resources/test_files/F980/Multi-IRTS.txt')

# Fluoracle
fluoracle_emission = path.join(basepath, '../resources/test_files/Fluoracle/Emission.txt')
fluoracle_emission_tl = path.join(basepath, '../resources/test_files/Fluoracle/Emission_timelapse.txt')
fluoracle_absorbance = path.join(basepath, '../resources/test_files/Fluoracle/Absorbance.txt')

# BeamPro
beampro = path.join(basepath, '../resources/test_files/BeamPro/laserprofile_sampledistance.txt')
