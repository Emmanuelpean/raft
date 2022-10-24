""" data_files """

import datetime as dt
import numpy as np
import lxml.etree as etree
import zipfile
import os
from renishawWiRE import WDFReader


from core.signal import SignalData, Dimension
from core import resources
import core.constants as pc
import core.utils as pdp


def get_uvvis_dimension(data, mode):
    """ Get the y dimension of a uv-vis data file depending on the mode
    :param np.ndarray data: y data
    :param str mode: any string containing T, R or A"""

    if 'T' in mode:
        return Dimension(data, pc.transmittance_qt, pc.percent_unit)
    elif 'R' in mode:
        return Dimension(data, pc.reflectance_qt, pc.percent_unit)
    elif 'A' in mode:
        return Dimension(data, pc.absorbance_qt)
    else:
        return Dimension(data)


def read_datafile(filename):
    """ Read the content of a data file and return its name and its z_dict containing its location
    :param UploadedFile filename: file path """

    if isinstance(filename, str):
        with open(filename, encoding="utf8", errors='ignore') as ofile:
            content = str(ofile.read()).splitlines()
        name = filename
    else:

        content = filename.getvalue()
        try:
            content = content.decode('utf8')
        except UnicodeDecodeError:
            try:
                content = content.decode('latin-1')
            except UnicodeDecodeError:
                pass
        content = content.splitlines()
        name = filename.name.split('.')[0]

    return content, name


# ------------------------------------------------------- BEAMPRO ------------------------------------------------------


def BeamproFile(filename):
    """ Read BeamPro 3.0 files
    :param str filename: file path
    Example
    -------
    >>> a = BeamproFile(resources.beampro)
    >>> a['x'].print()
    Dimension([0.00000e+00 6.60584e+00 1.32117e+01 ... 1.35089e+04 1.35156e+04
     1.35222e+04], distance, um)
    Dimension([1.24512 1.5625  1.14746 ... 1.14746 1.00098 1.34277], intensity, a.u.)
    >>> a['y'].print()
    Dimension([0.00000e+00 6.60584e+00 1.32117e+01 ... 1.35089e+04 1.35156e+04
     1.35222e+04], distance, um)
    Dimension([0.        0.0976563 1.12305   ... 1.0498    1.26953   1.02539  ], intensity, a.u.)"""

    content, name = read_datafile(filename)
    x_quantity, x_unit = pc.distance_qt, pc.micrometer_unit
    y_quantity, y_unit = pc.intensity_qt, pc.au_unit
    data = pdp.stringcolumn_to_array(content[5:])
    signal1 = SignalData(Dimension(data[0], x_quantity, x_unit), Dimension(data[1], y_quantity, y_unit), name + ' (x)')
    signal2 = SignalData(Dimension(data[2], x_quantity, x_unit), Dimension(data[3], y_quantity, y_unit), name + ' (y)')
    return {'x': signal1, 'y': signal2}


# ------------------------------------------------------- DEKTAK -------------------------------------------------------


def DektakFile(filename):
    """ Read the content of a Dektak file
    Requirements: The two last lines are assumed blank
    :param str filename: file path
    Example
    -------
    >>> a = DektakFile(resources.dektak)
    >>> a.print()
    Dimension([0.00000e+00 5.10000e-01 1.03000e+00 ... 1.99897e+03 1.99949e+03
     2.00000e+03], horizontal distance, um)
    Dimension([-649.13 -648.21 -643.29 ...  -70.38  -77.46  -84.53], vertical distance, um)
    >>> a.z_dict[pc.timestamp_id]
    Dimension(2017-05-23 11:35:19, time)"""

    content, name = read_datafile(filename)

    # Data
    data_index = pdp.grep(content, 'Horizontal Distance,Raw Data,')[0][1]
    x_data, y_data = pdp.stringcolumn_to_array(content[data_index + 2: -2], ',')[:2]
    x = Dimension(x_data, pc.horizontal_distance_qt, pc.micrometer_unit)
    y = Dimension(y_data, pc.vertical_distance_qt, pc.micrometer_unit)

    # Timestamp
    date_str = pdp.grep(content, 'Date,', 0)
    date_format = '%m/%d/%y %H:%M:%S'
    date = dt.datetime.strptime(date_str, date_format)
    z_dict = {pc.timestamp_id: Dimension(date, pc.time_qt)}

    return SignalData(x, y, name, z_dict)


# ------------------------------------------------------ DIFFRAC -------------------------------------------------------


def read_brml_rawdata_file(brml, rawdata_file):
    """ Read the content of a RawData file from a Diffrac .brml file
    :param zipfile.ZipFile brml: zipfile.ZipFile object
    :param zipfile.ZipInfo rawdata_file: rawdata file to read """

    z_dict = {}
    root = etree.ElementTree(etree.fromstring(brml.read(rawdata_file.filename)))
    data_xml = root.findall("DataRoutes/DataRoute/Datum")
    data = pdp.stringcolumn_to_array([child.text for child in data_xml], ',')

    # Data type
    dtype = root.findall("DataRoutes/DataRoute/ScanInformation")[0].get("VisibleName")
    x_quantity = pc.two_theta_qt
    if dtype == "Coupled TwoTheta/Theta":
        x_data, theta, y_data = data[2:]
    elif dtype == "TwoTheta":
        x_data, y_data = data[2:]
    elif dtype == "Rocking":
        x_data, y_data = data[2:]
        x_quantity = pc.theta_qt
    elif dtype == "PSD fixed":
        y_data = np.transpose(data)[0][:-1]  # for unknown reason, y_data is 1 element longer than x_data
        x0 = float(root.findall("DataRoutes/DataRoute/ScanInformation/ScanAxes/ScanAxisInfo/Start")[0].text)
        x1 = float(root.findall("DataRoutes/DataRoute/ScanInformation/ScanAxes/ScanAxisInfo/Stop")[0].text)
        dx = float(root.findall("DataRoutes/DataRoute/ScanInformation/ScanAxes/ScanAxisInfo/Increment")[0].text)
        x_data = np.arange(x0, x1 + dx, dx)
    else:
        raise AssertionError("Unknown data type")

    x = Dimension(x_data, x_quantity, pc.deg_unit)
    y = Dimension(y_data, pc.intensity_qt, pc.counts_unit)

    # TimeStamp
    date_str = root.findall("TimeStampStarted")[0].text
    date_format = '%Y-%m-%dT%H:%M:%S.%f'
    date = dt.datetime.strptime(date_str[:-7], date_format)  # only keep 6 digits and remove the UTC
    z_dict[pc.timestamp_id] = Dimension(date, pc.time_qt)

    # Integration time
    inttime_str = root.findall("DataRoutes/DataRoute/ScanInformation/TimePerStep")[0].text
    z_dict[pc.int_time_id] = Dimension(float(inttime_str), pc.time_qt, pc.second_unit)

    # X-rays wavelength
    wl_el = root.findall("FixedInformation/Instrument/PrimaryTracks/TrackInfoData/MountedOptics/InfoData/Tube/WaveLengthAverage")[0]
    wl_str = wl_el.get('Value')  # wavelength in angstrom
    z_dict[pc.wl_id] = Dimension(float(wl_str), pc.wavelength_qt, pc.angstrom_unit)

    # Measure time
    meas_time_str = root.findall("TimeStampFinished")[0].text
    try:
        end_time = dt.datetime.strptime(meas_time_str[:-7], date_format)
        measure_time = (end_time - date).total_seconds()
    except ValueError:
        measure_time = 0.
    z_dict[pc.measure_time_id] = Dimension(measure_time, pc.time_qt, pc.second_unit)

    # Position
    xml_drives = root.findall("FixedInformation/Drives/InfoData")
    for xml_d in xml_drives:
        drive_name = xml_d.get('LogicName')
        unit = xml_d.find('Position').get('Unit')
        val = xml_d.find('Position').get('Value')
        z_dict[drive_name] = Dimension(float(val), drive_name, unit)

    return SignalData(x, y, rawdata_file.filename, z_dict)


def DiffracBrmlFile(filename):
    """ Class for Diffrac files (.brml)
    :param str filename: file path

    Example
    -------
    >>> a = DiffracBrmlFile(resources.diffrac_brml)[0]
    >>> a.print()
    Dimension([10.0001 10.0197 10.0394 ... 59.9705 59.9901 60.0097], 2 theta, deg)
    Dimension([191. 208. 224. ...  95.  68.  86.], intensity, counts)
    >>> b = DiffracBrmlFile(resources.diffrac_timelapse)[0]
    >>> b.print()
    Dimension([12.0001 12.0021 12.0041 ... 14.4961 14.4981 14.5001], 2 theta, deg)
    Dimension([ 50.  41.  48. ... 190. 195. 167.], intensity, counts)
    >>> c = DiffracBrmlFile(resources.diffrac_brml_psd)[0]
    >>> c.print()
    Dimension([ 8.5001      8.5118762   8.52365241 ... 10.73757886 10.74935506
     10.76113126], 2 theta, deg)
    Dimension([5.000e+00 1.000e+00 2.129e+03 ... 0.000e+00 1.000e+00 1.000e+00], intensity, counts)"""

    with zipfile.ZipFile(filename) as brml:
        datafiles = [f for f in brml.infolist() if 'RawData' in f.filename]
        filenames_nb = [float(os.path.splitext(os.path.basename(s.filename.strip('/')))[0].split('RawData')[-1]) for s in datafiles]
        datafiles_sorted = pdp.sort((datafiles, filenames_nb), 1)[0]
        signals = [read_brml_rawdata_file(brml, f) for f in datafiles_sorted]

    return signals


# ------------------------------------------------------- EASYLOG ------------------------------------------------------


def EasyLogFile(filename):
    """ Read EasyLog files
    :param str filename: file path

    Example
    -------
    >>> a = EasyLogFile(resources.easylog_file)
    >>> a['Temperature'].print()
    Dimension([0.0000e+00 1.2000e+02 2.4000e+02 ... 2.5284e+05 2.5296e+05 2.5308e+05], time, s)
    Dimension([19.  19.  18.5 ... 18.5 19.5 22.5], temperature, deg C)
    >>> a['Humidity'].print()
    Dimension([0.0000e+00 1.2000e+02 2.4000e+02 ... 2.5284e+05 2.5296e+05 2.5308e+05], time, s)
    Dimension([48.  47.5 49.  ... 57.  55.5 50.5], humidity, %)"""

    content, name = read_datafile(filename)

    # Remove the serial number from the content
    data_str = [str(f).split(',') for f in content[1:]]  # str for binary files
    data_str[0].pop(-1)

    # X dimension
    date_format = '%Y-%m-%d %H:%M:%S'
    measure_time = [dt.datetime.strptime(f[1], date_format) for f in data_str]
    x_data = np.array([(f - min(measure_time)).total_seconds() for f in measure_time])
    x = Dimension(x_data, pc.time_qt, pc.second_unit)

    # First measurement as Timestamp
    z_dict = {pc.timestamp_id: Dimension(measure_time[0], pc.time_qt)}

    # Y dimension (temperature)
    y_temp_data = np.array([float(f[2]) for f in data_str])
    y_temp = Dimension(y_temp_data, pc.temperature_qt, pc.celsius_unit)
    temperature = SignalData(x, y_temp, name + ' (temperature)', z_dict.copy())

    # Y dimension (humidity)
    y_hum_data = np.array([float(f[3]) for f in data_str])
    y_hum = Dimension(y_hum_data, pc.humidity_qt, pc.percent_unit)
    humidity = SignalData(x, y_hum, name + ' (humidity)', z_dict.copy())

    return {'Temperature': temperature, 'Humidity': humidity}


# -------------------------------------------------- F980 & FLUORACLE --------------------------------------------------


def EdinstFile(filename, delimiter='\t'):
    """ Read the content of an exported Edinbugh Instrument file (F980 or Fluoracle)
    :param str filename: file path
    :param delimiter: data delimiter

    Examples
    --------
    >>> f980_a = EdinstFile(resources.f980_irf)[0]
    >>> f980_a.print()
    Dimension([  0.          0.9765625   1.953125  ... 497.07031   498.04688
     499.02344  ], time, ns)
    Dimension([4. 1. 2. ... 5. 1. 8.], intensity, counts)

    >>> f980_b = EdinstFile(resources.f980_emscan)[0]
    >>> f980_b.print()
    Dimension([675. 676. 677. ... 818. 819. 820.], wavelength, nm)
    Dimension([    0.           22.4496574    22.899437  ... 11287.127      6189.1084
         0.       ], intensity, counts)

    >>> f980_c = EdinstFile(resources.f980_multi_irts)
    >>> f980_c[-1].print()
    Dimension([  0.        3.90625   7.8125  ... 988.28125 992.1875  996.09375], time, ns)
    Dimension([10.  5. 12. ... 11.  9. 16.], intensity, counts)

    >>> f980_d = EdinstFile(resources.f980_irts_comma, delimiter=',')
    >>> f980_d[-1].print()
    Dimension([  0.         1.953125   3.90625  ... 494.14063  496.09375  498.04688 ], time, ns)
    Dimension([26. 29. 30. ... 38. 43. 26.], intensity, counts)

    >>> fluoracle_a = EdinstFile(resources.fluoracle_emission)
    >>> fluoracle_a[0].print()
    Dimension([375. 376. 377. ... 598. 599. 600.], wavelength, nm)
    Dimension([10774.1357   10104.75      9978.43262  ...   852.291443   884.633545
       796.848145], intensity, counts)

    >>> fluoracle_f = EdinstFile(resources.fluoracle_emission_multiple)
    >>> fluoracle_f[-1].print()
    Dimension([500.  500.5 501.  ... 599.  599.5 600. ], wavelength, nm)
    Dimension([27.7835999 28.2052975 27.0729065 ... 15.5252533 10.6562662 12.8452024], intensity, counts)

    >>> fluoracle_c = EdinstFile(resources.fluoracle_absorbance)
    >>> fluoracle_c[-1].print()
    Dimension([450.  450.5 451.  ... 599.  599.5 600. ], wavelength, nm)
    Dimension([90.638423   90.793699   90.5693948  ...  1.54868774  0.58697853
      1.20407082], absorptance, %)"""

    content, name = read_datafile(filename)
    index = pdp.grep(content, 'Labels')[0][1]
    content = content[index:]

    # Remove the extra column in the case where the delimiter is not \t
    if delimiter != '\t':
        content = [line.replace(delimiter, '\t') for line in content]

    index_header = pdp.get_data_index(content)
    headers = pdp.get_header_as_dicts(content[:index_header - 1])
    x_data, *ys_data = pdp.stringcolumn_to_array(content[index_header:])

    # X dimension
    measure_type = headers[0]['Type']
    if 'Emission Scan' in measure_type or 'Excitation Scan' in measure_type:
        x_quantity, x_unit = pc.wavelength_qt, pc.nm_unit
    elif 'Time Scan' in measure_type:
        x_quantity, x_unit = pc.time_qt, pdp.grep(content, 'Time(', 0, ')')
        if x_unit is None:
            x_unit = pc.nanosecond_unit  # assume ns
    elif 'Synchronous Scan' in measure_type:
        x_quantity, x_unit = pc.wavelength_qt, pc.nm_unit
    else:
        raise AssertionError('Unknown scan type')
    x = Dimension(x_data, x_quantity, x_unit)

    # Y dimension(s)
    ys_data = np.array(ys_data)
    if 'Yaxis' in headers[0]:
        yaxis = headers[0]['Yaxis']
    else:
        yaxis = headers[0]['YAxis']
    if 'Absorbance' in yaxis:
        y_quantity, y_unit = pc.absorptance_qt, pc.percent_unit
        ys_data *= 100
    elif 'Counts' in yaxis:
        y_quantity, y_unit = pc.intensity_qt, pc.counts_unit
    else:
        raise AssertionError('Unknown y axis')

    signals = []
    for y_data, header in zip(ys_data, headers):
        y = Dimension(y_data, y_quantity, y_unit)
        signals.append(SignalData(x, y, header['Labels'], header))

    return signals


# ---------------------------------------------------- FLUORESSENCE ----------------------------------------------------


def FluorEssenceFile(filename):
    """ Read the content of a single FluorEssence exported file
    The following columns/headers are required
    - Long Name header
    - Units header
    - The first 3 rows must contain, in order, the Short Name, the Long Name and the Units. If not, the first column
      must correspond to
    :param str filename: file path

    Examples
    --------
    >>> a = FluorEssenceFile(resources.fluoressence_allcol)
    >>> a['S1c / R1c'].print()
    Dimension([700. 701. 702. ... 848. 849. 850.], wavelength, nm)
    Dimension([243.32337943 253.62609798 244.50769122 ... 165.42510121 154.60077388
     154.53307392], intensity, CPS / MicroAmps)

    >>> b = FluorEssenceFile(resources.fluoressence_file)
    >>> b['S1c / R1c'].print()
    Dimension([490. 491. 492. ... 698. 699. 700.], wavelength, nm)
    Dimension([2322.54697681 2377.29177909 2484.62736598 ... 3092.58586409 3032.20574198
     2840.89992558], intensity, CPS / MicroAmps)
    >>> b['S1'].print()
    Dimension([490. 491. 492. ... 698. 699. 700.], wavelength, nm)
    Dimension([67564. 68900. 71712. ... 13672. 13236. 12248.], intensity, CPS)

    >>> c = FluorEssenceFile(resources.fluoressence_multiple)[0]
    >>> c.print()
    Dimension([650. 651. 652. ... 848. 849. 850.], wavelength, nm)
    Dimension([ 111.93601741   96.76899294  100.82534075 ... 1940.96407674 2386.23609332
     1665.61854766], intensity, CPS / MicroAmps)"""

    content, name = read_datafile(filename)

    # 'Correct' the content if it does not have a first column
    while content[0][:10] != 'Short Name':
        new_col = ['Short Name', 'Long Name', 'Units'] + list(map(str, range(1, len(content) - 2)))
        content = [f + '\t' + g for f, g in zip(new_col, content)]

    if pdp.grep(content, 'SourceName'):

        # Header and data
        index_header = pdp.get_data_index(content)
        headers = pdp.get_header_as_dicts(content[:index_header])
        data = pdp.stringcolumn_to_array(content[index_header:])[1:]  # first column is ignored

        # X dimension
        x = Dimension(data[0], pc.wavelength_qt, headers[0]['Units'])

        # ys
        ys_data = data[1:]
        ys_headers = headers[1:]
        y_quantity = pc.intensity_qt
        ys_unit = [header['Units'] for header in ys_headers]

        # Data type
        try:  # Excitation Emission map
            float(headers[1]['Long Name'])  # check if the Long Name is a float or int
            for header in ys_headers:
                wl = float(header['Long Name'])
                header[pc.pyda_id + pc.exc_wl_id] = Dimension(wl, pc.exc_wavelength_qt, pc.nm_unit)
            names = [header['Long Name'] + ' ' + pc.nm_unit for header in ys_headers]

        except ValueError:  # Timelapse
            date_format = '%m/%d/%Y %H:%M:%S'
            for header in ys_headers:
                date = dt.datetime.strptime(header['TimeStamp'], date_format)
                header[pc.pyda_id + pc.timestamp_id] = Dimension(date, pc.time_qt)
            names = [str(header[pc.timestamp_id]) for header in ys_headers]

        signals = []
        for y_data, y_unit, name, y_dict in zip(ys_data, ys_unit, names, ys_headers):
            signals.append(SignalData(x, Dimension(y_data, y_quantity, y_unit), name, y_dict))
        return signals

    else:

        # y dimension
        data_index = pdp.get_data_index(content)
        headers = pdp.get_header_as_dicts(content[:data_index])  # header for each column
        data = pdp.stringcolumn_to_array(content[data_index:])[1:]  # first column is ignored

        ys = {}
        for header, y_data in zip(headers, data):
            if header['Long Name'] == 'Wavelength':
                ys[header['Long Name']] = Dimension(y_data, pc.wavelength_qt, header['Units'])
            else:
                ys[header['Long Name']] = Dimension(y_data, pc.intensity_qt, header['Units'])

        signals = {}
        for key in ys:
            if key != 'Wavelength':
                signals[key] = SignalData(ys['Wavelength'], ys[key], name=name + ' (%s)' % key)

        return signals


# ------------------------------------------------------ FL WINLAB -----------------------------------------------------


def FlWinlabFile(filename):
    """ Read a file generated with the FL Winlab software
    :param str filename: file path
    Example
    -------
    >>> a = FlWinlabFile(resources.flwinlab_file)
    >>> a.print()
    Dimension([350.  350.5 351.  ... 598.5 599.  599.5], wavelength, nm)
    Dimension([65.282    54.64567  44.866456 ...  5.854989  5.846911  5.84554 ], intensity, a.u.)"""

    content, name = read_datafile(filename)

    # Data
    data_index = pdp.grep(content, '#DATA')[0][1] + 1
    x_data, y_data = pdp.stringcolumn_to_array(content[data_index:-2])
    x = Dimension(x_data, pc.wavelength_qt, pc.nm_unit)
    y = Dimension(y_data, pc.intensity_qt, pc.au_unit)

    # Timestamp
    date = dt.datetime.strptime(content[3] + ' ' + content[4][:8], '%d/%m/%y %X')
    z_dict = {pc.timestamp_id: Dimension(date, pc.time_qt)}

    return SignalData(x, y, name, z_dict)


# ------------------------------------------------------ LAMBDASPX -----------------------------------------------------


def LambdaSpxFile(filename):
    """ Read the content of a LambdaSPX file
    The file must suit the following template:
    - The y data must be located after a line with '#DATA'
    - The timestamp must be located at the 17th line
    - The measurement mode 'A', '%R' or '%T' must be located at the 10th line
    :param str filename: file path
    Examples
    --------
    >>> a = LambdaSpxFile(resources.lambdaspx_absorbance)
    >>> a.print()
    Dimension([700.  700.5 701.  ... 799.  799.5 800. ], wavelength, nm)
    Dimension([0.5041709  0.50298887 0.50147861 ... 0.21925835 0.21803915 0.21983826], absorbance)
    >>> a.z_dict[pc.scan_speed_id], a.z_dict[pc.timestamp_id]
    (Dimension(480.0, speed, nm/min), Dimension(2017-02-14 11:25:23, time))

    >>> b = LambdaSpxFile(resources.lambdaspx_reflectance)
    >>> b.print()
    Dimension([700.  700.5 701.  ... 799.  799.5 800. ], wavelength, nm)
    Dimension([31.24734461 31.35104477 31.39903992 ... 60.31588614 60.29761881
     60.16479731], reflectance, %)
    >>> b.z_dict[pc.scan_speed_id], b.z_dict[pc.timestamp_id]
    (Dimension(480.0, speed, nm/min), Dimension(2017-02-14 11:33:45, time))

    >>> f = LambdaSpxFile(resources.lambdaspx_transmittance)
    >>> f.print()
    Dimension([700.  700.5 701.  ... 799.  799.5 800. ], wavelength, nm)
    Dimension([31.42957985 31.43935651 31.4599216  ... 60.38159132 60.50872058
     60.50410867], transmittance, %)
    >>> f.z_dict[pc.scan_speed_id], f.z_dict[pc.timestamp_id]
    (Dimension(480.0, speed, nm/min), Dimension(2017-02-14 11:27:32, time))

    >>> d = LambdaSpxFile(resources.lambdaspx_absorbance2)
    >>> d.print()
    Dimension([200. 201. 202. ... 798. 799. 800.], wavelength, nm)
    Dimension([0.62599342 0.64735628 0.66461102 ... 0.1690872  0.16830412 0.16843767], absorbance)
    >>> d.z_dict[pc.scan_speed_id], d.z_dict[pc.timestamp_id]
    (Dimension(960.0, speed, nm/min), Dimension(2017-06-07 11:03:00, time))"""

    content, name = read_datafile(filename)

    # Y dimension
    data_index = pdp.grep(content, '#DATA')[0][1]
    y_data, = pdp.stringcolumn_to_array(content[data_index + 1:])
    y = get_uvvis_dimension(y_data, content[9])

    # X dimension
    x_index = np.where([c == 'nm' for c in content])[0][0]  # find the index of the line 'nm'
    x_min = float(content[x_index + 1])  # lowest x
    x_max = float(content[x_index + 2])  # highest x
    nb_points = int(content[x_index + 4])  # number of points measured
    x_data = np.linspace(x_min, x_max, nb_points)
    x = Dimension(x_data, pc.wavelength_qt, pc.nm_unit)

    # Timestamp
    date_format1 = '%m/%d/%Y %I:%M:%S %p'
    date_format2 = '%d/%m/%Y %H:%M:%S'
    try:
        date = dt.datetime.strptime(content[16], date_format2)
    except ValueError:
        date = dt.datetime.strptime(content[16], date_format1)
    z_dict = {pc.timestamp_id: Dimension(date, pc.time_qt), pc.scan_speed_id: Dimension(float(content[62]), pc.speed_qt, pc.nm_min_unit)}

    return SignalData(x, y, name, z_dict)


# ------------------------------------------------------ PRO-DATA ------------------------------------------------------


def ProDataSignal(filename):
    """ Read the content of a Pro-Data file
    :param str filename: file path

    Examples
    --------
    >>> a = ProDataSignal(resources.prodata_pll_12wl_1prop)
    >>> a['dT/T'][0].print()
    Dimension([-9.98e-07 -9.94e-07 -9.90e-07 ... -4.98e-07 -4.94e-07 -4.90e-07], time, s)
    Dimension([-3.12e-16 -3.12e-16 -3.12e-16 ... -3.12e-16 -3.12e-16 -3.12e-16], intensity, counts)

    >>> c = ProDataSignal(resources.prodata_tas_3prop)
    >>> c['100 % T Baseline'][0].print()
    Dimension([-9.89e-07 -9.73e-07 -9.57e-07 ...  9.61e-07  9.77e-07  9.93e-07], time, s)
    Dimension([0.440035 0.43998  0.440085 ... 0.440225 0.44     0.44009 ], intensity, V)"""

    content, name = read_datafile(filename)
    delimiter = ','

    # Timestamp
    date_format = '%a %b %d %X %Y'
    date_str = pdp.grep(content, 'Time Stamp :', 0)
    date = dt.datetime.strptime(date_str, date_format)
    z_dict = {pc.timestamp_id: Dimension(date, pc.time_qt)}

    nb_points = pdp.grep(content, 'Time,', 0, 'time points', 'int')

    # ---------------------------------------------------- MAIN DATA ---------------------------------------------------

    data_index = pdp.grep(content, 'Time')[1][1] + 2
    data_raw = content[data_index: data_index + nb_points]
    data = pdp.stringcolumn_to_array(data_raw, delimiter)

    # X dimension
    x = Dimension(data[0], pc.time_qt, pc.second_unit)

    # Y dimension
    ys_data = data[1:]
    data_type = content[data_index - 4]

    if data_type == 'Rel_Absorbance':
        y_quantity, y_unit = pc.rel_abs_qt, pc.none_unit
    elif data_type == 'Emission':
        y_quantity, y_unit = pc.intensity_qt, pc.counts_unit
    else:
        y_quantity, y_unit = pc.unknown_qt, pc.none_unit

    # Check if any array is all zeros
    nonzero_indices = [i for i in range(len(ys_data)) if np.any(ys_data[i])]
    ys_data = ys_data[nonzero_indices]

    # Z dimension
    z_type = content[data_index - 2].strip('Time,')
    z_data = np.array(content[data_index - 1].split(delimiter)[1:-1], dtype=float)
    z_data = z_data[nonzero_indices]

    if z_type == 'Repeat':
        z_quantity, z_unit = pc.repeats_qt, pc.none_unit
    elif z_type == 'Wavelength':
        z_quantity, z_unit = pc.em_wavelength_qt, pc.nm_unit
    else:
        z_quantity, z_unit = pc.unknown_qt, pc.none_unit

    z = Dimension(z_data, z_quantity, z_unit)

    # --------------------------------------------------- PROPERTIES ---------------------------------------------------

    ot_baseline_index = pdp.grep(content, ['0T_Baseline'])
    hunt_baseline_index = pdp.grep(content, ['100T_Baseline_Volt'])
    raw_abs_volt_index = pdp.grep(content, ['Raw_Abs_Volt'])

    def get_property(index):
        """ Return the data of a 'property' of the measure """

        if index is not None:
            data_index_ = index[0][1] + 4
            data_ = pdp.stringcolumn_to_array(content[data_index_: data_index_ + nb_points], delimiter)
            return data_[1:][nonzero_indices]  # do not keep the time axis
        else:
            return None

    signals = []

    # O% baseline
    otb_data = get_property(ot_baseline_index)
    otb_signals = []

    # 100% Baseline
    huntb_data = get_property(hunt_baseline_index)
    huntb_signals = []

    # Raw absorbance
    rawabs_data = get_property(raw_abs_volt_index)
    rawabs_signals = []

    for i in range(len(z_data)):

        z_dict['z'] = Dimension(np.array([z.data[i]]), z.quantity, z.unit)
        name = z_dict['z'].get_value_label_html()
        signals.append(SignalData(x, Dimension(ys_data[i], y_quantity, y_unit), name, z_dict))

        if otb_data is not None:
            otb_signals.append(SignalData(x, Dimension(otb_data[i], pc.intensity_qt, pc.volt_unit), name, z_dict))

        if huntb_data is not None:
            huntb_signals.append(SignalData(x, Dimension(huntb_data[i], pc.intensity_qt, pc.volt_unit), name, z_dict))

        if rawabs_data is not None:
            rawabs_signals.append(SignalData(x, Dimension(rawabs_data[i], pc.absorbance_qt), name, z_dict))

    data = {'dT/T': signals, '0 % T Baseline': otb_signals, '100 % T Baseline': huntb_signals, 'Raw data': rawabs_signals}
    # noinspection PyTypeChecker
    return {key: value for key, value in data.items() if value}


# -------------------------------------------------------- SBTPS -------------------------------------------------------


def SbtpsSeqFile(filename):
    """ Read the content of a X SEQ file
    Requirements:
    - First column is V forward in V
    - Second column is I forward in mA/cm2
    - Third column is V backward in V
    - Fourth column is I backward in mA/cm2
    :param str filename: file path

    Example
    -------
    >>> a = SbtpsSeqFile(resources.sbtps_seq1)
    >>> a['Forward'][0].print()
    Dimension([-0.1   -0.09  -0.075 ...  1.07   1.085  1.1  ], voltage, V)
    Dimension([ 17.90753  17.76527  17.40059 ... -29.98907 -29.98892 -29.98761], current density, mA/cm^2)
    >>> a['Reverse'][0].print()
    Dimension([-0.1   -0.085 -0.07  ...  1.08   1.095  1.1  ], voltage, V)
    Dimension([ 19.53833  17.73079  16.84296 ... -29.98825 -29.98798 -29.9876 ], current density, mA/cm^2)

    >>> b = SbtpsSeqFile(resources.sbtps_seq2)
    >>> b['Forward'][0].print()
    Dimension([-0.2  -0.18 -0.16 ...  1.06  1.08  1.1 ], voltage, V)
    Dimension([ 0.1799218  0.1585494  0.1367925 ... -9.380523  -9.624294  -8.67218  ], current density, mA/cm^2)
    >>> b['Reverse'][0].print()
    Dimension([-0.2  -0.18 -0.16 ...  1.06  1.08  1.1 ], voltage, V)
    Dimension([ 23.55118  23.51139  23.51163 ... -20.80241 -22.13996 -22.86284], current density, mA/cm^2)

    >>> c = SbtpsSeqFile(resources.sbtps_seq3)
    >>> c['Forward'][1].print()
    Dimension([-0.1  -0.1  -0.08 ...  1.14  1.16  1.18], voltage, V)
    Dimension([ 14.01749  13.96563  13.97495 ... -15.99    -18.01354 -20.43691], current density, mA/cm^2)

    >>> c['Reverse'][0].print()
    Dimension([-0.1  -0.08 -0.06 ...  1.16  1.18   nan], voltage, V)
    Dimension([ 15.07615  13.65758  13.56461 ... -19.03928 -21.11079       nan], current density, mA/cm^2)"""

    content, name = read_datafile(filename)
    data_index = pdp.grep(content, 'IV data')[0][1] + 1
    data_header = [x for x in content[data_index].split('\t') if x != '']

    data = pdp.stringcolumn_to_array([line.strip() for line in content[data_index + 1:]])

    try:
        fw_index = data_header.index('VSource Forward')
    except ValueError:
        fw_index = -1

    try:
        bk_index = data_header.index('VSource Backward')
    except ValueError:
        bk_index = -1

    def get_data(index_start, index_end, sname):
        """ Get the data """
        sweep_indexes = np.arange(index_start + 1, index_end)
        sweeps = [data_header[i].replace(' (mA/cm\xb2)', '') for i in sweep_indexes]
        s_data = pdp.sort(data[index_start: index_end], 0)
        x = Dimension(s_data[0], pc.voltage_qt, pc.volt_unit)
        ys = [Dimension(y_data, pc.current_density_qt, pc.ma_cm2_unit) for y_data in s_data[1:]]
        return [SignalData(x, y, name + '- %s (%s)' % (sname, s)) for s, y in zip(sweeps, ys)]

    # Forward scan
    if fw_index >= 0:
        if fw_index < bk_index:  # forward data before backward data
            end_i = bk_index
        else:  # forward data after backward data or no backward data
            end_i = len(data_header)
        forward = get_data(fw_index, end_i, 'Forward')
    else:
        forward = []

    if bk_index >= 0:
        if bk_index < fw_index:  # backward data before forward data
            end_i = fw_index
        else:  # backward data after forward data
            end_i = len(data_header)
        reverse = get_data(bk_index, end_i, 'Backward')
    else:
        reverse = []

    return {'Forward': forward, 'Reverse': reverse}


def SbtpsIvFile(filename):
    """ Class for SBTPS IV or Current files
    :param str filename: file path

    Example
    -------
    >>> a = SbtpsIvFile(resources.sbtps_iv1)
    >>> a['Current density'].print()
    Dimension([-0.1   -0.085 -0.07  ...  1.08   1.095  1.1  ], voltage, V)
    Dimension([ 18.99143  17.11235  16.38189 ... -27.84484 -29.17089 -29.87343], current density, mA/cm^2)
    >>> a['Time'].print()
    Dimension([-0.1   -0.085 -0.07  ...  1.08   1.095  1.1  ], voltage, V)
    Dimension([ 9.54  9.65  9.76 ... 19.02 19.14 19.25], time, s)

    >>> b= SbtpsIvFile(resources.sbtps_iv2)
    >>> b['Current density'].print()
    Dimension([0.4895 0.4895 0.4895 ... 0.4895 0.4895 0.4895], voltage, V)
    Dimension([21.1545  19.65363 19.4841  ... 16.92228 16.92511 16.9125 ], current density, mA/cm^2)

    >>> c = SbtpsIvFile(resources.sbtps_iv3)
    >>> c['Current density'].print()
    Dimension([-0.1  -0.1  -0.08 ...  1.14  1.16  1.18], voltage, V)
    Dimension([ 1.661264e-02  1.371791e-02  1.646814e-02 ... -1.613068e+01 -1.754159e+01
     -1.808632e+01], current density, mA/cm^2)"""

    content, name = read_datafile(filename)
    data_index = pdp.grep(content, 'VSource (V)\tCurrent Density (mA/cm')[0][1]
    data = pdp.stringcolumn_to_array(content[data_index + 1:])
    x_data, y_data_cd, y_data_c, y_data_p, y_data_t = pdp.sort(data, 0)

    x = Dimension(x_data, pc.voltage_qt, pc.volt_unit)
    current_density = SignalData(x, Dimension(y_data_cd, pc.current_density_qt, pc.ma_cm2_unit), name + ' (current density)')
    current = SignalData(x, Dimension(y_data_c, pc.current_qt, pc.ma_unit), name + ' (current)')
    power = SignalData(x, Dimension(y_data_p, pc.power_qt, pc.mw_unit), name + ' (power)')
    time = SignalData(x, Dimension(y_data_t, pc.time_qt, pc.second_unit), name + ' (time)')

    return {'Current density': current_density, 'Current': current, 'Power': power, 'Time': time}


# --------------------------------------------------- GENERAL CLASSES --------------------------------------------------


def SimpleDataFile(filename, delimiter=None):
    """ Read the content of a simple file with or without a header
    :param filename: file path
    :param delimiter: data delimiter

    Examples
    --------
    >>> a = SimpleDataFile(resources.simple_tab)[0]
    >>> a.print()
    Dimension([ 340.07  340.45  340.82 ... 1028.37 1028.66 1028.94])
    Dimension([768.57 768.57 768.57 ... 818.06 809.32 819.71])
    >>> b = SimpleDataFile(resources.simple_semicolon, ';')[0]
    >>> b.print()
    Dimension([ 800.14495197  802.07594206  804.9724272  ... 1786.78433957 1792.19111183
     1797.59788408])
    Dimension([0.12535344 0.09868099 0.1913995  ... 0.08115339 0.07994678 0.07543786])"""

    content, name = read_datafile(filename)
    data_index = pdp.get_data_index(content, delimiter)
    if data_index is None:
        raise AssertionError()
    data = pdp.stringcolumn_to_array(content[data_index:], delimiter)
    x_data, *ys_data = pdp.sort(data, 0)

    signals = []
    for y_data in ys_data:
        signals.append(SignalData(Dimension(x_data), Dimension(y_data), name))
    return signals


# ---------------------------------------------------- SPECTRASUITE ----------------------------------------------------


def SpectraSuiteFile(filename):
    """ Read the content of a SpectraSuite file with header
    The file must follow the following template:
    - Data (2 columns) start at line 18 and end one line before the end of the file
    - The timestamp of the measure is located at the third line after 'Date: '
      This timestamp can be found in the format '%a %b %d %H:%M:%S %Z %Y' where %Z can be 'GMT' or 'BST'
    :param str filename: file path

    Examples
    --------
    >>> a = SpectraSuiteFile(resources.spectrasuite_header)
    >>> a.print()
    Dimension([ 340.07  340.45  340.82 ... 1028.37 1028.66 1028.94], wavelength, nm)
    Dimension([776.36 776.36 776.36 ... 833.99 829.02 829.92], intensity, counts)
    >>> a.z_dict[pc.timestamp_id], a.z_dict[pc.int_time_id]
    (Dimension(2016-11-16 18:07:25, time), Dimension(0.1, time, s))

    >>> b = SpectraSuiteFile(resources.spectrasuite_header_bst)
    >>> b.print()
    Dimension([ 340.07  340.45  340.82 ... 1028.37 1028.66 1028.94], wavelength, nm)
    Dimension([ 732.18  732.18  732.18 ...  973.99 1003.36 1089.24], intensity, counts)
    >>> b.z_dict[pc.timestamp_id], b.z_dict[pc.int_time_id]
    (Dimension(2016-10-21 10:57:53, time), Dimension(2.5, time, s))"""

    content, name = read_datafile(filename)

    # Data
    x_data, y_data = pdp.stringcolumn_to_array(content[17: -1])
    x = Dimension(x_data, pc.wavelength_qt, pc.nm_unit)
    y = Dimension(y_data, pc.intensity_qt, pc.counts_unit)

    # Timestamp
    date_str = content[2][6:]
    date_format = '%a %b %d %H:%M:%S %Z %Y'
    if 'BST' in date_str:  # some files show 'BST' instead of 'GMT'
        date_format = date_format.replace('%Z', 'BST')
    date = dt.datetime.strptime(date_str, date_format)
    if 'BST' in date_str:
        date -= dt.timedelta(seconds=3600)
    z_dict = {pc.timestamp_id: Dimension(date, pc.time_qt)}

    # Integration time
    int_time = pdp.grep(content, "Integration Time (usec):", 0, "(", 'float')
    z_dict[pc.int_time_id] = Dimension(int_time / 1e6, pc.time_qt, pc.second_unit)

    return SignalData(x, y, name, z_dict)


# ------------------------------------------------------ SPECTRUM ------------------------------------------------------


def SpectrumFile(filename):
    """ Read csv files generated from the Spectrum 10 software
    :param str filename: file path

    Examples
    --------
    >>> a = SpectrumFile(resources.spectrum_file)
    >>> a.print()
    Dimension([ 650.   650.5  651.  ... 3999.  3999.5 4000. ], wavenumber, cm^-1)
    Dimension([80.18 80.25 80.27 ... 96.38 96.37 96.36], transmittance, %)
    >>> a.z_dict[pc.timestamp_id]
    Dimension(2018-02-16 00:00:00, time)

    >>> b = SpectrumFile(resources.spectrum_multiple)
    >>> b[0].print()
    Dimension([ 450.  451.  452. ... 1048. 1049. 1050.], wavenumber, cm^-1)
    Dimension([0.0168 0.0289 0.0151 ... 0.116  0.117  0.118 ])"""

    content, name = read_datafile(filename)
    index = pdp.get_data_index(content, ',')
    data = pdp.stringcolumn_to_array(content[index:], ',')
    x_data, *ys_data = pdp.sort(data, 0)

    if index == 2:
        x_q, y_q = content[1].split(',')

        # X dimension
        if x_q == 'cm-1':
            x = Dimension(x_data, pc.wavenumber_qt, pc.cm_1_unit)
        else:
            x = Dimension(x_data, x_q)

        # Y dimension
        if y_q == '%T':
            y = Dimension(ys_data[0], pc.transmittance_qt, pc.percent_unit)
        elif y_q == 'A':
            y = Dimension(ys_data[0], pc.absorbance_qt, pc.none_unit)
        else:
            y = Dimension(ys_data[0], y_q)

        # Timestamp
        date = dt.datetime.strptime(content[0].split(',')[-1], ' %B %d %Y')
        z_dict = {pc.timestamp_id: Dimension(date, pc.time_qt)}

        return SignalData(x, y, name, z_dict)

    else:

        # X dimension
        if content[5] == '"Wavenumber"':
            x = Dimension(x_data, pc.wavenumber_qt, pc.cm_1_unit)
        else:
            x = Dimension(x_data, content[5])

        return [SignalData(x, Dimension(y_data), name) for y_data in ys_data]


# ------------------------------------------------------ UV WINLAB -----------------------------------------------------


def UvWinlabFile(filename):
    """ Read a csv file generated with the UV Winlab software
    :param str filename: file path
    Example
    -------
    >>> a = UvWinlabFile(resources.uvwinlab_csv)
    >>> a.print()
    Dimension([200. 201. 202. ... 698. 699. 700.], wavelength, nm)
    Dimension([1.00000e+01 2.74454e-01 1.74635e-01 ... 7.53000e-04 4.97000e-04
     4.65000e-04], absorbance)"""

    content, name = read_datafile(filename)
    if 'Created as New Dataset,' in content[0]:
        content = content[1:]
    data = pdp.stringcolumn_to_array(content[1:], ',')
    x_data, y_data = pdp.sort(data, 0)

    # X dimension
    x = Dimension(x_data, pc.wavelength_qt, pc.nm_unit)

    # Y dimension
    y = get_uvvis_dimension(y_data, content[0].split(',')[1])

    return SignalData(x, y, name)


def UVWinLabASCII(filename):
    """ Read ASCII files from UVWinLab
    :param str filename: file path
    Example
    -------
    >>> a = UVWinLabASCII(resources.uvwinlab_ascii)
    """

    content, name = read_datafile(filename)

    # Date
    date_format2 = '%d/%m/%y %H:%M:%S'
    date = dt.datetime.strptime(content[3] + ' ' + content[4][:-3], date_format2)
    z_dict = {pc.timestamp_id: Dimension(date, pc.time_qt)}

    # Data
    data_index = pdp.grep(content, '#DATA')[0][1] + 1
    data = pdp.stringcolumn_to_array(content[data_index:])[:, ::-1]
    x = Dimension(data[0], pc.wavelength_qt, pc.nm_unit)
    y = get_uvvis_dimension(data[1], content[80])

    return SignalData(x, y, name, z_dict)


# ------------------------------------------------------- VESTA --------------------------------------------------------


def VestaDiffractionFile(filename):
    """ Read the content of an exported Vesta diffraction file
    :param str filename: file path
    Example
    -------
    >>> a = VestaDiffractionFile(resources.vesta_diffraction)
    >>> a.print()
    Dimension([  1.     1.01   1.02 ... 119.97 119.98 119.99], 2 theta, deg)
    Dimension([0.11191 0.10984 0.10782 ... 0.0693  0.07054 0.07185], intensity, a.u.)"""

    content, name = read_datafile(filename)
    x_data, y_data = pdp.stringcolumn_to_array(content)[:2]
    x = Dimension(x_data, pc.two_theta_qt, pc.deg_unit)
    y = Dimension(y_data, pc.intensity_qt, pc.au_unit)
    return SignalData(x, y, name)


# -------------------------------------------------------- WIRE --------------------------------------------------------


def WireFile(filename):
    """ Read Renishaw WiRE files
    :param str filename: file path

    Examples
    --------
    >>> a = WireFile(resources.wire_wdf1)
    >>> a.print()
    Dimension([ 794.19727  795.4375   796.6758  ... 1940.9746  1942.0098  1943.0449 ], wavenumber, cm^-1)
    Dimension([ 84046.164  84312.37   84407.18  ... 130683.18  130495.22  129160.39 ], intensity)

    >>> b = WireFile(resources.wire_wdf2)
    >>> b.print()
    Dimension([ 830.9918  832.1725  833.3541 ... 1897.2906 1898.2369 1899.1832], wavenumber, cm^-1)
    Dimension([28863.012 28688.145 28792.688 ... 21222.156 20938.07  20763.61 ], intensity)"""

    if not isinstance(filename, str):
        with open('temp_', 'wb') as ofile:
            ofile.write(filename.read())
            filename = 'temp_'

    reader = WDFReader(filename)
    x_data, y_data = reader.xdata[::-1], reader.spectra[::-1]
    if reader.xlist_unit.name == 'RamanShift':
        x = Dimension(x_data, pc.wavenumber_qt, pc.cm_1_unit)
    else:
        x = Dimension(x_data, pc.wavelength_qt, pc.nm_unit)
    y = Dimension(y_data, pc.intensity_qt)
    return SignalData(x, y, reader.title)


functions = {'SpectraSuite (.txt)': SpectraSuiteFile,
             'FluorEssence (.txt)': FluorEssenceFile,
             'EasyLog (.txt)': EasyLogFile,
             'Beampro (.txt)': BeamproFile,
             'F980/Fluoracle (.txt, tab)': EdinstFile,
             'F980/Fluoracle (.csv, comma)': lambda filename: EdinstFile(filename, delimiter=','),
             'UvWinlab (.csv)': UvWinlabFile,
             'Spectrum (.csv)': SpectrumFile,
             'Dektak (.csv)': DektakFile,
             'ProData (.csv)': ProDataSignal,
             'UVWinLab (.asc)': UVWinLabASCII,
             'FlWinlab': FlWinlabFile,
             'Diffrac (.brml)': DiffracBrmlFile,
             'Vesta (.xy)': VestaDiffractionFile,
             'LambdaSpx (.dsp)': LambdaSpxFile,
             'SBTPS (.SEQ)': SbtpsSeqFile,
             'SBTPS (.IV)': SbtpsIvFile,
             'WiRE (.wdf)': WireFile,
             'Simple (tab)': SimpleDataFile,
             'Simple (comma)': lambda filename: SimpleDataFile(filename, delimiter=','),
             'Simple (semicolon)': lambda filename: SimpleDataFile(filename, delimiter=';')}

extensions = {'SpectraSuite (.txt)': 'txt',
              'FluorEssence (.txt)': 'txt',
              'LambdaSpx (.dsp)': 'dsp',
              'UvWinlab (.csv)': 'csv',
              'UVWinLab (.asc)': 'asc',
              'FlWinlab': '',
              'Spectrum (.csv)': 'csv',
              'Diffrac (.brml)': 'brml',
              'F980/Fluoracle (.txt)': 'txt',
              'F980/Fluoracle (.csv, comma)': 'csv',
              'Vesta (.xy)': 'xy',
              'Dektak (.csv)': 'csv',
              'ProData (.csv)': 'csv',
              'EasyLog (.txt)': 'txt',
              'Beampro (.txt)': 'txt',
              'WiRE (.wdf)': 'wdf',
              'SBTPS (.SEQ)': 'SEQ',
              'SBTPS (.IV)': 'IV',
              'Simple (comma)': '',
              'Simple (semicolon)': '',
              'Simple (tab)': ''}


if __name__ == '__main__':
    import doctest
    doctest.testmod()
