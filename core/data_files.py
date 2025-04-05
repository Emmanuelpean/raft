"""data_files"""

import datetime as dt
import os
import zipfile
from io import BytesIO

import lxml.etree as etree
import numpy as np
from renishawWiRE import WDFReader

import core.constants as pc
from core.signal import SignalData, Dimension
from core.utils import grep, get_data_index, stringlist_to_matrix, sort_lists, get_header_as_dicts


def get_uvvis_dimension(
        data: np.ndarray,
        mode: str,
) -> Dimension:
    """Get the y dimension of a uv-vis data file depending on the mode
    :param data: data array
    :param mode: any string containing T, R or A"""

    if "T" in mode:
        return Dimension(data, pc.transmittance_qt, pc.percent_unit)
    elif "R" in mode:
        return Dimension(data, pc.reflectance_qt, pc.percent_unit)
    elif "A" in mode:
        return Dimension(data, pc.absorbance_qt)
    else:
        return Dimension(data)


def read_datafile(filename: str | BytesIO) -> tuple[list[str], str]:
    """Read the content of a data file and return its name and its z_dict containing its location
    :param filename: file path or BytesIO"""

    if isinstance(filename, str):
        with open(filename, encoding="utf8", errors="ignore") as ofile:
            content = str(ofile.read()).splitlines()
        name = os.path.basename(filename).split('.')[0]

    else:
        content = filename.getvalue()
        try:
            content = content.decode("utf8")
        except UnicodeDecodeError:
            try:
                content = content.decode("latin-1")
            except UnicodeDecodeError:
                pass
        content = content.splitlines()
        name = filename.name.split(".")[0]

    return content, name


# ------------------------------------------------------- BEAMPRO ------------------------------------------------------


def BeamproFile(filename: str) -> dict[str, SignalData]:
    """Read BeamPro 3.0 files
    :param filename: file path"""

    content, name = read_datafile(filename)
    if content[0] not in ("BeamPro 3.0 Crosshair file", "Beamage Crosshair file"):
        raise AssertionError()  # pragma: no cover
    x_quantity, x_unit = pc.distance_qt, pc.micrometer_unit
    y_quantity, y_unit = pc.intensity_qt, pc.au_unit
    data_index = get_data_index(content)
    if data_index != 5:
        raise AssertionError()  # pragma: no cover
    data = stringlist_to_matrix(content[data_index:])
    signal1 = SignalData(Dimension(data[0], x_quantity, x_unit), Dimension(data[1], y_quantity, y_unit), name, "X")
    signal2 = SignalData(Dimension(data[2], x_quantity, x_unit), Dimension(data[3], y_quantity, y_unit), name, "Y")
    return {"x": signal1, "y": signal2}


# ------------------------------------------------------- DEKTAK -------------------------------------------------------


def DektakFile(filename: str) -> SignalData | None:
    """Read the content of a Dektak file
    Requirements: The two last lines are assumed blank
    :param filename: file path"""

    content, name = read_datafile(filename)

    # Data
    data_index = grep(content, "Horizontal Distance,Raw Data,")[0][1]
    if not isinstance(data_index, int):
        raise AssertionError()  # pragma: no cover
    x_data, y_data = stringlist_to_matrix(content[data_index + 2: -2], ",")[:2]
    x = Dimension(x_data, pc.horizontal_distance_qt, pc.micrometer_unit)
    y = Dimension(y_data, pc.vertical_distance_qt, pc.micrometer_unit)

    # Timestamp
    date_str = grep(content, "Date,", 0)
    date_format = "%m/%d/%y %H:%M:%S"
    date = dt.datetime.strptime(date_str, date_format)
    z_dict = {pc.timestamp_id: Dimension(date, pc.time_qt)}

    return SignalData(x, y, name, z_dict=z_dict)


# ------------------------------------------------------ DIFFRAC -------------------------------------------------------


def read_brml_rawdata_file(
        brml: zipfile.ZipFile,
        rawdata_file: zipfile.ZipInfo,
) -> SignalData:
    """Read the content of a RawData file from a Diffrac .brml file
    :param brml: zipfile.ZipFile object
    :param rawdata_file: rawdata file to read"""

    z_dict = {}
    name = os.path.splitext(os.path.basename(brml.filename))[0]
    root = etree.ElementTree(etree.fromstring(brml.read(rawdata_file.filename)))
    data_xml = root.findall("DataRoutes/DataRoute/Datum")
    data = stringlist_to_matrix([child.text for child in data_xml], ",")

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
        raise AssertionError("Unknown data type")  # pragma: no cover

    x = Dimension(x_data, x_quantity, pc.deg_unit)
    y = Dimension(y_data, pc.intensity_qt, pc.counts_unit)

    # TimeStamp
    date_str = root.findall("TimeStampStarted")[0].text
    date_format = "%Y-%m-%dT%H:%M:%S.%f"
    date = dt.datetime.strptime(date_str[:-7], date_format)  # only keep 6 digits and remove the UTC
    z_dict[pc.timestamp_id] = Dimension(date, pc.time_qt)

    # Integration time
    inttime_str = root.findall("DataRoutes/DataRoute/ScanInformation/TimePerStep")[0].text
    z_dict[pc.int_time_id] = Dimension(float(inttime_str), pc.time_qt, pc.second_unit)

    # X-rays wavelength
    wl_el = root.findall(
        "FixedInformation/Instrument/PrimaryTracks/TrackInfoData/MountedOptics/InfoData/Tube/WaveLengthAverage"
    )[0]
    wl_str = wl_el.get("Value")  # wavelength in angstrom
    z_dict[pc.wl_id] = Dimension(float(wl_str), pc.wavelength_qt, pc.angstrom_unit)

    # Measure time
    meas_time_str = root.findall("TimeStampFinished")[0].text
    try:
        end_time = dt.datetime.strptime(meas_time_str[:-7], date_format)
        measure_time = (end_time - date).total_seconds()
    except ValueError:
        measure_time = 0.0
    z_dict[pc.measure_time_id] = Dimension(measure_time, pc.time_qt, pc.second_unit)

    # Position
    xml_drives = root.findall("FixedInformation/Drives/InfoData")
    for xml_d in xml_drives:
        drive_name = xml_d.get("LogicName")
        unit = xml_d.find("Position").get("Unit")
        val = xml_d.find("Position").get("Value")
        z_dict[drive_name] = Dimension(float(val), drive_name, unit)

    return SignalData(x, y, name, rawdata_file.filename.split("/")[1].strip(".xml"), z_dict)


def DiffracBrmlFile(filename: str) -> list[SignalData]:
    """Class for Diffrac files (.brml)
    :param filename: file path"""

    with zipfile.ZipFile(filename) as brml:
        datafiles = [f for f in brml.infolist() if "RawData" in f.filename]
        filenames_nb = [
            float(os.path.splitext(os.path.basename(s.filename.strip("/")))[0].split("RawData")[-1]) for s in datafiles
        ]
        datafiles_sorted = sort_lists((datafiles, filenames_nb), 1)[0]
        signals = [read_brml_rawdata_file(brml, f) for f in datafiles_sorted]
    return signals


# ------------------------------------------------------- EASYLOG ------------------------------------------------------


def EasyLogFile(filename: str) -> dict[str, SignalData]:
    """Read EasyLog files
    :param filename: file path"""

    content, name = read_datafile(filename)

    # Remove the serial number from the content
    data_str = [str(f).split(",") for f in content[1:]]  # str for binary files
    data_str[0].pop(-1)

    # X dimension
    date_format = "%Y-%m-%d %H:%M:%S"
    measure_time = [dt.datetime.strptime(f[1], date_format) for f in data_str]
    x_data = np.array([(f - min(measure_time)).total_seconds() for f in measure_time])
    x = Dimension(x_data, pc.time_qt, pc.second_unit)

    # First measurement as Timestamp
    z_dict = {pc.timestamp_id: Dimension(measure_time[0], pc.time_qt)}

    # Y dimension (temperature)
    y_temp_data = np.array([float(f[2]) for f in data_str])
    y_temp = Dimension(y_temp_data, pc.temperature_qt, pc.celsius_unit)
    temperature = SignalData(x, y_temp, name, "Temperature", z_dict.copy())

    # Y dimension (humidity)
    y_hum_data = np.array([float(f[3]) for f in data_str])
    y_hum = Dimension(y_hum_data, pc.humidity_qt, pc.percent_unit)
    humidity = SignalData(x, y_hum, name, "Humidity", z_dict.copy())

    return {"Temperature": temperature, "Humidity": humidity}


# -------------------------------------------------- F980 & FLUORACLE --------------------------------------------------


def EdinstFile(filename: str, delimiter: str | None = "\t", ) -> list[SignalData]:
    """Read the content of an exported Edinburgh Instrument file (F980 or Fluoracle)
    :param filename: file path
    :param delimiter: data delimiter"""

    content, name = read_datafile(filename)
    index = grep(content, "Labels")[0][1]
    content = content[index:]

    index_header = get_data_index(content, delimiter)
    if not isinstance(index_header, int):
        raise AssertionError()  # pragma: no cover
    headers = get_header_as_dicts(content[: index_header - 1], delimiter)

    if delimiter == ",":
        data = []
        for line in content[index_header:]:
            line_data = []
            for element in line.split(delimiter):
                if element.isspace() or element == "":
                    element = "nan"
                line_data.append(float(element))
            data.append(line_data)
        x_data, *ys_data, _ = np.transpose(data)
    else:
        x_data, *ys_data = stringlist_to_matrix(content[index_header:])

    # X dimension
    measure_type = headers[0]["Type"]
    if "Emission Scan" in measure_type or "Excitation Scan" in measure_type:
        x_quantity, x_unit = pc.wavelength_qt, pc.nm_unit
    elif "Time Scan" in measure_type:
        x_quantity, x_unit = pc.time_qt, grep(content, "Time(", 0, ")")
        if x_unit is None:
            x_unit = pc.nanosecond_unit  # assume ns
    elif "Synchronous Scan" in measure_type:
        x_quantity, x_unit = pc.wavelength_qt, pc.nm_unit
    else:
        raise AssertionError("Unknown scan type")  # pragma: no cover
    x = Dimension(x_data, x_quantity, x_unit)

    # Y dimension(s)
    ys_data = np.array(ys_data)
    if "Yaxis" in headers[0]:
        yaxis = headers[0]["Yaxis"]
    else:
        yaxis = headers[0]["YAxis"]
    if "Absorbance" in yaxis:
        y_quantity, y_unit = pc.absorptance_qt, pc.percent_unit
        ys_data *= 100
    elif "Counts" in yaxis:
        y_quantity, y_unit = pc.intensity_qt, pc.counts_unit
    else:
        raise AssertionError("Unknown y axis")  # pragma: no cover

    signals = []
    for y_data, header in zip(ys_data, headers):
        y = Dimension(y_data, y_quantity, y_unit)
        z_dict = {}
        for key in header:
            z_dict[key] = Dimension(header[key])
        signals.append(SignalData(x, y, name, header["Labels"], z_dict))

    return signals


# ---------------------------------------------------- FLUORESSENCE ----------------------------------------------------


def FluorEssenceFile(filename: str) -> list[SignalData] | dict[str, SignalData]:
    """Read the content of a single FluorEssence exported file
    The following columns/headers are required
    - Long Name header
    - Units header
    - The first 3 rows must contain, in order, the Short Name, the Long Name and the Units. If not, the first column
      must correspond to
    :param filename: file path"""

    content, name = read_datafile(filename)

    # 'Correct' the content if it does not have a first column
    while content[0][:10] != "Short Name":
        new_col = ["Short Name", "Long Name", "Units"] + list(map(str, range(1, len(content) - 2)))
        content = [f + r"\t" + g for f, g in zip(new_col, content)]

    if grep(content, "SourceName"):

        # Header and data
        index_header = get_data_index(content)
        if not isinstance(index_header, int):
            raise AssertionError()  # pragma: no cover
        headers = get_header_as_dicts(content[:index_header])
        data = stringlist_to_matrix(content[index_header:])[1:]  # first column is ignored

        # X dimension
        x = Dimension(data[0], pc.wavelength_qt, headers[0]["Units"])

        # ys
        ys_data = data[1:]
        ys_headers = headers[1:]
        y_quantity = pc.intensity_qt
        ys_unit = [header["Units"] for header in ys_headers]

        # Data type
        try:  # Excitation Emission map
            float(headers[1]["Long Name"])  # check if the Long Name is a float or int
            for header in ys_headers:
                wl = float(header["Long Name"])
                header[pc.pyda_id + pc.exc_wl_id] = Dimension(wl, pc.exc_wavelength_qt, pc.nm_unit)
            names = [header["Long Name"] + " " + pc.nm_unit for header in ys_headers]

        except ValueError:  # Timelapse
            date_format = "%m/%d/%Y %H:%M:%S"
            for header in ys_headers:
                date = dt.datetime.strptime(header["TimeStamp"], date_format)
                header[pc.pyda_id + pc.timestamp_id] = Dimension(date, pc.time_qt)
            names = [str(header[pc.timestamp_id]) for header in ys_headers]

        signals = []
        for y_data, y_unit, name_, y_dict in zip(ys_data, ys_unit, names, ys_headers):
            signals.append(SignalData(x, Dimension(y_data, y_quantity, y_unit), name, name_, y_dict))
        return signals

    else:

        # y dimension
        data_index = get_data_index(content)
        if not isinstance(data_index, int):
            raise AssertionError()  # pragma: no cover
        headers = get_header_as_dicts(content[:data_index])  # header for each column
        data = stringlist_to_matrix(content[data_index:])[1:]  # first column is ignored

        ys = {}
        for header, y_data in zip(headers, data):
            if header["Long Name"] == "Wavelength":
                ys[header["Long Name"]] = Dimension(y_data, pc.wavelength_qt, header["Units"])
            else:
                ys[header["Long Name"]] = Dimension(y_data, pc.intensity_qt, header["Units"])

        signals = {}
        for key in ys:
            if key != "Wavelength":
                signals[key] = SignalData(ys["Wavelength"], ys[key], name, "(%s)" % key)

        return signals


# ------------------------------------------------------ FL WINLAB -----------------------------------------------------


def FlWinlabFile(filename: str) -> SignalData:
    """Read a file generated with the FL Winlab software
    :param filename: file path"""

    content, name = read_datafile(filename)
    if not content[0][:5] == "PE FL":
        raise AssertionError()  # pragma: no cover

    # Data
    data_index = grep(content, "#DATA")[0][1]
    if not isinstance(data_index, int):
        raise AssertionError()  # pragma: no cover
    x_data, y_data = stringlist_to_matrix(content[data_index + 1:-2])
    x = Dimension(x_data, pc.wavelength_qt, pc.nm_unit)
    y = Dimension(y_data, pc.intensity_qt, pc.au_unit)

    # Timestamp
    date = dt.datetime.strptime(content[3] + " " + content[4][:8], "%d/%m/%y %X")
    z_dict = {pc.timestamp_id: Dimension(date, pc.time_qt)}

    return SignalData(x, y, name, z_dict=z_dict)


# ------------------------------------------------------ LAMBDASPX -----------------------------------------------------


def LambdaSpxFile(filename: str) -> SignalData:
    """Read the content of a LambdaSPX file
    The file must suit the following template:
    - The y data must be located after a line with '#DATA'
    - The timestamp must be located at the 17th line
    - The measurement mode 'A', '%R' or '%T' must be located at the 10th line
    :param filename: file path"""

    content, name = read_datafile(filename)

    # Y dimension
    data_index = grep(content, "#DATA")[0][1]
    if not isinstance(data_index, int):
        raise AssertionError()  # pragma: no cover
    (y_data,) = stringlist_to_matrix(content[data_index + 1:])
    y = get_uvvis_dimension(y_data, content[9])

    # X dimension
    x_index = np.where([c == "nm" for c in content])[0][0]  # find the index of the line 'nm'
    x_min = float(content[x_index + 1])  # lowest x
    x_max = float(content[x_index + 2])  # highest x
    nb_points = int(content[x_index + 4])  # number of points measured
    x_data = np.linspace(x_min, x_max, nb_points)
    x = Dimension(x_data, pc.wavelength_qt, pc.nm_unit)

    # Timestamp
    date_format1 = "%m/%d/%Y %I:%M:%S %p"
    date_format2 = "%d/%m/%Y %H:%M:%S"
    try:
        date = dt.datetime.strptime(content[16], date_format2)
    except ValueError:
        date = dt.datetime.strptime(content[16], date_format1)
    z_dict = {
        pc.timestamp_id: Dimension(date, pc.time_qt),
        pc.scan_speed_id: Dimension(float(content[62]), pc.speed_qt, pc.nm_min_unit),
    }

    return SignalData(x, y, name, z_dict=z_dict)


# ------------------------------------------------------ PRO-DATA ------------------------------------------------------


def ProDataSignal(filename: str) -> dict[str, list[SignalData]]:
    """Read the content of a Pro-Data file
    :param filename: file path"""

    content, name = read_datafile(filename)
    delimiter = ","

    # Timestamp
    date_format = "%a %b %d %X %Y"
    date_str = grep(content, "Time Stamp :", 0)
    date = dt.datetime.strptime(date_str, date_format)
    z_dict = {pc.timestamp_id: Dimension(date, pc.time_qt)}

    nb_points = grep(content, "Time,", 0, "time points", "int")

    # ---------------------------------------------------- MAIN DATA ---------------------------------------------------

    data_index = grep(content, "Time")[1][1]
    if not isinstance(data_index, int):
        raise AssertionError()  # pragma: no cover
    data_index += 2
    data_raw = content[data_index: data_index + nb_points]
    data = stringlist_to_matrix(data_raw, delimiter)

    # X dimension
    x = Dimension(data[0], pc.time_qt, pc.second_unit)

    # Y dimension
    ys_data = data[1:]
    data_type = content[data_index - 4]

    if data_type == "Rel_Absorbance":
        y_quantity, y_unit = pc.rel_abs_qt, pc.none_unit
    elif data_type == "Emission":
        y_quantity, y_unit = pc.intensity_qt, pc.counts_unit
    else:
        y_quantity, y_unit = pc.unknown_qt, pc.none_unit

    # Check if any array is all zeros
    nonzero_indices = [i for i in range(len(ys_data)) if np.any(ys_data[i])]
    ys_data = ys_data[nonzero_indices]

    # Z dimension
    z_type = content[data_index - 2].strip("Time,")
    z_data = np.array(content[data_index - 1].split(delimiter)[1:-1], dtype=float)
    z_data = z_data[nonzero_indices]

    if z_type == "Repeat":
        z_quantity, z_unit = pc.repeats_qt, pc.none_unit
    elif z_type == "Wavelength":
        z_quantity, z_unit = pc.em_wavelength_qt, pc.nm_unit
    else:
        z_quantity, z_unit = pc.unknown_qt, pc.none_unit

    z = Dimension(z_data, z_quantity, z_unit)

    # --------------------------------------------------- PROPERTIES ---------------------------------------------------

    ot_baseline_index = grep(content, ["0T_Baseline"])
    hunt_baseline_index = grep(content, ["100T_Baseline_Volt"])
    raw_abs_volt_index = grep(content, ["Raw_Abs_Volt"])

    def get_property(index) -> np.ndarray | None:
        """Return the data of a 'property' of the measure"""

        if index is not None:
            data_index_ = index[0][1] + 4
            if not isinstance(data_index_, int):
                raise AssertionError()  # pragma: no cover
            data_ = stringlist_to_matrix(content[data_index_: data_index_ + nb_points], delimiter)
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

        z_dict["z"] = Dimension(z.data[i], z.quantity, z.unit)
        shortname = z_dict["z"].get_value_label_html()
        signals.append(SignalData(x, Dimension(ys_data[i], y_quantity, y_unit), name, shortname, z_dict))

        if otb_data is not None:
            otb_signals.append(
                SignalData(x, Dimension(otb_data[i], pc.intensity_qt, pc.volt_unit), name, shortname, z_dict)
            )

        if huntb_data is not None:
            huntb_signals.append(
                SignalData(x, Dimension(huntb_data[i], pc.intensity_qt, pc.volt_unit), name, shortname, z_dict)
            )

        if rawabs_data is not None:
            rawabs_signals.append(SignalData(x, Dimension(rawabs_data[i], pc.absorbance_qt), name, shortname, z_dict))

    data = {
        "dT/T": signals,
        "0 % T Baseline": otb_signals,
        "100 % T Baseline": huntb_signals,
        "Raw data": rawabs_signals,
    }
    return {key: value for key, value in data.items() if value}


# -------------------------------------------------------- SBTPS -------------------------------------------------------


def SbtpsSeqFile(filename: str) -> dict[str, list[SignalData]]:
    """Read the content of a X SEQ file
    Requirements:
    - First column is V forward in V
    - Second column is I forward in mA/cm2
    - Third column is V backward in V
    - Fourth column is I backward in mA/cm2
    :param filename: file path"""

    content, name = read_datafile(filename)
    # noinspection PyTypeChecker
    data_index = grep(content, "IV data")[0][1] + 1
    if not isinstance(data_index, int):
        raise AssertionError()  # pragma: no cover
    data_header = [x for x in content[data_index].split("\t") if x != ""]

    data = stringlist_to_matrix([line.strip() for line in content[data_index + 1:]])

    try:
        fw_index = data_header.index("VSource Forward")
    except ValueError:
        fw_index = -1

    try:
        bk_index = data_header.index("VSource Backward")
    except ValueError:
        bk_index = -1

    def get_data(index_start: int, index_end: int, sname: str,) -> list[SignalData]:
        """Get the data"""
        sweep_indexes = np.arange(index_start + 1, index_end)
        sweeps = [data_header[i].replace(" (mA/cm\xb2)", "") for i in sweep_indexes]
        s_data = sort_lists(data[index_start:index_end], 0)
        x = Dimension(s_data[0], pc.voltage_qt, pc.volt_unit)
        ys = [Dimension(y_data, pc.current_density_qt, pc.ma_cm2_unit) for y_data in s_data[1:]]
        return [SignalData(x, y, name, "%s (%s)" % (sname, s)) for s, y in zip(sweeps, ys)]

    # Forward scan
    if fw_index >= 0:
        if fw_index < bk_index:  # forward data before backward data
            end_i = bk_index
        else:  # forward data after backward data or no backward data
            end_i = len(data_header)
        forward = get_data(fw_index, end_i, "Forward")
    else:
        forward = []

    if bk_index >= 0:
        if bk_index < fw_index:  # backward data before forward data
            end_i = fw_index
        else:  # backward data after forward data
            end_i = len(data_header)
        reverse = get_data(bk_index, end_i, "Backward")
    else:
        reverse = []

    return {"Forward": forward, "Reverse": reverse}


def SbtpsIvFile(filename: str) -> dict[str, SignalData]:
    """Class for SBTPS IV or Current files
    :param filename: file path"""

    content, name = read_datafile(filename)
    data_index = grep(content, "VSource (V)\tCurrent Density (mA/cm")[0][1]
    if not isinstance(data_index, int):
        raise AssertionError()  # pragma: no cover
    data = stringlist_to_matrix(content[data_index + 1:])
    x_data, y_data_cd, y_data_c, y_data_p, y_data_t = sort_lists(data, 0)

    x = Dimension(x_data, pc.voltage_qt, pc.volt_unit)
    current_density = SignalData(
        x, Dimension(y_data_cd, pc.current_density_qt, pc.ma_cm2_unit), name, "Current density"
    )
    current = SignalData(x, Dimension(y_data_c, pc.current_qt, pc.ma_unit), name, "Current")
    power = SignalData(x, Dimension(y_data_p, pc.power_qt, pc.mw_unit), name, "Power")
    time = SignalData(x, Dimension(y_data_t, pc.time_qt, pc.second_unit), name, "Time")

    return {"Current density": current_density, "Current": current, "Power": power, "Time": time}


# --------------------------------------------------- GENERAL CLASSES --------------------------------------------------


def SimpleDataFile(filename: str, delimiter: str | None = None) -> list[SignalData]:
    """Read the content of a simple file with or without a header
    :param filename: file path
    :param delimiter: data delimiter"""

    content, name = read_datafile(filename)
    data_index = get_data_index(content, delimiter)
    if not isinstance(data_index, int):
        raise AssertionError()  # pragma: no cover
    data = stringlist_to_matrix(content[data_index:], delimiter)
    x_data, *ys_data = sort_lists(data, 0)

    # Get the data labels
    header = content[:data_index]
    labels = [""] * len(ys_data)
    for line in header:
        c = line.split(delimiter)
        if len(np.unique(c)) > 2:
            labels = c[1:]
            break

    signals = []
    for y_data, label in zip(ys_data, labels):
        signals.append(SignalData(Dimension(x_data), Dimension(y_data), name, label))
    return signals


# ---------------------------------------------------- SPECTRASUITE ----------------------------------------------------


def SpectraSuiteFile(filename: str) -> SignalData:
    """Read the content of a SpectraSuite file with header
    The file must follow the following template:
    - Data (2 columns) start at line 18 and end one line before the end of the file
    - The timestamp of the measure is located at the third line after 'Date: '
      This timestamp can be found in the format '%a %b %d %H:%M:%S %Z %Y' where %Z can be 'GMT' or 'BST'
    :param filename: file path"""

    content, name = read_datafile(filename)

    # Data
    x_data, y_data = stringlist_to_matrix(content[17:-1])
    x = Dimension(x_data, pc.wavelength_qt, pc.nm_unit)
    y = Dimension(y_data, pc.intensity_qt, pc.counts_unit)

    # Timestamp
    date_str = content[2][6:]
    date_format = "%a %b %d %H:%M:%S %Z %Y"
    if "BST" in date_str:  # some files show 'BST' instead of 'GMT'
        date_format = date_format.replace("%Z", "BST")
    date = dt.datetime.strptime(date_str, date_format)
    if "BST" in date_str:
        date -= dt.timedelta(seconds=3600)
    z_dict = {pc.timestamp_id: Dimension(date, pc.time_qt)}

    # Integration time
    int_time = grep(content, "Integration Time (usec):", 0, "(", "float")
    z_dict[pc.int_time_id] = Dimension(int_time / 1e6, pc.time_qt, pc.second_unit)

    return SignalData(x, y, name, z_dict=z_dict)


# ------------------------------------------------------ SPECTRUM ------------------------------------------------------


def PerkinElmerFile(filename: str) -> SignalData | list[SignalData]:
    """Read csv files generated from the Spectrum 10 software
    :param filename: file path"""

    content, name = read_datafile(filename)
    if "Created as New Dataset," in content[0]:
        content = content[1:]

    # Check if there is a header
    index = get_data_index(content, ",")
    if not isinstance(index, int):
        raise AssertionError()  # pragma: no cover

    # Convert the data
    data = stringlist_to_matrix(content[index:], ",")
    x_data, *ys_data = sort_lists(data, 0)

    if index == 1:
        x_q, y_q = content[0].split(",")

        # X dimension
        if x_q == "cm-1":
            x = Dimension(x_data, pc.wavenumber_qt, pc.cm_1_unit)
        elif x_q == "nm":
            x = Dimension(x_data, pc.wavelength_qt, pc.nm_unit)
        else:
            x = Dimension(x_data, x_q)

        # Y dimension
        y = get_uvvis_dimension(ys_data[0], y_q)

        return SignalData(x, y, name)

    else:

        # X dimension
        if content[index - 1] == '"Wavenumber"':
            x = Dimension(x_data, pc.wavenumber_qt, pc.cm_1_unit)
        elif content[index - 1] == '"Wavelength"':
            x = Dimension(x_data, pc.wavelength_qt, pc.nm_unit)
        else:
            x = Dimension(x_data, content[5])

        return [SignalData(x, Dimension(y_data), name, "%i" % i) for i, y_data in enumerate(ys_data)]


def UVWinLabASCII(filename: str) -> SignalData:
    """Read ASCII files from UVWinLab
    :param filename: file path"""

    content, name = read_datafile(filename)
    if not content[0][:5] == "PE UV":
        raise AssertionError()  # pragma: no cover

    # Data
    data_index = grep(content, "#DATA")[0][1]
    if not isinstance(data_index, int):
        raise AssertionError()  # pragma: no cover
    data = stringlist_to_matrix(content[data_index + 1:])[:, ::-1]
    x = Dimension(data[0], pc.wavelength_qt, pc.nm_unit)
    y = get_uvvis_dimension(data[1], content[80])

    # Date
    date = dt.datetime.strptime(content[3] + " " + content[4][:-3], "%d/%m/%y %H:%M:%S")
    z_dict = {pc.timestamp_id: Dimension(date, pc.time_qt)}

    return SignalData(x, y, name, z_dict=z_dict)


# ------------------------------------------------------- VESTA --------------------------------------------------------


def VestaDiffractionFile(filename: str) -> SignalData:
    """Read the content of an exported Vesta diffraction file
    :param filename: file path"""

    content, name = read_datafile(filename)
    x_data, y_data = stringlist_to_matrix(content)[:2]
    x = Dimension(x_data, pc.two_theta_qt, pc.deg_unit)
    y = Dimension(y_data, pc.intensity_qt, pc.au_unit)
    return SignalData(x, y, name)


# -------------------------------------------------------- WIRE --------------------------------------------------------


def WireFile(filename: str | BytesIO) -> SignalData:
    """Read Renishaw WiRE files
    :param filename: file path"""

    if not isinstance(filename, str):
        with open("temp_", "wb") as ofile:
            ofile.write(filename.read())
            filename = "temp_"

    reader = WDFReader(filename)
    x_data, y_data = np.array(reader.xdata[::-1], dtype=float), np.array(reader.spectra[::-1], dtype=float)
    if reader.xlist_unit.name == "RamanShift":
        x = Dimension(x_data, pc.wavenumber_qt, pc.cm_1_unit)
    else:
        x = Dimension(x_data, pc.wavelength_qt, pc.nm_unit)
    y = Dimension(y_data, pc.intensity_qt)
    return SignalData(x, y, reader.title)


# -------------------------------------------------------- ZEM3 --------------------------------------------------------


def Zem3(filename: str) -> dict[str, SignalData]:
    """Read Zem3 files
    :param filename: file path"""

    content, name = read_datafile(filename)
    data_index = get_data_index(content)
    if data_index not in (2, 5):
        raise AssertionError()  # pragma: no cover
    data = stringlist_to_matrix(content[data_index:])
    headers = content[data_index - 1].split("\t")
    if data_index == 2:
        xcol = 0
    else:
        xcol = 1
    data = {
        h: SignalData(Dimension(data[xcol], headers[xcol]), Dimension(d, h), name)
        for d, h in zip(data[xcol + 1:], headers[xcol + 1:])
    }
    if not data:
        raise AssertionError()  # pragma: no cover
    else:
        return data


functions = {
    "SpectraSuite (.txt)": (SpectraSuiteFile, "txt"),
    "FluorEssence (.txt)": (FluorEssenceFile, "txt"),
    "EasyLog (.txt)": (EasyLogFile, "txt"),
    "Beampro (.txt)": (BeamproFile, "txt"),
    "F980/Fluoracle (.txt, tab)": (EdinstFile, "txt"),
    "F980/Fluoracle (.csv, comma)": (lambda filename: EdinstFile(filename, delimiter=","), "csv"),
    "UvWinlab/Spectrum (.csv)": (PerkinElmerFile, "csv"),
    "Dektak (.csv)": (DektakFile, "csv"),
    "ProData (.csv)": (ProDataSignal, "csv"),
    "UVWinLab (.asc)": (UVWinLabASCII, "asc"),
    "FlWinlab": (FlWinlabFile, ""),
    "Diffrac (.brml)": (DiffracBrmlFile, "brml"),
    "Vesta (.xy)": (VestaDiffractionFile, "xy"),
    "LambdaSpx (.dsp)": (LambdaSpxFile, "dsp"),
    "Zem3 (tab)": (Zem3, ""),
    "SBTPS (.SEQ)": (SbtpsSeqFile, "SEQ"),
    "SBTPS (.IV)": (SbtpsIvFile, "IV"),
    "WiRE (.wdf)": (WireFile, "wdf"),
    "Spectrum (.csv)": (PerkinElmerFile, "csv"),
    "Simple (tab)": (SimpleDataFile, ""),
    "Simple (comma)": (lambda filename: SimpleDataFile(filename, delimiter=","), ""),
    "Simple (semicolon)": (lambda filename: SimpleDataFile(filename, delimiter=";"), ""),
}
