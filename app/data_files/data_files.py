"""Module containing functions to read different types of data files"""

import datetime as dt
import os
import zipfile
from io import BytesIO

from lxml import etree
import numpy as np
from renishawWiRE import WDFReader

from config import constants
from data_files.data_extraction import stringlist_to_matrix, get_header_as_dicts, grep, get_data_index
from data_files.signal_data import SignalData, Dimension
from utils.miscellaneous import sort_lists


def get_uvvis_dimension(
    data: np.ndarray,
    mode: str,
) -> Dimension:
    """Get the y dimension of a uv-vis data file depending on the mode
    :param data: data array
    :param mode: any string containing T, R or A"""

    if "T" in mode:
        return Dimension(data, constants.TRANSMITTANCE_QT, constants.PERCENT_UNIT)
    elif "R" in mode:
        return Dimension(data, constants.REFLECTANCE_QT, constants.PERCENT_UNIT)
    elif "A" in mode:
        return Dimension(data, constants.ABSORBANCE_QT)
    else:
        return Dimension(data)


def read_datafile(filename: str | BytesIO) -> tuple[list[str], str]:
    """Read the content of a data file and return its filename and its z_dict containing its location
    :param filename: file path or BytesIO"""

    if isinstance(filename, str):
        with open(filename, encoding="utf8", errors="ignore") as ofile:
            content = str(ofile.read()).splitlines()
        name = os.path.basename(filename).split(".")[0]

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
    x_quantity, x_unit = constants.DISTANCE_QT, constants.MICROMETER_UNIT
    y_quantity, y_unit = constants.INTENSITY_QT, constants.AU_UNIT
    data_index = get_data_index(content, delimiter="\t")
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
    x_data, y_data = stringlist_to_matrix(content[data_index + 2 : -2], ",")[:2]
    x = Dimension(x_data, constants.HORIZONTAL_DISTANCE_QT, constants.MICROMETER_UNIT)
    y = Dimension(y_data, constants.VERTICAL_DISTANCE_QT, constants.MICROMETER_UNIT)

    # Timestamp
    date_str = grep(content, "Date,", 0)
    date_format = "%m/%d/%y %H:%M:%S"
    date = dt.datetime.strptime(date_str, date_format)
    z_dict = {constants.TIMESTAMP_ID: Dimension(date, constants.TIME_QT)}

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
    x_quantity = constants.TWO_THETA_QT
    if dtype == "Coupled TwoTheta/Theta":
        x_data, theta, y_data = data[2:]
    elif dtype == "TwoTheta":
        x_data, y_data = data[2:]
    elif dtype == "Rocking":
        x_data, y_data = data[2:]
        x_quantity = constants.THETA_QT
    elif dtype == "PSD fixed":
        y_data = np.transpose(data)[0][:-1]  # for unknown reason, y_data is 1 element longer than x_data
        x0 = float(root.findall("DataRoutes/DataRoute/ScanInformation/ScanAxes/ScanAxisInfo/Start")[0].text)
        x1 = float(root.findall("DataRoutes/DataRoute/ScanInformation/ScanAxes/ScanAxisInfo/Stop")[0].text)
        dx = float(root.findall("DataRoutes/DataRoute/ScanInformation/ScanAxes/ScanAxisInfo/Increment")[0].text)
        x_data = np.arange(x0, x1 + dx, dx)
    else:
        raise AssertionError("Unknown data type")  # pragma: no cover

    x = Dimension(x_data, x_quantity, constants.DEG_UNIT)
    y = Dimension(y_data, constants.INTENSITY_QT, constants.COUNTS_UNIT)

    # TimeStamp
    date_str = root.findall("TimeStampStarted")[0].text
    date_format = "%Y-%m-%dT%H:%M:%S.%f"
    date = dt.datetime.strptime(date_str[:-7], date_format)  # only keep 6 digits and remove the UTC
    z_dict[constants.TIMESTAMP_ID] = Dimension(date, constants.TIME_QT)

    # Integration time
    inttime_str = root.findall("DataRoutes/DataRoute/ScanInformation/TimePerStep")[0].text
    z_dict[constants.INTEGRATION_TIME_ID] = Dimension(float(inttime_str), constants.TIME_QT, constants.SECOND_UNIT)

    # X-rays wavelength
    wl_el = root.findall(
        "FixedInformation/Instrument/PrimaryTracks/TrackInfoData/MountedOptics/InfoData/Tube/WaveLengthAverage"
    )[0]
    wl_str = wl_el.get("Value")  # wavelength in angstrom
    z_dict[constants.WAVELENGTH_ID] = Dimension(float(wl_str), constants.WAVELENGTH_QT, constants.ANGSTROM_UNIT)

    # Measure time
    meas_time_str = root.findall("TimeStampFinished")[0].text
    try:
        end_time = dt.datetime.strptime(meas_time_str[:-7], date_format)
        measure_time = (end_time - date).total_seconds()
    except ValueError:
        measure_time = 0.0
    z_dict[constants.MEASUREMENT_TIME_ID] = Dimension(measure_time, constants.TIME_QT, constants.SECOND_UNIT)

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
    x = Dimension(x_data, constants.TIME_QT, constants.SECOND_UNIT)

    # First measurement as Timestamp
    z_dict = {constants.TIMESTAMP_ID: Dimension(measure_time[0], constants.TIME_QT)}

    # Y dimension (temperature)
    y_temp_data = np.array([float(f[2]) for f in data_str])
    y_temp = Dimension(y_temp_data, constants.TEMPERATURE_QT, constants.CELSIUS_UNIT)
    temperature = SignalData(x, y_temp, name, "Temperature", z_dict.copy())

    # Y dimension (humidity)
    y_hum_data = np.array([float(f[3]) for f in data_str])
    y_hum = Dimension(y_hum_data, constants.HUMIDITY_QT, constants.PERCENT_UNIT)
    humidity = SignalData(x, y_hum, name, "Humidity", z_dict.copy())

    return {"Temperature": temperature, "Humidity": humidity}


# -------------------------------------------------- F980 & FLUORACLE --------------------------------------------------


def EdinstFile(
    filename: str,
    delimiter: str | None = "\t",
) -> list[SignalData]:
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
        x_quantity, x_unit = constants.WAVELENGTH_QT, constants.NM_UNIT
    elif "Time Scan" in measure_type:
        x_quantity, x_unit = constants.TIME_QT, grep(content, "Time(", 0, ")")
        if x_unit is None:
            x_unit = constants.NANOSECOND_UNIT  # assume ns
    elif "Synchronous Scan" in measure_type:
        x_quantity, x_unit = constants.WAVELENGTH_QT, constants.NM_UNIT
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
        y_quantity, y_unit = constants.ABSORPTANCE_QT, constants.PERCENT_UNIT
        ys_data *= 100
    elif "Counts" in yaxis:
        y_quantity, y_unit = constants.INTENSITY_QT, constants.COUNTS_UNIT
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
        content = [f + "\t" + g for f, g in zip(new_col, content)]

    if grep(content, "SourceName"):

        # Header and data
        index_header = get_data_index(content)
        if not isinstance(index_header, int):
            raise AssertionError()  # pragma: no cover
        headers = get_header_as_dicts(content[:index_header])
        data = stringlist_to_matrix(content[index_header:])[1:]  # first column is ignored

        # X dimension
        x = Dimension(data[0], constants.WAVELENGTH_QT, headers[0]["Units"])

        # ys
        ys_data = data[1:]
        ys_headers = headers[1:]
        y_quantity = constants.INTENSITY_QT
        ys_unit = [header["Units"] for header in ys_headers]

        # Data type
        try:  # Excitation Emission map
            for header in ys_headers:
                wl = float(header["Long Name"])
                dim = Dimension(wl, constants.EXC_WAVELENGTH_QT, constants.NM_UNIT)
                header[constants.EXCITATION_WAVELENGTH_ID] = dim
            names = [header["Long Name"] + " " + constants.NM_UNIT for header in ys_headers]

        except ValueError:  # Timelapse
            date_format = "%m/%d/%Y %H:%M:%S"
            for header in ys_headers:
                date = dt.datetime.strptime(header["TimeStamp"], date_format)
                header[constants.TIMESTAMP_ID] = Dimension(date, constants.TIME_QT)
                del header["TimeStamp"]
            names = [str(header[constants.TIMESTAMP_ID].data) for header in ys_headers]

        # Converts the remaining values to dimensions
        for header in ys_headers:
            for key in header:
                if not isinstance(header[key], Dimension):
                    header[key] = Dimension(header[key])

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
                ys[header["Long Name"]] = Dimension(y_data, constants.WAVELENGTH_QT, header["Units"])
            else:
                ys[header["Long Name"]] = Dimension(y_data, constants.INTENSITY_QT, header["Units"])

        signals = {}
        for key in ys:
            if key != "Wavelength":
                signals[key] = SignalData(ys["Wavelength"], ys[key], name, f"({key})")

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
    x_data, y_data = stringlist_to_matrix(content[data_index + 1 : -2])
    x = Dimension(x_data, constants.WAVELENGTH_QT, constants.NM_UNIT)
    y = Dimension(y_data, constants.INTENSITY_QT, constants.AU_UNIT)

    # Timestamp
    date = dt.datetime.strptime(content[3] + " " + content[4][:8], "%d/%m/%y %X")
    z_dict = {constants.TIMESTAMP_ID: Dimension(date, constants.TIME_QT)}

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
    (y_data,) = stringlist_to_matrix(content[data_index + 1 :])
    y = get_uvvis_dimension(y_data, content[9])

    # X dimension
    x_index = np.where([c == "nm" for c in content])[0][0]  # find the index of the line 'nm'
    x_min = float(content[x_index + 1])  # lowest x
    x_max = float(content[x_index + 2])  # highest x
    nb_points = int(content[x_index + 4])  # number of points measured
    x_data = np.linspace(x_min, x_max, nb_points)
    x = Dimension(x_data, constants.WAVELENGTH_QT, constants.NM_UNIT)

    # Timestamp
    date_format1 = "%m/%d/%Y %I:%M:%S %p"
    date_format2 = "%d/%m/%Y %H:%M:%S"
    try:
        date = dt.datetime.strptime(content[16], date_format2)
    except ValueError:
        date = dt.datetime.strptime(content[16], date_format1)
    z_dict = {
        constants.TIMESTAMP_ID: Dimension(date, constants.TIME_QT),
        constants.SCAN_SPEED_ID: Dimension(float(content[62]), constants.SPEED_QT, constants.NM_MIN_UNIT),
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
    z_dict = {constants.TIMESTAMP_ID: Dimension(date, constants.TIME_QT)}

    nb_points = grep(content, "Time,", 0, "time points", "int")

    # ---------------------------------------------------- MAIN DATA ---------------------------------------------------

    data_index = grep(content, "Time")[1][1]
    if not isinstance(data_index, int):
        raise AssertionError()  # pragma: no cover
    data_index += 2
    data_raw = content[data_index : data_index + nb_points]
    data = stringlist_to_matrix(data_raw, delimiter)

    # X dimension
    x = Dimension(data[0], constants.TIME_QT, constants.SECOND_UNIT)

    # Y dimension
    ys_data = data[1:]
    data_type = content[data_index - 4]

    if data_type == "Rel_Absorbance":
        y_quantity, y_unit = constants.REL_ABS_QT, ""
    elif data_type == "Emission":
        y_quantity, y_unit = constants.INTENSITY_QT, constants.COUNTS_UNIT
    else:
        y_quantity, y_unit = "", ""

    # Check if any array is all zeros
    nonzero_indices = [i for i in range(len(ys_data)) if np.any(ys_data[i])]
    ys_data = ys_data[nonzero_indices]

    # Z dimension
    z_type = content[data_index - 2].strip("Time,")
    if z_type == "Repeat":
        dtype = int
    else:
        dtype = float
    z_data = np.array(content[data_index - 1].split(delimiter)[1:-1], dtype=dtype)
    z_data = z_data[nonzero_indices]

    if z_type == "Repeat":
        z_quantity, z_unit = constants.REPEATS_QT, ""
    elif z_type == "Wavelength":
        z_quantity, z_unit = constants.EM_WAVELENGTH_QT, constants.NM_UNIT
    else:
        z_quantity, z_unit = "", ""

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
            data_ = stringlist_to_matrix(content[data_index_ : data_index_ + nb_points], delimiter)
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
        signalname = z_dict["z"].get_value_label_html(5, "g", auto_exponent=False)
        signals.append(SignalData(x, Dimension(ys_data[i], y_quantity, y_unit), name, signalname, z_dict))

        if otb_data is not None:
            y = Dimension(otb_data[i], constants.INTENSITY_QT, constants.VOLT_UNIT)
            otb_signals.append(SignalData(x, y, name, signalname, z_dict))

        if huntb_data is not None:
            y = Dimension(huntb_data[i], constants.INTENSITY_QT, constants.VOLT_UNIT)
            huntb_signals.append(SignalData(x, y, name, signalname, z_dict))

        if rawabs_data is not None:
            y = Dimension(rawabs_data[i], constants.ABSORBANCE_QT)
            rawabs_signals.append(SignalData(x, y, name, signalname, z_dict))

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
    data_index = grep(content, "IV data")[0][1]
    if not isinstance(data_index, int):
        raise AssertionError()  # pragma: no cover
    data_index += 1
    if not isinstance(data_index, int):
        raise AssertionError()  # pragma: no cover
    data_header = [x for x in content[data_index].split("\t") if x != ""]

    data = stringlist_to_matrix([line.strip() for line in content[data_index + 1 :]])

    try:
        fw_index = data_header.index("VSource Forward")
    except ValueError:
        fw_index = -1

    try:
        bk_index = data_header.index("VSource Backward")
    except ValueError:
        bk_index = -1

    def get_data(
        index_start: int,
        index_end: int,
        sname: str,
    ) -> list[SignalData]:
        """Get the data"""
        sweep_indexes = np.arange(index_start + 1, index_end)
        sweeps = [data_header[i].replace(" (mA/cm\xb2)", "") for i in sweep_indexes]
        s_data = sort_lists(data[index_start:index_end], 0)
        x = Dimension(s_data[0], constants.VOLTAGE_QT, constants.VOLT_UNIT)
        ys = [Dimension(y_data, constants.CURRENT_DENSITY_QT, constants.MA_CM_2_UNIT) for y_data in s_data[1:]]
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
    data = stringlist_to_matrix(content[data_index + 1 :])
    x_data, y_data_cd, y_data_c, y_data_p, y_data_t = sort_lists(data, 0)

    x = Dimension(x_data, constants.VOLTAGE_QT, constants.VOLT_UNIT)
    current_density = SignalData(
        x, Dimension(y_data_cd, constants.CURRENT_DENSITY_QT, constants.MA_CM_2_UNIT), name, "Current density"
    )
    current = SignalData(x, Dimension(y_data_c, constants.CURRENT_QT, constants.MA_UNIT), name, "Current")
    power = SignalData(x, Dimension(y_data_p, constants.POWER_QT, constants.MW_UNIT), name, "Power")
    time = SignalData(x, Dimension(y_data_t, constants.TIME_QT, constants.SECOND_UNIT), name, "Time")

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
    x = Dimension(x_data, constants.WAVELENGTH_QT, constants.NM_UNIT)
    y = Dimension(y_data, constants.INTENSITY_QT, constants.COUNTS_UNIT)

    # Timestamp
    date_str = content[2][6:]
    date_format = "%a %b %d %H:%M:%S %Z %Y"
    if "BST" in date_str:  # some files show 'BST' instead of 'GMT'
        date_format = date_format.replace("%Z", "BST")
    date = dt.datetime.strptime(date_str, date_format)
    if "BST" in date_str:
        date -= dt.timedelta(seconds=3600)
    z_dict = {constants.TIMESTAMP_ID: Dimension(date, constants.TIME_QT)}

    # Integration time
    int_time = grep(content, "Integration Time (usec):", 0, "(", "float")
    z_dict[constants.INTEGRATION_TIME_ID] = Dimension(int_time / 1e6, constants.TIME_QT, constants.SECOND_UNIT)

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
            x = Dimension(x_data, constants.WAVENUMBER_QT, constants.CM_1_UNIT)
        elif x_q == "nm":
            x = Dimension(x_data, constants.WAVELENGTH_QT, constants.NM_UNIT)
        else:
            x = Dimension(x_data, x_q)

        # Y dimension
        y = get_uvvis_dimension(ys_data[0], y_q)

        return SignalData(x, y, name)

    else:

        # X dimension
        if content[index - 1] == '"Wavenumber"':
            x = Dimension(x_data, constants.WAVENUMBER_QT, constants.CM_1_UNIT)
        elif content[index - 1] == '"Wavelength"':
            x = Dimension(x_data, constants.WAVELENGTH_QT, constants.NM_UNIT)
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
    data = stringlist_to_matrix(content[data_index + 1 :])[:, ::-1]
    x = Dimension(data[0], constants.WAVELENGTH_QT, constants.NM_UNIT)
    y = get_uvvis_dimension(data[1], content[80])

    # Date
    date = dt.datetime.strptime(content[3] + " " + content[4][:-3], "%d/%m/%y %H:%M:%S")
    z_dict = {constants.TIMESTAMP_ID: Dimension(date, constants.TIME_QT)}

    return SignalData(x, y, name, z_dict=z_dict)


# ------------------------------------------------------- VESTA --------------------------------------------------------


def VestaDiffractionFile(filename: str) -> SignalData:
    """Read the content of an exported Vesta diffraction file
    :param filename: file path"""

    content, name = read_datafile(filename)
    x_data, y_data = stringlist_to_matrix(content)[:2]
    x = Dimension(x_data, constants.TWO_THETA_QT, constants.DEG_UNIT)
    y = Dimension(y_data, constants.INTENSITY_QT, constants.AU_UNIT)
    return SignalData(x, y, name)


# -------------------------------------------------------- WIRE --------------------------------------------------------


# noinspection PyTypeChecker
def WireFile(filename: str | BytesIO) -> SignalData:
    """Read Renishaw WiRE files
    :param filename: file path"""

    temp_filename = "temp_"
    if not isinstance(filename, str):
        with open(temp_filename, "wb") as ofile:
            ofile.write(filename.read())
            filename = temp_filename

    try:
        reader = WDFReader(filename)
        # noinspection PyUnresolvedReferences
        x_data, y_data = np.array(reader.xdata[::-1], dtype=float), np.array(reader.spectra[::-1], dtype=float)
        if reader.xlist_unit.name == "RamanShift":
            x = Dimension(x_data, constants.WAVENUMBER_QT, constants.CM_1_UNIT)
        else:
            x = Dimension(x_data, constants.WAVELENGTH_QT, constants.NM_UNIT)
        y = Dimension(y_data, constants.INTENSITY_QT)
        return SignalData(x, y, reader.title)
    finally:
        try:
            # noinspection PyUnboundLocalVariable
            reader.close()
        except:
            pass
        if filename == temp_filename:
            try:
                os.remove(filename)
            except:
                pass


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
        for d, h in zip(data[xcol + 1 :], headers[xcol + 1 :])
    }
    if not data:
        raise AssertionError()  # pragma: no cover
    else:
        return data


# -------------------------------------------------------- READ FUNCTIONS --------------------------------------------------------

READ_FUNCTIONS = {
    "SpectraSuite (.txt)": (SpectraSuiteFile, "txt"),
    "FluorEssence (.txt)": (FluorEssenceFile, "txt"),
    "EasyLog (.txt)": (EasyLogFile, "txt"),
    "Beampro (.txt)": (BeamproFile, "txt"),
    "F980/Fluoracle (.txt, tab)": (EdinstFile, "txt"),
    "F980/Fluoracle (.txt, comma)": (lambda filename: EdinstFile(filename, delimiter=","), "txt"),
    "Dektak (.csv)": (DektakFile, "csv"),
    "ProData (.csv)": (ProDataSignal, "csv"),
    "UvWinlab/Spectrum (.csv)": (PerkinElmerFile, "csv"),
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


FILETYPES = ["Detect"] + sorted(READ_FUNCTIONS.keys())


def detect_file_type(filename: str | BytesIO) -> tuple[dict | list | SignalData, str] | tuple[None, None]:
    """Detect the type of file and load it
    :param filename: file to load
    :return: a tuple of the signal returned (dict, list or SignalData) and the detected file type or (None, None)"""

    # Only search through the filetypes matching the extension
    if isinstance(filename, str):
        extension = os.path.splitext(filename)[1]
    else:
        extension = os.path.splitext(filename.name)[1]
    filtered_functions = {}
    for key, value in READ_FUNCTIONS.items():
        if extension.lower() == "." + READ_FUNCTIONS[key][1].lower() or READ_FUNCTIONS[key][1] == "":
            filtered_functions[key] = value[0]

    # Try to load the data with each function until successful
    for filetype in filtered_functions:
        try:
            signal = READ_FUNCTIONS[filetype][0](filename)
            return signal, filetype
        except:
            pass

    return None, None


def read_data_file(
    filenames: str | list[str],
    filetype: str,
) -> tuple[list[SignalData], str | None]:
    """Read the file(s) content.
    If filetype is "Detect" and filename is a list of files, detect the first file type and then reuse that type for
    the other files. If a file cannot be loaded in the list, it is not returned.
    :param filenames: file filename
    :param filetype: file type
    :return: a list of SignalData and the file type loaded (None if not loaded)"""

    # Convert single file to list
    if not isinstance(filenames, list):
        filenames = [filenames]

    # Load the first signal
    if filetype == FILETYPES[0]:
        signal, filetype = detect_file_type(filenames[0])
    else:
        try:
            signal = READ_FUNCTIONS[filetype][0](filenames[0])
        except Exception as e:
            signal = None
            print(e)

    # Iterate over the remaining paths
    if signal is not None:  # Make sure that the file could be read
        signals = [signal]
        for file in filenames[1:]:

            try:
                signals.append(READ_FUNCTIONS[filetype][0](file))
            except Exception as e:
                print(e)

        return signals, filetype

    else:
        return [], None
