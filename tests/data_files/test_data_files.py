"""Test module for the functions in the `data_files.py` module.

This module contains unit tests for the functions implemented in the `data_files.py` module. The purpose of these tests is to
ensure the correct functionality of each function in different scenarios and to validate that the expected outputs are
returned.

Tests should cover various edge cases, valid inputs, and any other conditions that are necessary to confirm the
robustness of the functions."""

import os.path
import tempfile

import pytest

from config import resources
from data_files.data_files import *
from utils.checks import are_close


@pytest.mark.parametrize(
    "mode, expected_quantity, expected_unit",
    [
        ("T", constants.TRANSMITTANCE_QT, constants.PERCENT_UNIT),
        ("R", constants.REFLECTANCE_QT, constants.PERCENT_UNIT),
        ("A", constants.ABSORBANCE_QT, ""),
        ("XYZ", "", ""),  # default case
    ],
)
def test_get_uvvis_dimension(mode: str, expected_quantity: str, expected_unit: str) -> None:

    data = np.array([1, 2, 3])
    dim = get_uvvis_dimension(data, mode)

    assert isinstance(dim, Dimension)
    assert are_close(dim.data, data)
    assert are_close(dim.quantity, expected_quantity)
    assert are_close(dim.unit, expected_unit)


class TestReadDatafile:

    def setup_method(self) -> None:
        """Create temporary files for testing"""

        self.utf8_content = "Line 1\nLine 2\nLine 3"
        self.latin1_content = "Café\nÉtude\nNaïve"
        self.binary_content = b"\x80\x81\x82\nBinary data\n\xff\xfe"

    def test_read_string_path(self) -> None:
        """Test reading from a file path as string"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as temp_file:
            temp_file.write(self.utf8_content)
            temp_path = temp_file.name

        try:
            content, name = read_datafile(temp_path)
            assert content == ["Line 1", "Line 2", "Line 3"]
            assert name == os.path.basename(temp_path).split(".")[0]
        finally:
            os.unlink(temp_path)

    def test_read_bytesio_utf8(self) -> None:
        """Test reading from BytesIO with UTF-8 content"""

        mock_file = BytesIO(self.utf8_content.encode("utf-8"))
        mock_file.name = "test_file.txt"

        content, name = read_datafile(mock_file)
        assert content == ["Line 1", "Line 2", "Line 3"]
        assert name == "test_file"

    def test_read_bytesio_latin1(self) -> None:
        """Test reading from BytesIO with Latin-1 content"""

        mock_file = BytesIO(self.latin1_content.encode("latin-1"))
        mock_file.name = "latin_file.txt"

        content, name = read_datafile(mock_file)
        assert content == ["Café", "Étude", "Naïve"]
        assert name == "latin_file"

    def test_read_bytesio_binary(self) -> None:
        """Test reading from BytesIO with binary content"""

        mock_file = BytesIO(self.binary_content)
        mock_file.name = "binary_file.bin"

        content, name = read_datafile(mock_file)
        # Binary content might be read differently depending on platform
        # Just verify we get some content back without errors
        assert isinstance(content, list)
        assert len(content) > 0
        assert name == "binary_file"

    def test_filename_with_multiple_dots(self) -> None:
        """Test handling filenames with multiple dots"""

        mock_file = BytesIO(b"Some content")
        mock_file.name = "file.filename.with.dots.txt"

        content, name = read_datafile(mock_file)
        assert name == "file"  # Should take only the part before the first dot

    def test_empty_file(self) -> None:
        """Test reading an empty file"""

        mock_file = BytesIO(b"")
        mock_file.name = "empty_file.txt"

        content, name = read_datafile(mock_file)
        assert content == []  # Empty string gets split into a list with one empty element
        assert name == "empty_file"

    def test_file_with_special_chars(self) -> None:
        """Test file with special characters and mixed line endings"""

        special_content = "Line 1\r\nLine 2\nSymbols: !@#$%^&*()\rTab\tTest"
        mock_file = BytesIO(special_content.encode("utf-8"))
        mock_file.name = "special.txt"

        content, name = read_datafile(mock_file)
        assert len(content) == 4  # Should handle mixed line endings
        assert "Symbols: !@#$%^&*()" in content
        assert "Tab\tTest" in content


def print_signal_info(signal: SignalData) -> None:  # pragma: no cover
    """Print the expect signal data"""

    expected = {
        "x_data": np.concatenate([signal.x.data[:3], signal.x.data[-3:]]),
        "y_data": np.concatenate([signal.y.data[:3], signal.y.data[-3:]]),
        "x_quantity": signal.x.quantity,
        "y_quantity": signal.y.quantity,
        "x_unit": signal.x.unit,
        "y_unit": signal.y.unit,
        "filename": signal.filename,
        "signalname": signal.signalnames[0],
        "z_dict": signal.z_dict,
    }

    def format_value(val: any) -> str:  # pragma: no cover
        """Format a value into a string"""
        if isinstance(val, np.ndarray):
            # Handle arrays of datetime
            if np.issubdtype(val.dtype, np.datetime64):
                return f"np.array([{', '.join(format_value(v) for v in val)}])"
            else:
                return f"np.array{repr(val).replace('array', '')}"
        elif isinstance(val, dt.datetime):
            return f"dt.datetime({val.year}, {val.month}, {val.day}, {val.hour}, {val.minute}, {val.second}, {val.microsecond})"
        else:
            return repr(val)

    print("expected = {")
    for key, value in expected.items():
        if key == "z_dict":
            print("    'z_dict': {")
            for key2, value2 in value.items():
                print(
                    f"        '{key2}': Dimension({format_value(value2.data)}, '{value2.quantity}', '{value2.unit}'),"
                )
            print("        }")
        else:
            print(f"    '{key}': {format_value(value)},")
    print("    }")


class TestDataFiles:

    @staticmethod
    def assert_signal(
        signal: SignalData,
        expected: dict,
    ) -> None:
        """Check that the signal data correspond to expected
        :param signal: SignalData object
        :param expected: expected data"""

        assert are_close(np.concatenate([signal.x.data[:3], signal.x.data[-3:]]), expected["x_data"])
        assert are_close(np.concatenate([signal.y.data[:3], signal.y.data[-3:]]), expected["y_data"])
        assert signal.x.quantity == expected["x_quantity"]
        assert signal.y.quantity == expected["y_quantity"]
        assert signal.x.unit == expected["x_unit"]
        assert signal.y.unit == expected["y_unit"]
        assert signal.filename == expected["filename"]
        assert signal.signalnames[0] == expected["signalname"]
        for key in expected["z_dict"]:
            assert are_close(signal.z_dict[key].data, expected["z_dict"][key].data)
            assert are_close(signal.z_dict[key].quantity, expected["z_dict"][key].quantity)
            assert are_close(signal.z_dict[key].unit, expected["z_dict"][key].unit)

    def test_breampro(self) -> None:
        data = BeamproFile(resources.BEAMPRO_PATH)
        expected = {
            "x_data": np.array([0.00000e00, 6.60584e00, 1.32117e01, 1.35089e04, 1.35156e04, 1.35222e04]),
            "y_data": np.array([1.24512, 1.5625, 1.14746, 1.14746, 1.00098, 1.34277]),
            "x_quantity": "distance",
            "y_quantity": "intensity",
            "x_unit": "um",
            "y_unit": "a.u.",
            "filename": "BeamPro",
            "signalname": "X",
            "z_dict": {},
        }
        self.assert_signal(data["x"], expected)
        expected = {
            "x_data": np.array([0.00000e00, 6.60584e00, 1.32117e01, 1.35089e04, 1.35156e04, 1.35222e04]),
            "y_data": np.array([0.0, 0.0976563, 1.12305, 1.0498, 1.26953, 1.02539]),
            "x_quantity": "distance",
            "y_quantity": "intensity",
            "x_unit": "um",
            "y_unit": "a.u.",
            "filename": "BeamPro",
            "signalname": "Y",
            "z_dict": {},
        }
        self.assert_signal(data["y"], expected)

    def test_dektak(self) -> None:

        data = DektakFile(resources.DEKTAK_PATH)
        expected = {
            "x_data": np.array([0.00000e00, 5.10000e-01, 1.03000e00, 1.99897e03, 1.99949e03, 2.00000e03]),
            "y_data": np.array([-649.13, -648.21, -643.29, -70.38, -77.46, -84.53]),
            "x_quantity": "horizontal distance",
            "y_quantity": "vertical distance",
            "x_unit": "um",
            "y_unit": "um",
            "filename": "Dektak",
            "signalname": "",
            "z_dict": {
                "Date & Time": Dimension(dt.datetime(2017, 5, 23, 11, 35, 19), "time", ""),
            },
        }
        self.assert_signal(data, expected)

    def test_read_brml_rawdata_file(self) -> None:

        with zipfile.ZipFile(resources.DIFFRAC_PATH) as brml:
            datafile = brml.infolist()[-3]
            data = read_brml_rawdata_file(brml, datafile)

        expected = {
            "x_data": np.array([10.0001, 10.0197, 10.0394, 59.9705, 59.9901, 60.0097]),
            "y_data": np.array([191.0, 208.0, 224.0, 95.0, 68.0, 86.0]),
            "x_quantity": "2 theta",
            "y_quantity": "intensity",
            "x_unit": "deg",
            "y_unit": "counts",
            "filename": "diffrac",
            "signalname": "RawData0",
            "z_dict": {
                "Date & Time": Dimension(dt.datetime(2017, 8, 4, 16, 43, 2, 345953), "time", ""),
                "Integration Time": Dimension(0.15, "time", "s"),
                "Wavelength": Dimension(1.5418, "wavelength", "angstrom"),
                "Measurement Time": Dimension(412.58348, "time", "s"),
                "Chi": Dimension(-0.0, "Chi", "°"),
                "Phi": Dimension(0.0, "Phi", "°"),
                "X": Dimension(-0.0, "X", "mm"),
                "Y": Dimension(-0.0, "Y", "mm"),
                "Z": Dimension(1.4, "Z", "mm"),
            },
        }
        self.assert_signal(data, expected)

    def test_diffrac(self) -> None:

        data = DiffracBrmlFile(resources.DIFFRAC_PATH)
        expected = {
            "x_data": np.array([10.0001, 10.0197, 10.0394, 59.9705, 59.9901, 60.0097]),
            "y_data": np.array([191.0, 208.0, 224.0, 95.0, 68.0, 86.0]),
            "x_quantity": "2 theta",
            "y_quantity": "intensity",
            "x_unit": "deg",
            "y_unit": "counts",
            "filename": "diffrac",
            "signalname": "RawData0",
            "z_dict": {
                "Date & Time": Dimension(dt.datetime(2017, 8, 4, 16, 43, 2, 345953), "time", ""),
                "Integration Time": Dimension(0.15, "time", "s"),
                "Wavelength": Dimension(1.5418, "wavelength", "angstrom"),
                "Measurement Time": Dimension(412.58348, "time", "s"),
                "Chi": Dimension(-0.0, "Chi", "°"),
                "Phi": Dimension(0.0, "Phi", "°"),
                "X": Dimension(-0.0, "X", "mm"),
                "Y": Dimension(-0.0, "Y", "mm"),
                "Z": Dimension(1.4, "Z", "mm"),
            },
        }
        self.assert_signal(data[0], expected)

        data = DiffracBrmlFile(resources.DIFFRAC_TIMELAPSE_PATH)
        expected = {
            "x_data": np.array([12.0001, 12.0021, 12.0041, 14.4961, 14.4981, 14.5001]),
            "y_data": np.array([0.0, 0.0, 178.0, 52.0, 48.0, 53.0]),
            "x_quantity": "2 theta",
            "y_quantity": "intensity",
            "x_unit": "deg",
            "y_unit": "counts",
            "filename": "Diffrac_multiple",
            "signalname": "RawData79",
            "z_dict": {
                "Date & Time": Dimension(dt.datetime(2017, 10, 11, 9, 29, 34, 753172), "time", ""),
                "Integration Time": Dimension(0.5, "time", "s"),
                "Wavelength": Dimension(1.5418, "wavelength", "angstrom"),
                "Measurement Time": Dimension(676.947589, "time", "s"),
                "Theta": Dimension(2.0186, "Theta", "°"),
                "Chi": Dimension(-0.0, "Chi", "°"),
                "Phi": Dimension(0.0, "Phi", "°"),
                "X": Dimension(-0.0, "X", "mm"),
                "Y": Dimension(-0.0, "Y", "mm"),
                "Z": Dimension(-0.0395, "Z", "mm"),
            },
        }
        self.assert_signal(data[-1], expected)

        data = DiffracBrmlFile(resources.DIFFRAC_PSD_PATH)
        expected = {
            "x_data": np.array([8.5001, 8.5118762, 8.52365241, 10.73757886, 10.74935506, 10.76113126]),
            "y_data": np.array([5.000e00, 1.000e00, 2.129e03, 0.000e00, 1.000e00, 1.000e00]),
            "x_quantity": "2 theta",
            "y_quantity": "intensity",
            "x_unit": "deg",
            "y_unit": "counts",
            "filename": "Diffrac_PSD",
            "signalname": "RawData0",
            "z_dict": {
                "Date & Time": Dimension(dt.datetime(2018, 3, 5, 12, 11, 11, 816786), "time", ""),
                "Integration Time": Dimension(5.0, "time", "s"),
                "Wavelength": Dimension(1.5418, "wavelength", "angstrom"),
                "Measurement Time": Dimension(23.337641, "time", "s"),
                "Theta": Dimension(4.8153, "Theta", "°"),
                "Chi": Dimension(-0.0, "Chi", "°"),
                "Phi": Dimension(0.0, "Phi", "°"),
                "X": Dimension(-0.0, "X", "mm"),
                "Y": Dimension(-0.0, "Y", "mm"),
                "Z": Dimension(0.8455, "Z", "mm"),
            },
        }
        self.assert_signal(data[0], expected)

    def test_easylog(self) -> None:

        data = EasyLogFile(resources.EASYLOG_PATH)
        expected = {
            "x_data": np.array([0.0000e00, 1.2000e02, 2.4000e02, 2.5284e05, 2.5296e05, 2.5308e05]),
            "y_data": np.array([19.0, 19.0, 18.5, 18.5, 19.5, 22.5]),
            "x_quantity": "time",
            "y_quantity": "temperature",
            "x_unit": "s",
            "y_unit": "deg C",
            "filename": "Easylog",
            "signalname": "Temperature",
            "z_dict": {
                "Date & Time": Dimension(dt.datetime(2017, 2, 19, 11, 29, 22), "time", ""),
            },
        }
        self.assert_signal(data["Temperature"], expected)
        expected = {
            "x_data": np.array([0.0000e00, 1.2000e02, 2.4000e02, 2.5284e05, 2.5296e05, 2.5308e05]),
            "y_data": np.array([48.0, 47.5, 49.0, 57.0, 55.5, 50.5]),
            "x_quantity": "time",
            "y_quantity": "humidity",
            "x_unit": "s",
            "y_unit": "%",
            "filename": "Easylog",
            "signalname": "Humidity",
            "z_dict": {
                "Date & Time": Dimension(dt.datetime(2017, 2, 19, 11, 29, 22), "time", ""),
            },
        }
        self.assert_signal(data["Humidity"], expected)

    def test_edinst(self) -> None:

        data = EdinstFile(resources.F980_IRF_PATH)
        expected = {
            "x_data": np.array([0.0, 0.9765625, 1.953125, 497.07031, 498.04688, 499.02344]),
            "y_data": np.array([4.0, 1.0, 2.0, 5.0, 1.0, 8.0]),
            "x_quantity": "time",
            "y_quantity": "intensity",
            "x_unit": "ns",
            "y_unit": "counts",
            "filename": "F980_IRF",
            "signalname": np.str_("T5"),
            "z_dict": {
                "Labels": Dimension("T5", "", ""),
                "Type": Dimension("Instrument Response Time Scan", "", ""),
                "Comment": Dimension("", "", ""),
                "Start": Dimension("0.00", "", ""),
                "Stop": Dimension("499.023440", "", ""),
                "Step": Dimension("0.97656250", "", ""),
                "Fixed/Offset": Dimension(" ", "", ""),
                "XAxis": Dimension("Time(ns)", "", ""),
                "YAxis": Dimension("Counts", "", ""),
            },
        }
        self.assert_signal(data[0], expected)

        data = EdinstFile(resources.F980_EMSCAN_PATH)
        expected = {
            "x_data": np.array([675.0, 676.0, 677.0, 818.0, 819.0, 820.0]),
            "y_data": np.array([0.0, 22.4496574, 22.899437, 11287.127, 6189.1084, 0.0]),
            "x_quantity": "wavelength",
            "y_quantity": "intensity",
            "x_unit": "nm",
            "y_unit": "counts",
            "filename": "F980_EmScan",
            "signalname": np.str_("EmScan9"),
            "z_dict": {
                "Labels": Dimension("EmScan9", "", ""),
                "Type": Dimension("Emission Scan", "", ""),
                "Comment": Dimension("", "", ""),
                "Start": Dimension("675.00", "", ""),
                "Stop": Dimension("820.00", "", ""),
                "Step": Dimension("1.00", "", ""),
                "Fixed/Offset": Dimension("-100.00", "", ""),
                "Xaxis": Dimension("Wavelength", "", ""),
                "Yaxis": Dimension("Counts", "", ""),
                "Scan Corr. by File": Dimension("True", "", ""),
                "Corr. by Ref. Det.": Dimension("False", "", ""),
                "Fixed/Offset Corr. by File": Dimension("False", "", ""),
                "Repeats": Dimension("1", "", ""),
                "Dwell Time": Dimension("0.20", "", ""),
                "Lamp": Dimension("TCSPC Diode", "", ""),
                "Temp": Dimension("25.20", "", ""),
                "Scan Polariser": Dimension("None", "", ""),
                "Scan Slit": Dimension("20.037106", "", ""),
                "Fixed/Offset Polariser": Dimension("None", "", ""),
                "Fixed/Offset Slit": Dimension("0.00", "", ""),
                "Detector": Dimension("MCP", "", ""),
            },
        }
        self.assert_signal(data[0], expected)

        data = EdinstFile(resources.F980_IRF_MULTIPLE_PATH)
        expected = {
            "x_data": np.array([0.0, 3.90625, 7.8125, 988.28125, 992.1875, 996.09375]),
            "y_data": np.array([7.0, 2.0, 3.0, 3.0, 4.0, 7.0]),
            "x_quantity": "time",
            "y_quantity": "intensity",
            "x_unit": "ns",
            "y_unit": "counts",
            "filename": "F980_IRF_multiple",
            "signalname": np.str_("Decay1"),
            "z_dict": {
                "Labels": Dimension("Decay1", "", ""),
                "Type": Dimension("Time Scan", "", ""),
                "Comment": Dimension("", "", ""),
                "Start": Dimension("0.00", "", ""),
                "Stop": Dimension("996.093750", "", ""),
                "Step": Dimension("3.906250", "", ""),
                "Fixed/Offset": Dimension(" ", "", ""),
                "XAxis": Dimension("Time(ns)", "", ""),
                "YAxis": Dimension("Counts", "", ""),
            },
        }
        self.assert_signal(data[0], expected)

        data = EdinstFile(resources.F980_IRF_COMMA_PATH, delimiter=",")
        expected = {
            "x_data": np.array([0.0, 1.953125, 3.90625, 494.14063, 496.09375, 498.04688]),
            "y_data": np.array([26.0, 29.0, 30.0, 38.0, 43.0, 26.0]),
            "x_quantity": "time",
            "y_quantity": "intensity",
            "x_unit": "ns",
            "y_unit": "counts",
            "filename": "F980_IRF_comma",
            "signalname": np.str_("Z10"),
            "z_dict": {
                "Labels": Dimension("Z10", "", ""),
                "Type": Dimension("Time Scan", "", ""),
                "Comment": Dimension("", "", ""),
                "Start": Dimension("0.00", "", ""),
                "Stop": Dimension("498.046880", "", ""),
                "Step": Dimension("1.9531250", "", ""),
                "Fixed/Offset": Dimension(" ", "", ""),
                "XAxis": Dimension("Time(ns)", "", ""),
                "YAxis": Dimension("Counts", "", ""),
            },
        }
        self.assert_signal(data[0], expected)

        data = EdinstFile(resources.FLUORACLE_EMISSION_PATH)
        expected = {
            "x_data": np.array([375.0, 376.0, 377.0, 598.0, 599.0, 600.0]),
            "y_data": np.array([10774.1357, 10104.75, 9978.43262, 852.291443, 884.633545, 796.848145]),
            "x_quantity": "wavelength",
            "y_quantity": "intensity",
            "x_unit": "nm",
            "y_unit": "counts",
            "filename": "Fluoracle_emission",
            "signalname": np.str_("noSample_emission"),
            "z_dict": {
                "Labels": Dimension("noSample_emission", "", ""),
                "Type": Dimension("Emission Scan", "", ""),
                "Comment": Dimension("", "", ""),
                "Start": Dimension("375.00", "", ""),
                "Stop": Dimension("600.00", "", ""),
                "Step": Dimension("1.00", "", ""),
                "Fixed/Offset": Dimension("350.00", "", ""),
                "Xaxis": Dimension("Wavelength", "", ""),
                "Yaxis": Dimension("Counts", "", ""),
                "Scan Corr. by File": Dimension("True", "", ""),
                "Corr. by Ref. Det.": Dimension("True", "", ""),
                "Fixed/Offset Corr. by File": Dimension("False", "", ""),
                "Repeats": Dimension("1", "", ""),
                "Dwell Time": Dimension("0.10", "", ""),
                "Lamp": Dimension("Xenon", "", ""),
                "Temp": Dimension("0.00", "", ""),
                "Scan Polariser": Dimension("None", "", ""),
                "Scan Slit": Dimension("4.2503014", "", ""),
                "Fixed/Offset Polariser": Dimension("None", "", ""),
                "Fixed/Offset Slit": Dimension("4.2503014", "", ""),
                "Detector": Dimension("Detector 1", "", ""),
            },
        }
        self.assert_signal(data[0], expected)

        data = EdinstFile(resources.FLUORACLE_MULTIPLE_EMISSION_PATH)
        expected = {
            "x_data": np.array([500.0, 500.5, 501.0, 599.0, 599.5, 600.0]),
            "y_data": np.array([25.375803, 27.6460876, 26.2228432, 14.2663317, 17.3304443, 15.4786711]),
            "x_quantity": "wavelength",
            "y_quantity": "intensity",
            "x_unit": "nm",
            "y_unit": "counts",
            "filename": "Fluoracle_emission_multiple",
            "signalname": np.str_("Em 1"),
            "z_dict": {
                "Labels": Dimension("Em 1", "", ""),
                "Type": Dimension("Emission Scan", "", ""),
                "Comment": Dimension("", "", ""),
                "Start": Dimension("500.00", "", ""),
                "Stop": Dimension("600.00", "", ""),
                "Step": Dimension("0.500", "", ""),
                "Fixed/Offset": Dimension("460.00", "", ""),
                "Xaxis": Dimension("Wavelength", "", ""),
                "Yaxis": Dimension("Counts", "", ""),
                "Scan Corr. by File": Dimension("True", "", ""),
                "Corr. by Ref. Det.": Dimension("True", "", ""),
                "Fixed/Offset Corr. by File": Dimension("False", "", ""),
                "Repeats": Dimension("1", "", ""),
                "Dwell Time": Dimension("0.50", "", ""),
                "Lamp": Dimension("Xenon", "", ""),
                "Temp": Dimension("0.00", "", ""),
                "Scan Polariser": Dimension("None", "", ""),
                "Scan Slit": Dimension("0.40031144", "", ""),
                "Fixed/Offset Polariser": Dimension("None", "", ""),
                "Fixed/Offset Slit": Dimension("3.9998331", "", ""),
                "Detector": Dimension("Detector 1", "", ""),
            },
        }
        self.assert_signal(data[0], expected)

    def test_fluoressence(self) -> None:

        data = FluorEssenceFile(resources.FLUORESSENCE_PATH)
        expected = {
            "x_data": np.array([490.0, 491.0, 492.0, 698.0, 699.0, 700.0]),
            "y_data": np.array(
                [2322.54697681, 2377.29177909, 2484.62736598, 3092.58586409, 3032.20574198, 2840.89992558]
            ),
            "x_quantity": "wavelength",
            "y_quantity": "intensity",
            "x_unit": np.str_("nm"),
            "y_unit": np.str_("CPS / MicroAmps"),
            "filename": "FluorEssence",
            "signalname": "(S1c / R1c)",
            "z_dict": {},
        }
        self.assert_signal(data["S1c / R1c"], expected)

        data = FluorEssenceFile(resources.FLUORESSENCE_ALLCOL_PATH)
        expected = {
            "x_data": np.array([700.0, 701.0, 702.0, 848.0, 849.0, 850.0]),
            "y_data": np.array([243.32337943, 253.62609798, 244.50769122, 165.42510121, 154.60077388, 154.53307392]),
            "x_quantity": "wavelength",
            "y_quantity": "intensity",
            "x_unit": np.str_("nm"),
            "y_unit": np.str_("CPS / MicroAmps"),
            "filename": "FluorEssence_allcol",
            "signalname": "(S1c / R1c)",
            "z_dict": {},
        }
        self.assert_signal(data["S1c / R1c"], expected)

        expected = {
            "x_data": np.array([700.0, 701.0, 702.0, 848.0, 849.0, 850.0]),
            "y_data": np.array([9800.0, 10220.0, 9850.0, 6660.0, 6230.0, 6230.0]),
            "x_quantity": "wavelength",
            "y_quantity": "intensity",
            "x_unit": np.str_("nm"),
            "y_unit": np.str_("CPS"),
            "filename": "FluorEssence_allcol",
            "signalname": "(S1)",
            "z_dict": {},
        }
        self.assert_signal(data["S1"], expected)

        data = FluorEssenceFile(resources.FLUORESSENCE_MULTIPLE_PATH)
        expected = {
            "x_data": np.array([650.0, 651.0, 652.0, 848.0, 849.0, 850.0]),
            "y_data": np.array([111.93601741, 96.76899294, 100.82534075, 1940.96407674, 2386.23609332, 1665.61854766]),
            "x_quantity": "wavelength",
            "y_quantity": "intensity",
            "x_unit": np.str_("nm"),
            "y_unit": np.str_("CPS / MicroAmps"),
            "filename": "FluorEssence_multiple",
            "signalname": "2017-05-12 17:24:47",
            "z_dict": {
                "Short Name": Dimension(np.str_("B"), "", ""),
                "Long Name": Dimension(np.str_("05/12/2017 17:24:47"), "", ""),
                "Units": Dimension(np.str_("CPS / MicroAmps"), "", ""),
                "SourceName": Dimension(np.str_("S1c / R1c"), "", ""),
                "CycleNumber": Dimension(np.str_("1"), "", ""),
                "Date & Time": Dimension(dt.datetime(2017, 5, 12, 17, 24, 47, 0), "time"),
            },
        }
        self.assert_signal(data[0], expected)

    def test_flwinlab(self) -> None:

        data = FlWinlabFile(resources.FLWINLAB_PATH)
        expected = {
            "x_data": np.array([350.0, 350.5, 351.0, 598.5, 599.0, 599.5]),
            "y_data": np.array([65.282, 54.64567, 44.866456, 5.854989, 5.846911, 5.84554]),
            "x_quantity": "wavelength",
            "y_quantity": "intensity",
            "x_unit": "nm",
            "y_unit": "a.u.",
            "filename": "FLWinlab",
            "signalname": "",
            "z_dict": {
                "Date & Time": Dimension(dt.datetime(2015, 3, 18, 16, 1, 4), "time", ""),
            },
        }
        self.assert_signal(data, expected)

    def test_lambdaspx(self) -> None:

        data = LambdaSpxFile(resources.LAMBDASPX_ABSORBANCE_PATH)
        expected = {
            "x_data": np.array([700.0, 700.5, 701.0, 799.0, 799.5, 800.0]),
            "y_data": np.array([0.5041709, 0.50298887, 0.50147861, 0.21925835, 0.21803915, 0.21983826]),
            "x_quantity": "wavelength",
            "y_quantity": "absorbance",
            "x_unit": "nm",
            "y_unit": "",
            "filename": "LambdaSPX_absorbance",
            "signalname": "",
            "z_dict": {
                "Date & Time": Dimension(dt.datetime(2017, 2, 14, 11, 25, 23), "time", ""),
                "Scan Speed": Dimension(480.0, "speed", "nm/min"),
            },
        }
        self.assert_signal(data, expected)

        data = LambdaSpxFile(resources.LAMBDASPX_REFLECTANCE_PATH)
        expected = {
            "x_data": np.array([700.0, 700.5, 701.0, 799.0, 799.5, 800.0]),
            "y_data": np.array([31.24734461, 31.35104477, 31.39903992, 60.31588614, 60.29761881, 60.16479731]),
            "x_quantity": "wavelength",
            "y_quantity": "reflectance",
            "x_unit": "nm",
            "y_unit": "%",
            "filename": "LambdaSPX_reflectance",
            "signalname": "",
            "z_dict": {
                "Date & Time": Dimension(dt.datetime(2017, 2, 14, 11, 33, 45), "time", ""),
                "Scan Speed": Dimension(480.0, "speed", "nm/min"),
            },
        }
        self.assert_signal(data, expected)

        data = LambdaSpxFile(resources.LAMBDASPX_TRANSMITTANCE_PATH)
        expected = {
            "x_data": np.array([700.0, 700.5, 701.0, 799.0, 799.5, 800.0]),
            "y_data": np.array([31.42957985, 31.43935651, 31.4599216, 60.38159132, 60.50872058, 60.50410867]),
            "x_quantity": "wavelength",
            "y_quantity": "transmittance",
            "x_unit": "nm",
            "y_unit": "%",
            "filename": "LambdaSPX_transmittance",
            "signalname": "",
            "z_dict": {
                "Date & Time": Dimension(dt.datetime(2017, 2, 14, 11, 27, 32), "time", ""),
                "Scan Speed": Dimension(480.0, "speed", "nm/min"),
            },
        }
        self.assert_signal(data, expected)

        data = LambdaSpxFile(resources.LAMBDASPX_ABSORBANCE2_PATH)
        expected = {
            "x_data": np.array([200.0, 201.0, 202.0, 798.0, 799.0, 800.0]),
            "y_data": np.array([0.62599342, 0.64735628, 0.66461102, 0.1690872, 0.16830412, 0.16843767]),
            "x_quantity": "wavelength",
            "y_quantity": "absorbance",
            "x_unit": "nm",
            "y_unit": "",
            "filename": "LambdaSPX_absorbance_other-date-format",
            "signalname": "",
            "z_dict": {
                "Date & Time": Dimension(dt.datetime(2017, 6, 7, 11, 3), "time", ""),
                "Scan Speed": Dimension(960.0, "speed", "nm/min"),
            },
        }
        self.assert_signal(data, expected)

    def test_prodata(self) -> None:

        data = ProDataSignal(resources.PRODATA_PLL_12WL_1PROP_PATH)
        expected = {
            "x_data": np.array([-9.98e-07, -9.94e-07, -9.90e-07, -4.98e-07, -4.94e-07, -4.90e-07]),
            "y_data": np.array([-3.12e-16, -3.12e-16, -3.12e-16, -3.12e-16, -3.12e-16, -3.12e-16]),
            "x_quantity": "time",
            "y_quantity": "intensity",
            "x_unit": "s",
            "y_unit": "counts",
            "filename": "prodata-pll_12wl_1prop",
            "signalname": "Emission wavelength: 820 nm",
            "z_dict": {
                "Date & Time": Dimension(dt.datetime(2017, 4, 3, 15, 11, 20), "time", ""),
                "z": Dimension(815.0, "emission wavelength", "nm"),
            },
        }
        self.assert_signal(data["dT/T"][0], expected)

        data = ProDataSignal(resources.PRODATA_TAS_3PROP_PATH)
        expected = {
            "x_data": np.array([-9.89e-07, -9.73e-07, -9.57e-07, 9.61e-07, 9.77e-07, 9.93e-07]),
            "y_data": np.array([0.440035, 0.43998, 0.440085, 0.440225, 0.44, 0.44009]),
            "x_quantity": "time",
            "y_quantity": "intensity",
            "x_unit": "s",
            "y_unit": "V",
            "filename": "prodata-tas_3prop",
            "signalname": "Repeats: 1",
            "z_dict": {
                "Date & Time": Dimension(dt.datetime(2017, 7, 27, 17, 39, 4, 0), "time", ""),
                "z": Dimension(np.float64(3.0), "repeats", ""),
            },
        }
        self.assert_signal(data["100 % T Baseline"][0], expected)

    def test_sbtps_seq(self) -> None:
        data = SbtpsSeqFile(resources.SBTPS_SEQ1_PATH)
        expected = {
            "x_data": np.array([-0.1, -0.09, -0.075, 1.07, 1.085, 1.1]),
            "y_data": np.array([17.90753, 17.76527, 17.40059, -29.98907, -29.98892, -29.98761]),
            "x_quantity": "voltage",
            "y_quantity": "current density",
            "x_unit": "V",
            "y_unit": "mA/cm^2",
            "filename": "SBTPS_SEQ1",
            "signalname": "Forward (Sweep 000 (mA/cm≤))",
            "z_dict": {},
        }
        self.assert_signal(data["Forward"][0], expected)
        expected = {
            "x_data": np.array([-0.1, -0.085, -0.07, 1.08, 1.095, 1.1]),
            "y_data": np.array([19.53833, 17.73079, 16.84296, -29.98825, -29.98798, -29.9876]),
            "x_quantity": "voltage",
            "y_quantity": "current density",
            "x_unit": "V",
            "y_unit": "mA/cm^2",
            "filename": "SBTPS_SEQ1",
            "signalname": "Backward (Sweep 001 (mA/cm≤))",
            "z_dict": {},
        }
        self.assert_signal(data["Reverse"][-1], expected)

        data = SbtpsSeqFile(resources.SBTPS_SEQ2_PATH)
        expected = {
            "x_data": np.array([-0.2, -0.18, -0.16, 1.06, 1.08, 1.1]),
            "y_data": np.array([0.1799218, 0.1585494, 0.1367925, -9.380523, -9.624294, -8.67218]),
            "x_quantity": "voltage",
            "y_quantity": "current density",
            "x_unit": "V",
            "y_unit": "mA/cm^2",
            "filename": "SBTPS_SEQ2",
            "signalname": "Forward (Sweep 000 (mA/cm≤))",
            "z_dict": {},
        }
        self.assert_signal(data["Forward"][0], expected)
        expected = {
            "x_data": np.array([-0.2, -0.18, -0.16, 1.06, 1.08, 1.1]),
            "y_data": np.array([23.55118, 23.51139, 23.51163, -20.80241, -22.13996, -22.86284]),
            "x_quantity": "voltage",
            "y_quantity": "current density",
            "x_unit": "V",
            "y_unit": "mA/cm^2",
            "filename": "SBTPS_SEQ2",
            "signalname": "Backward (Sweep 002 (mA/cm≤))",
            "z_dict": {},
        }
        self.assert_signal(data["Reverse"][0], expected)

        data = SbtpsSeqFile(resources.SBTPS_SEQ3_PATH)
        expected = {
            "x_data": np.array([-0.1, -0.1, -0.08, 1.14, 1.16, 1.18]),
            "y_data": np.array([1.661264e-02, 1.371791e-02, 1.646814e-02, -1.613068e01, -1.754159e01, -1.808632e01]),
            "x_quantity": "voltage",
            "y_quantity": "current density",
            "x_unit": "V",
            "y_unit": "mA/cm^2",
            "filename": "SBTPS_SEQ3",
            "signalname": "Forward (Sweep 000 (mA/cm))",
            "z_dict": {},
        }
        self.assert_signal(data["Forward"][0], expected)
        expected = {
            "x_data": np.array([-0.1, -0.08, -0.06, 1.16, 1.18, float("nan")]),
            "y_data": np.array([15.07615, 13.65758, 13.56461, -19.03928, -21.11079, float("nan")]),
            "x_quantity": "voltage",
            "y_quantity": "current density",
            "x_unit": "V",
            "y_unit": "mA/cm^2",
            "filename": "SBTPS_SEQ3",
            "signalname": "Backward (Sweep 002 (mA/cm))",
            "z_dict": {},
        }
        self.assert_signal(data["Reverse"][0], expected)

    def test_sbtps_iv(self) -> None:
        data = SbtpsIvFile(resources.SBTPS_IV1_PATH)
        expected = {
            "x_data": np.array([-0.1, -0.085, -0.07, 1.08, 1.095, 1.1]),
            "y_data": np.array([18.99143, 17.11235, 16.38189, -27.84484, -29.17089, -29.87343]),
            "x_quantity": "voltage",
            "y_quantity": "current density",
            "x_unit": "V",
            "y_unit": "mA/cm^2",
            "filename": "SBTPS_IV1",
            "signalname": "Current density",
            "z_dict": {},
        }
        self.assert_signal(data["Current density"], expected)
        expected = {
            "x_data": np.array([-0.1, -0.085, -0.07, 1.08, 1.095, 1.1]),
            "y_data": np.array([9.54, 9.65, 9.76, 19.02, 19.14, 19.25]),
            "x_quantity": "voltage",
            "y_quantity": "time",
            "x_unit": "V",
            "y_unit": "s",
            "filename": "SBTPS_IV1",
            "signalname": "Time",
            "z_dict": {},
        }
        self.assert_signal(data["Time"], expected)

        data = SbtpsIvFile(resources.SBTPS_IV2_PATH)
        expected = {
            "x_data": np.array([0.4895, 0.4895, 0.4895, 0.4895, 0.4895, 0.4895]),
            "y_data": np.array([21.1545, 19.65363, 19.4841, 16.92228, 16.92511, 16.9125]),
            "x_quantity": "voltage",
            "y_quantity": "current density",
            "x_unit": "V",
            "y_unit": "mA/cm^2",
            "filename": "SBTPS_IV2",
            "signalname": "Current density",
            "z_dict": {},
        }
        self.assert_signal(data["Current density"], expected)

        data = SbtpsIvFile(resources.SBTPS_IV3_PATH)
        expected = {
            "x_data": np.array([-0.1, -0.1, -0.08, 1.14, 1.16, 1.18]),
            "y_data": np.array([1.661264e-02, 1.371791e-02, 1.646814e-02, -1.613068e01, -1.754159e01, -1.808632e01]),
            "x_quantity": "voltage",
            "y_quantity": "current density",
            "x_unit": "V",
            "y_unit": "mA/cm^2",
            "filename": "SBTPS_IV3",
            "signalname": "Current density",
            "z_dict": {},
        }
        self.assert_signal(data["Current density"], expected)

    def test_simple_data(self) -> None:
        data = SimpleDataFile(resources.SIMPLE_TAB_PATH)
        expected = {
            "x_data": np.array([340.07, 340.45, 340.82, 1028.37, 1028.66, 1028.94]),
            "y_data": np.array([768.57, 768.57, 768.57, 818.06, 809.32, 819.71]),
            "x_quantity": "X-quantity",
            "y_quantity": "Y-quantity",
            "x_unit": "X-unit",
            "y_unit": "Y-unit",
            "filename": "Simple_tab",
            "signalname": "",
            "z_dict": {},
        }
        self.assert_signal(data[0], expected)

        data = SimpleDataFile(resources.SIMPLE_SEMICOLON_PATH, ";")
        expected = {
            "x_data": np.array([800.14495197, 802.07594206, 804.9724272, 1786.78433957, 1792.19111183, 1797.59788408]),
            "y_data": np.array([0.12535344, 0.09868099, 0.1913995, 0.08115339, 0.07994678, 0.07543786]),
            "x_quantity": "X-quantity",
            "y_quantity": "Y-quantity",
            "x_unit": "X-unit",
            "y_unit": "Y-unit",
            "filename": "Simple_semicolon",
            "signalname": "",
            "z_dict": {},
        }
        self.assert_signal(data[0], expected)

    def test_spectrasuit(self) -> None:
        data = SpectraSuiteFile(resources.SPECTRASUITE_HEADER_PATH)
        expected = {
            "x_data": np.array([340.07, 340.45, 340.82, 1028.37, 1028.66, 1028.94]),
            "y_data": np.array([776.36, 776.36, 776.36, 833.99, 829.02, 829.92]),
            "x_quantity": "wavelength",
            "y_quantity": "intensity",
            "x_unit": "nm",
            "y_unit": "counts",
            "filename": "SpectraSuite_Header",
            "signalname": "",
            "z_dict": {
                "Date & Time": Dimension(dt.datetime(2016, 11, 16, 18, 7, 25), "time", ""),
                "Integration Time": Dimension(0.1, "time", "s"),
            },
        }
        self.assert_signal(data, expected)

        data = SpectraSuiteFile(resources.SPECTRASUITE_HEADER_BST_PATH)
        expected = {
            "x_data": np.array([340.07, 340.45, 340.82, 1028.37, 1028.66, 1028.94]),
            "y_data": np.array([732.18, 732.18, 732.18, 973.99, 1003.36, 1089.24]),
            "x_quantity": "wavelength",
            "y_quantity": "intensity",
            "x_unit": "nm",
            "y_unit": "counts",
            "filename": "SpectraSuite_Header_BST",
            "signalname": "",
            "z_dict": {
                "Date & Time": Dimension(dt.datetime(2016, 10, 21, 10, 57, 53), "time", ""),
                "Integration Time": Dimension(2.5, "time", "s"),
            },
        }
        self.assert_signal(data, expected)

    def test_perkinelmer(self) -> None:

        data = PerkinElmerFile(resources.SPECTRUM_PATH)
        expected = {
            "x_data": np.array([650.0, 650.5, 651.0, 3999.0, 3999.5, 4000.0]),
            "y_data": np.array([80.18, 80.25, 80.27, 96.38, 96.37, 96.36]),
            "x_quantity": "wavenumber",
            "y_quantity": "transmittance",
            "x_unit": "cm^-1",
            "y_unit": "%",
            "filename": "spectrum",
            "signalname": "",
            "z_dict": {},
        }
        self.assert_signal(data, expected)

        data = PerkinElmerFile(resources.SPECTRUM_MULTIPLE_PATH)
        expected = {
            "x_data": np.array([450.0, 451.0, 452.0, 1048.0, 1049.0, 1050.0]),
            "y_data": np.array([0.0168, 0.0289, 0.0151, 0.116, 0.117, 0.118]),
            "x_quantity": "wavenumber",
            "y_quantity": "Y-quantity",
            "x_unit": "cm^-1",
            "y_unit": "Y-unit",
            "filename": "spectrum_multiple",
            "signalname": "0",
            "z_dict": {},
        }
        self.assert_signal(data[0], expected)

        data = PerkinElmerFile(resources.UVWINLAB_PATH)
        expected = {
            "x_data": np.array([200.0, 201.0, 202.0, 698.0, 699.0, 700.0]),
            "y_data": np.array([1.00000e01, 2.74454e-01, 1.74635e-01, 7.53000e-04, 4.97000e-04, 4.65000e-04]),
            "x_quantity": "wavelength",
            "y_quantity": "absorbance",
            "x_unit": "nm",
            "y_unit": "",
            "filename": "UVWinlab",
            "signalname": "",
            "z_dict": {},
        }
        self.assert_signal(data, expected)

    def test_uvwinlab_ascii(self) -> None:
        data = UVWinLabASCII(resources.UVWINLAB_ASCII_PATH)
        expected = {
            "x_data": np.array([390.0, 391.0, 392.0, 418.0, 419.0, 420.0]),
            "y_data": np.array([26.218752, 26.132979, 26.056731, 23.728336, 23.64869, 23.606037]),
            "x_quantity": "wavelength",
            "y_quantity": "reflectance",
            "x_unit": "nm",
            "y_unit": "%",
            "filename": "UVWinlab_ASCII",
            "signalname": "",
            "z_dict": {
                "Date & Time": Dimension(dt.datetime(2027, 5, 22, 11, 57, 32), "time", ""),
            },
        }
        self.assert_signal(data, expected)

    def test_vesta(self) -> None:
        data = VestaDiffractionFile(resources.VESTA_PATH)
        expected = {
            "x_data": np.array([1.0, 1.01, 1.02, 119.97, 119.98, 119.99]),
            "y_data": np.array([0.11191, 0.10984, 0.10782, 0.0693, 0.07054, 0.07185]),
            "x_quantity": "2 theta",
            "y_quantity": "intensity",
            "x_unit": "deg",
            "y_unit": "a.u.",
            "filename": "Vesta",
            "signalname": "",
            "z_dict": {},
        }
        self.assert_signal(data, expected)

    def test_wire(self) -> None:

        data = WireFile(resources.WIRE1_PATH)
        expected = {
            "x_data": np.array([794.19726562, 795.4375, 796.67578125, 1940.97460938, 1942.00976562, 1943.04492188]),
            "y_data": np.array(
                [84046.1640625, 84312.3671875, 84407.1796875, 130683.1796875, 130495.21875, 129160.390625]
            ),
            "x_quantity": "wavenumber",
            "y_quantity": "intensity",
            "x_unit": "cm^-1",
            "y_unit": "",
            "filename": "Single scan measurement 4",
            "signalname": "",
            "z_dict": {},
        }
        self.assert_signal(data, expected)

        data = WireFile(resources.WIRE2_PATH)
        expected = {
            "x_data": np.array([830.99182129, 832.17248535, 833.35412598, 1897.29064941, 1898.23693848, 1899.18322754]),
            "y_data": np.array([28863.01171875, 28688.14453125, 28792.6875, 21222.15625, 20938.0703125, 20763.609375]),
            "x_quantity": "wavenumber",
            "y_quantity": "intensity",
            "x_unit": "cm^-1",
            "y_unit": "",
            "filename": "Single scan measurement 3",
            "signalname": "",
            "z_dict": {},
        }
        self.assert_signal(data, expected)

    def test_zem3(self) -> None:
        data = Zem3(resources.ZEM3_PATH)
        expected = {
            "x_data": np.array([44.16055, 59.66502, 77.633, 248.0079, 267.0932, 286.0288]),
            "y_data": np.array([3.276580e-06, 3.357355e-06, 3.403639e-06, 3.281571e-06, 3.180939e-06, 5.350202e-06]),
            "x_quantity": "Measurement temp.(C)",
            "y_quantity": "Resistivity(Ohm m)",
            "x_unit": "",
            "y_unit": "",
            "filename": "Zem3",
            "signalname": "",
            "z_dict": {},
        }
        self.assert_signal(data["Resistivity(Ohm m)"], expected)

        data = Zem3(resources.ZEM3_TXT_PATH)
        expected = {
            "x_data": np.array([39.44201, 42.1815, 44.11031, 283.6468, 286.4197, 288.0198]),
            "y_data": np.array([-0.3088848, 0.8997093, 2.302329, -0.7177747, 0.3782233, 1.430436]),
            "x_quantity": "Sample Temp.(C)",
            "y_quantity": "Delta Temp(C)",
            "x_unit": "",
            "y_unit": "",
            "filename": "Zem3",
            "signalname": "",
            "z_dict": {},
        }
        self.assert_signal(data["Delta Temp(C)"], expected)


def test_detect_file_type() -> None:
    """Check that the test files are properly detected"""

    for path in resources.FILE_TYPE_DICT:
        assert detect_file_type(path)[1] == resources.FILE_TYPE_DICT[path]


class TestReadDataFile:

    def test_single_file_detect(self) -> None:

        signal, filetype = read_data_file(resources.SPECTRASUITE_HEADER_PATH, FILETYPES[0])
        assert filetype == "SpectraSuite (.txt)"

    def test_multiple_files_detect(self) -> None:

        filenames = [resources.SPECTRASUITE_HEADER_PATH, resources.SPECTRASUITE_HEADER_BST_PATH]
        signal, filetype = read_data_file(filenames, FILETYPES[0])
        assert filetype == "SpectraSuite (.txt)"
        assert len(signal) == 2

    def test_multiple_files_different_detect(self) -> None:

        filenames = [resources.SPECTRASUITE_HEADER_PATH, resources.EASYLOG_PATH, resources.SPECTRASUITE_HEADER_BST_PATH]
        signal, filetype = read_data_file(filenames, FILETYPES[0])
        assert filetype == "SpectraSuite (.txt)"
        assert len(signal) == 2

    def test_single_file_fixed(self) -> None:

        signal, filetype = read_data_file(resources.SPECTRASUITE_HEADER_PATH, "SpectraSuite (.txt)")
        assert filetype == "SpectraSuite (.txt)"

    def test_multiple_files_fixed(self) -> None:

        filenames = [resources.SPECTRASUITE_HEADER_PATH, resources.SPECTRASUITE_HEADER_BST_PATH]
        signal, filetype = read_data_file(filenames, "SpectraSuite (.txt)")
        assert filetype == "SpectraSuite (.txt)"
        assert len(signal) == 2

    def test_multiple_files_different_fixed(self) -> None:

        filenames = [resources.SPECTRASUITE_HEADER_PATH, resources.SPECTRASUITE_HEADER_BST_PATH, resources.EASYLOG_PATH]
        signal, filetype = read_data_file(filenames, "SpectraSuite (.txt)")
        assert filetype == "SpectraSuite (.txt)"
        assert len(signal) == 2

    def test_single_file_fixed_incorrect(self) -> None:

        signal, filetype = read_data_file(resources.SPECTRASUITE_HEADER_PATH, "EasyLog (.txt)")
        assert filetype is None

    def test_multiple_files_fixed_incorrect(self) -> None:

        filenames = [resources.SPECTRASUITE_HEADER_PATH, resources.SPECTRASUITE_HEADER_BST_PATH]
        signal, filetype = read_data_file(filenames, "EasyLog (.txt)")
        assert filetype is None
        assert len(signal) == 0

    def test_multiple_files_different_fixed_incorrect(self) -> None:

        filenames = [resources.SPECTRASUITE_HEADER_PATH, resources.SPECTRASUITE_HEADER_BST_PATH, resources.EASYLOG_PATH]
        signal, filetype = read_data_file(filenames, "EasyLog (.txt)")
        assert filetype is None
        assert len(signal) == 0
