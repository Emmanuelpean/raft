"""Test module for the functions in the `data_files/signal.py` module.

This module contains unit tests for the functions implemented in the `data_files/signal.py` module. The purpose of
these tests is to ensure the correct functionality of each function in different scenarios and to validate that the
expected outputs are returned.

Tests should cover various edge cases, valid inputs, and any other conditions that are necessary to confirm the
robustness of the functions."""

import numpy as np
import plotly.graph_objects as go
import pytest
import datetime as dt

from data_files.signal_data import Dimension, SignalData, average_signals, get_z_dim
from data_processing.fitting import gaussian
from utils.checks import are_identical, are_close


class TestDimension:
    """Test class for the Dimension class"""

    def test_init(self) -> None:
        """Test initialization of Dimension object"""

        # Test with basic data
        dim = Dimension(5, "length", "m")
        assert dim.data == 5
        assert dim.quantity == "length"
        assert dim.unit == "m"

        # Test with array data
        data = np.array([1, 2, 3])
        dim = Dimension(data, "time", "s")
        assert are_identical(dim.data, data)
        assert dim.quantity == "time"
        assert dim.unit == "s"

        # Test with default values
        dim = Dimension(10)
        assert dim.data == 10
        assert dim.quantity == ""
        assert dim.unit == ""

    def test_call(self) -> None:
        """Test the __call__ method for creating new instances with modified attributes"""

        # Create base dimension
        base_dim = Dimension(5, "length", "m")

        # Test modifying data
        new_dim = base_dim(10)
        assert new_dim.data == 10
        assert new_dim.quantity == "length"
        assert new_dim.unit == "m"

        # Test modifying quantity
        new_dim = base_dim(quantity="mass")
        assert new_dim.data == 5
        assert new_dim.quantity == "mass"
        assert new_dim.unit == "m"

        # Test modifying unit
        new_dim = base_dim(unit="km")
        assert new_dim.data == 5
        assert new_dim.quantity == "length"
        assert new_dim.unit == "km"

        # Test modifying multiple attributes
        new_dim = base_dim(10, "mass", "kg")
        assert new_dim.data == 10
        assert new_dim.quantity == "mass"
        assert new_dim.unit == "kg"

        # Test modifying multiple attributes mixed case
        new_dim = base_dim(10, quantity="mass", unit="kg")
        assert new_dim.data == 10
        assert new_dim.quantity == "mass"
        assert new_dim.unit == "kg"

    def test_getitem(self) -> None:
        """Test the __getitem__ special method"""

        base_dim = Dimension(np.array([1, 2, 4, 5]), "length", "m")
        assert base_dim[1].data == 2
        assert base_dim[-1].data == 5

    def test_get_quantity_label_html(self) -> None:
        """Test get_quantity_label_html method"""

        # Test with multi-character quantity
        dim = Dimension(5, "length", "m")
        assert dim.get_quantity_label_html() == "Length"

        # Test with special quantity
        dim = Dimension(5, r"\lambda", "m")
        assert dim.get_quantity_label_html() == r"\lambda"

        # Test with single character quantity
        dim = Dimension(5, "x", "m")
        assert dim.get_quantity_label_html() == "x"

        # Test with empty quantity
        dim = Dimension(5)
        assert dim.get_quantity_label_html() == ""

    def test_get_unit_label_html(self) -> None:
        """Test get_unit_label_html method"""

        # Test with unit
        dim = Dimension(5, "length", "m")
        assert dim.get_unit_label_html() == "m"

        # Test with empty unit
        dim = Dimension(5, "length")
        assert dim.get_unit_label_html() == ""

    def test_get_label_html(self) -> None:
        """Test get_axis_label_html method"""

        # Test with both quantity and unit
        dim = Dimension(5, "length", "m")
        assert dim.get_label_html() == "Length (m)"

        # Test with no unit
        dim = Dimension(5, "length")
        assert dim.get_label_html() == "Length"

    def test_get_label_raw(self) -> None:
        """Test get_label_raw method"""

        # Test with both quantity and unit
        dim = Dimension(5, "length", "m")
        assert dim.get_label_raw() == "length (m)"

        # Test with only quantity
        dim = Dimension(5, "length")
        assert dim.get_label_raw() == "length"

    def test_get_value_label_html(self) -> None:
        """Test get_value_label_html method"""

        # Test with scalar value
        dim = Dimension(5, "length", "m")
        assert dim.get_value_label_html(5, "g") == "Length: 5 m"
        assert dim.get_value_label_html(5, "f") == "Length: 5.00000 m"

        # Test without unit
        dim = Dimension(5, "length")
        assert dim.get_value_label_html() == "Length: 5.00000"

        # Test with high scalar value
        dim = Dimension(1.423235e5, "length", "m")
        assert dim.get_value_label_html(10, "g") == "Length: 1.423235E5 m"
        assert dim.get_value_label_html(5, "g", True) == "Length: 1.4232 &#10005; 10<sup>5</sup> m"
        assert dim.get_value_label_html(10, "f") == "Length: 1.4232350000E5 m"
        assert dim.get_value_label_html(10, "f", True) == "Length: 1.4232350000 &#10005; 10<sup>5</sup> m"

        # Test incorrect
        with pytest.raises(AssertionError):
            Dimension(np.array([1, 2, 4, 5]), "length", "m").get_value_label_html()

    def test_repr(self) -> None:
        """Test __repr__ method"""

        # Test with all attributes
        dim = Dimension(5, "length", "m")
        assert repr(dim) == "Dimension(5, length, m)"

        # Test with only data
        dim = Dimension(5)
        assert repr(dim) == "Dimension(5)"

        # Test with data and quantity
        dim = Dimension(5, "length")
        assert repr(dim) == "Dimension(5, length)"


class TestSignalData:
    """Test class for the SignalData class"""

    @staticmethod
    def sample_dimensions() -> tuple[Dimension, Dimension]:
        """Fixture that returns sample x and y dimensions for testing"""

        x_data = np.linspace(0, 10, 21)
        y_data = np.exp(-((x_data - 3) ** 2) / 2)
        x_dim = Dimension(x_data, "time", "s")
        y_dim = Dimension(y_data, "amplitude", "V")
        return x_dim, y_dim

    def sample_signal(self, value=None) -> SignalData:
        """Fixture that returns a sample SignalData object for testing"""

        x_dim, y_dim = self.sample_dimensions()
        if value is not None:
            y_dim.data[-1] = value
        return SignalData(x_dim, y_dim, "Test Signal", "TS", {"param1": "value1"})

    def test_init(self) -> None:
        """Test initialization of SignalData object"""

        x_dim, y_dim = self.sample_dimensions()

        # Test with all parameters
        signal = SignalData(x_dim, y_dim, "Test Signal", "TS", {"param1": "value1"})
        assert signal.x == x_dim
        assert signal.y == y_dim
        assert signal.filename == "Test Signal"
        assert signal.signalnames == ["TS"]
        assert signal.z_dict == {"param1": "value1"}

        # Test with default parameters
        signal = SignalData(x_dim, y_dim)
        assert signal.x == x_dim
        assert signal.y == y_dim
        assert signal.filename == ""
        assert signal.signalnames == [""]
        assert signal.z_dict == {}

        # Test with list of dims
        dims = [Dimension(5), Dimension(5), Dimension(7)]
        signal = SignalData(dims, dims)
        assert are_identical(signal.x.data, np.array([5, 5, 7]))
        assert are_identical(signal.y.data, np.array([5, 5, 7]))

    def test_get_name(self) -> None:
        """Test get_name method"""

        sample_signal = self.sample_signal()

        # With filename=True
        assert sample_signal.get_name(True) == "Test Signal: TS"

        # With filename=False
        assert sample_signal.get_name(False) == "TS"

        # When signalname is empty
        sample_signal.signalnames = ""
        assert sample_signal.get_name(True) == "Test Signal"
        assert sample_signal.get_name(False) == "Test Signal"

    def test_plot(self) -> None:
        """Test plot method"""

        # Regular plot
        sample_signal = self.sample_signal()
        figure = sample_signal.plot()
        trace = figure.data[0]
        assert isinstance(trace, go.Scattergl)
        assert are_identical(trace.x, sample_signal.x.data)
        assert are_identical(trace.y, sample_signal.y.data)
        assert trace.name == sample_signal.get_name(False)
        assert trace.showlegend is None

        # With datetime
        x_dims = np.array([dt.datetime(2025, 5, 5), dt.datetime(2025, 5, 7)])
        y_dims = np.array([1, 2])
        signal = SignalData(Dimension(x_dims), Dimension(y_dims))
        figure = signal.plot()
        assert figure.layout.xaxis.tickformat is None

        # Secondary axis
        figure = sample_signal.plot(secondary_y=True)
        assert "yaxis2" in figure.layout

    # ---------------------------------------------- SIGNAL TRANSFORMATION ---------------------------------------------

    def test_remove_background(self) -> None:
        """Test remove_background method"""

        sample_dimensions = self.sample_dimensions()
        y = sample_dimensions[1](sample_dimensions[1].data + 3)
        signal = SignalData(sample_dimensions[0], y, "Test Signal")
        new_signal = signal.remove_background([0, 1])
        assert are_close(new_signal.y.data, signal.y.data - 3.02752297)

        # Nan value
        y = sample_dimensions[1](sample_dimensions[1].data * float("nan"))
        signal = SignalData(sample_dimensions[0], y, "Test Signal")
        with pytest.raises(AssertionError):
            signal.remove_background([0, 1])

    def test_smooth(self) -> None:
        """Test smooth method"""

        sample_signal = self.sample_signal()
        smoothed = sample_signal.smooth(5, 3)
        assert isinstance(smoothed, SignalData)
        assert are_identical(smoothed.x.data, sample_signal.x.data)
        assert are_close(smoothed.y.data[:3], np.array([0.01174766, 0.04138229, 0.13916725]))
        assert smoothed.filename == sample_signal.filename
        assert smoothed.signalnames == ["TS", "Smoothed"]
        assert smoothed.z_dict == sample_signal.z_dict

    def test_reduce_range(self) -> None:
        """Test reduce_range method"""

        # Set up test range
        x_range = [2.0, 8.0]

        # Find expected indices
        index1 = 4
        index2 = 16

        # Call reduce_range
        sample_signal = self.sample_signal()
        reduced = sample_signal.reduce_range(x_range)

        # Verify results
        assert isinstance(reduced, SignalData)
        assert are_identical(reduced.x.data, sample_signal.x.data[index1:index2])
        assert are_identical(reduced.y.data, sample_signal.y.data[index1:index2])
        assert reduced.filename == sample_signal.filename
        assert reduced.signalnames == sample_signal.signalnames
        assert reduced.z_dict == sample_signal.z_dict

    def test_interpolate(self) -> None:

        sample_signal = self.sample_signal()
        signal = sample_signal.interpolate(3.0)
        assert are_close(signal.x.data, [0.0, 3.0, 6.0, 9.0])
        assert are_close(signal.y.data, [1.11089965e-02, 1.00000000e00, 1.11089965e-02, 1.52299797e-08])

    def test_derive(self) -> None:

        sample_signal = self.sample_signal()
        signal = sample_signal.derive()
        assert are_close(signal.x.data[:3], [0.25, 0.75, 1.25])
        assert are_close(signal.y.data[:3], [6.56558742e-02, 1.82796699e-01, 3.78634368e-01])

    def test_normalise(self) -> None:

        sample_signal = self.sample_signal()
        x_dim, y_dim = self.sample_dimensions()
        y_dim = y_dim(y_dim.data * 5 + 3)
        signal = SignalData(x_dim, y_dim)

        # With respect to itself
        norm_signal = signal.normalise()
        assert are_close(norm_signal.x.data, signal.x.data)
        assert are_close(norm_signal.y.data[:3], [0.38194312, 0.40246058, 0.45958455])
        assert are_close(norm_signal.y.data.max(), 1.0)

        # With respect to another
        norm_signal = signal.normalise(sample_signal.y.data)
        assert are_close(norm_signal.x.data, signal.x.data)
        assert are_close(norm_signal.y.data[:3], [3.05554498, 3.21968467, 3.67667642])
        assert are_close(norm_signal.y.data.max(), 8.0)

    def test_feature_scale(self) -> None:

        sample_signal = self.sample_signal()
        x_dim, y_dim = self.sample_dimensions()
        y_dim = y_dim(y_dim.data * 5 + 3)
        signal = SignalData(x_dim, y_dim)

        # With respect to itself
        norm_signal = signal.feature_scale(7, 2)
        assert are_close(norm_signal.x.data, signal.x.data)
        assert are_close(norm_signal.y.data[:3], [2.05554498, 2.21968467, 2.67667642])
        assert are_close(norm_signal.y.data.max(), 7.0)
        assert are_close(norm_signal.y.data.min(), 2.0)

        # With respect to other
        norm_signal = signal.feature_scale(7, 2, sample_signal.y.data)
        assert are_close(norm_signal.x.data, signal.x.data)
        assert are_close(norm_signal.y.data[:3], [17.27772491, 18.09842334, 20.38338208])
        assert are_close(norm_signal.y.data.max(), 42.0)
        assert are_close(norm_signal.y.data.min(), 17.0)

    def test_fit(self) -> None:

        sample_signal = self.sample_signal()
        fit = sample_signal.fit(gaussian, dict(a=4, mu=5, sigma=3, c=4))
        assert are_close(fit[1], [1.00000000e00, 3.00000000e00, 1.00000000e00, -1.90398713e-11])
        assert are_close(fit[2], [1.21468176e-11, 1.40104267e-11, 1.40703837e-11, 3.39731184e-12])
        assert are_close(fit[3], 1.0)
        assert are_close(fit[0].x.data, self.sample_dimensions()[0].data)
        assert are_close(fit[0].y.data, self.sample_dimensions()[1].data)

    # ------------------------------------------------- DATA EXTRACTION ------------------------------------------------

    def test_get_point(self) -> None:
        """Test _get_point method"""

        sample_signal = self.sample_signal()

        # Test min point
        x_min, y_min, i_min = sample_signal._get_point("min")
        min_index = sample_signal.y.data.argmin()
        assert x_min.data == sample_signal.x.data[min_index]
        assert x_min.quantity == "min. " + sample_signal.x.quantity
        assert x_min.unit == sample_signal.x.unit
        assert y_min.data == sample_signal.y.data[min_index]
        assert y_min.quantity == "min. " + sample_signal.y.quantity
        assert y_min.unit == sample_signal.y.unit
        assert i_min.data == min_index

        # Test max point
        x_max, y_max, i_max = sample_signal._get_point("max")
        max_index = sample_signal.y.data.argmax()
        assert x_max.data == sample_signal.x.data[max_index]
        assert x_max.quantity == "max. " + sample_signal.x.quantity
        assert x_max.unit == sample_signal.x.unit
        assert y_max.data == sample_signal.y.data[max_index]
        assert y_max.quantity == "max. " + sample_signal.y.quantity
        assert y_max.unit == sample_signal.y.unit
        assert i_max.data == max_index

        # Test max point with no left of the peak
        x_dim, y_dim = self.sample_dimensions()
        signal = SignalData(x_dim, x_dim)
        x_max, y_max, i_max = signal._get_point("max", interpolation=True)
        assert np.isnan(x_max.data)

        # Test max point with successful interpolation
        x_max, y_max, i_max = sample_signal._get_point("max", interpolation=True)
        assert are_close(x_max.data, 3.0)
        assert are_close(y_max.data, 1.0)

    def test_get_max(self) -> None:
        """Test get_max method"""

        sample_signal = self.sample_signal()
        result = sample_signal.get_max()
        assert are_close(result[0].data, 3.0)
        assert result[0].quantity == "max. time"
        assert result[0].unit == "s"
        assert are_close(result[1].data, 1.0)
        assert result[1].quantity == "max. amplitude"
        assert result[1].unit == "V"
        assert are_close(result[2].data, 6)
        assert result[2].quantity == ""
        assert result[2].unit == ""

        signal2 = self.sample_signal(float("nan"))
        result = signal2.get_max()
        assert are_close(result[0].data, 3.0)
        assert result[0].quantity == "max. time"
        assert result[0].unit == "s"
        assert are_close(result[1].data, 1.0)
        assert result[1].quantity == "max. amplitude"
        assert result[1].unit == "V"
        assert are_close(result[2].data, 6)
        assert result[2].quantity == ""
        assert result[2].unit == ""

    def test_get_min(self) -> None:
        """Test get_min method"""

        sample_signal = self.sample_signal()
        result = sample_signal.get_min()
        assert are_close(result[0].data, 10.0)
        assert result[0].quantity == "min. time"
        assert result[0].unit == "s"
        assert are_close(result[1].data, 2.289734845645553e-11)
        assert result[1].quantity == "min. amplitude"
        assert result[1].unit == "V"
        assert are_close(result[2].data, 20)
        assert result[2].quantity == ""
        assert result[2].unit == ""

    def test_get_extrema(self) -> None:
        """Test get_extrema method"""

        sample_signal = self.sample_signal()
        result = sample_signal.get_extrema()
        assert are_close(result[0].data, 3.0)
        assert result[0].quantity == "extremum time"
        assert result[0].unit == "s"
        assert are_close(result[1].data, 1.0)
        assert result[1].quantity == "extremum amplitude"
        assert result[1].unit == "V"
        assert are_close(result[2].data, 6)
        assert result[2].quantity == ""
        assert result[2].unit == ""

        x_dim, y_dim = self.sample_dimensions()
        signal = SignalData(x_dim, Dimension(-y_dim.data))
        result = signal.get_extrema()
        assert are_close(result[0].data, 3.0)
        assert result[0].quantity == "extremum time"
        assert result[0].unit == "s"
        assert are_close(result[1].data, -1.0)
        assert result[1].quantity == "extremum Y-quantity"
        assert result[1].unit == "Y-unit"
        assert are_close(result[2].data, 6)
        assert result[2].quantity == ""
        assert result[2].unit == ""

    def test_get_fwhm(self) -> None:
        """Test get_fwhm method"""

        sample_signal = self.sample_signal()
        fwhm_dim, x_left, y_left, x_right, y_right = sample_signal.get_fwhm()
        assert are_close(fwhm_dim.data, 2.0)
        assert fwhm_dim.quantity == "fwhm"
        assert are_close(x_left.data, 2.0)
        assert are_close(y_left.data, 0.6065306597126334)
        assert are_close(x_right.data, 4.0)
        assert are_close(y_right.data, 0.6065306597126334)

        # With interpolation
        fwhm_dim, x_left, y_left, x_right, y_right = sample_signal.get_fwhm(True)
        assert are_close(fwhm_dim.data, 2.3683683683683685)
        assert fwhm_dim.quantity == "fwhm"
        assert are_close(x_left.data, 1.820820820820821)
        assert are_close(y_left.data, 0.5055172534435308)
        assert are_close(x_right.data, 4.1891891891891895)
        assert are_close(y_right.data, 0.4998740463893908)

        # Test edge case: m == 0
        ext_i = Dimension(np.array([1, 0]))
        signal = SignalData(ext_i, ext_i)
        fwhm_dim, x_left, y_left, x_right, y_right = signal.get_fwhm()
        assert np.isnan(fwhm_dim.data)
        assert np.isnan(x_left.data)
        assert np.isnan(y_left.data)
        assert x_right.data == 1
        assert y_right.data == 1

        # Test edge case: m == 0
        ext_i = Dimension(np.array([0, 1]))
        signal = SignalData(ext_i, ext_i)
        fwhm_dim, x_left, y_left, x_right, y_right = signal.get_fwhm()
        assert np.isnan(fwhm_dim.data)
        assert x_left.data == 0
        assert y_left.data == 0
        assert np.isnan(x_right.data)
        assert np.isnan(y_right.data)

    def test_get_half_int(self) -> None:

        sample_signal = self.sample_signal()

        # Test the function with a peak pointing up and no interpolation.
        x_half, y_half = sample_signal.get_halfint_point(sample_signal.x.data, sample_signal.y.data, False)
        assert are_close(x_half, 2.0)
        assert are_close(y_half, 0.606)

        # Test the function with a peak pointing down and no interpolation
        x_half, y_half = sample_signal.get_halfint_point(sample_signal.x.data, -sample_signal.y.data, False)
        assert are_close(x_half, 2.0)
        assert are_close(y_half, -0.606)

        # Test the function with flat data
        signal = SignalData(Dimension(np.linspace(0, 10, 21)), Dimension(np.ones(21)))
        x_half, y_half = signal.get_halfint_point(signal.x.data, signal.y.data, interpolation=False)
        assert are_close(x_half, 0.0)
        assert are_close(y_half, 1.0)

        # Test that interpolation is called correctly
        x_half, y_half = sample_signal.get_halfint_point(sample_signal.x.data, sample_signal.y.data, True)
        assert are_close(x_half, 1.810810810810811)
        assert are_close(y_half, 0.49987404638939104)


class TestAverageSignals:

    @pytest.fixture
    def sample_signal(self) -> tuple[list[np.ndarray], list[SignalData]]:
        """Sample SignalData object for testing"""

        x_data = np.linspace(0, 10, 21)
        ys_data = []
        signals = []
        for i in (0.2, 0.3, 0.4, 0.5):
            y_data = np.exp(-((x_data - 3) ** 2) / 2) + i
            ys_data.append(y_data)
            x_dim = Dimension(x_data, "time", "s")
            y_dim = Dimension(y_data, "amplitude", "V")
            signal = SignalData(x_dim, y_dim, "Test Signal", "TS", {"param1": "value1"})
            signals.append(signal)
        return ys_data, signals

    def test_low_n(self, sample_signal) -> None:

        ys_data, signals = sample_signal
        output = average_signals(signals, 2)
        assert len(output) == 2
        assert are_close(output[0].y.data, np.mean(ys_data[:2], axis=0))
        assert are_close(output[1].y.data, np.mean(ys_data[2:], axis=0))

    def test_non_matching_n(self, sample_signal) -> None:

        ys_data, signals = sample_signal
        output = average_signals(signals, 3)
        assert len(output) == 1
        assert are_close(output[0].y.data, np.mean(ys_data[:3], axis=0))

    def test_matching_n(self, sample_signal) -> None:

        ys_data, signals = sample_signal
        output = average_signals(signals, 4)
        assert len(output) == 1
        assert are_close(output[0].y.data, np.mean(ys_data, axis=0))

    def test_high_n(self, sample_signal) -> None:

        ys_data, signals = sample_signal
        output = average_signals(signals, 5)
        assert len(output) == 1
        assert are_close(output[0].y.data, np.mean(ys_data, axis=0))


class TestGetZDim:

    @pytest.fixture
    def sample_signal(self) -> list[SignalData]:
        """Sample SignalData object for testing"""

        x_data = np.linspace(0, 10, 21)
        signals = []
        for i in (2022, 2023, 2026, 2024, 2025):
            y_data = np.exp(-((x_data - 3) ** 2) / 2) + i
            x_dim = Dimension(x_data, "time", "s")
            y_dim = Dimension(y_data, "amplitude", "V")
            signal = SignalData(
                x_dim,
                y_dim,
                f"Test Signal {i}",
                "TS",
                {
                    "Date & Time": Dimension(dt.datetime(i, 1, 1)),
                    "Z": Dimension("f"),
                    "X": Dimension(f"{i}"),
                },
            )
            signals.append(signal)
        return signals

    def test_signal(self, sample_signal) -> None:
        """Input is a list of 1 signal"""

        output = get_z_dim([sample_signal[0]], "Filename", False)
        assert are_identical(output[0][-1].y.data, sample_signal[0].y.data)

    def test_filename(self, sample_signal) -> None:
        """Input is a list of signals"""

        output = get_z_dim(sample_signal, "Filename", False)
        assert are_identical(output[0][-1].y.data, sample_signal[2].y.data)

    def test_timestamp(self, sample_signal) -> None:
        """Input is a list of signals"""

        output = get_z_dim(sample_signal, "Date & Time", False)
        assert are_identical(output[0][-1].y.data, sample_signal[2].y.data)

    def test_timestamp_shift(self, sample_signal) -> None:
        """Input is a list of signals"""

        output = get_z_dim(sample_signal, "Date & Time", True)
        assert are_identical(output[0][-1].y.data, sample_signal[2].y.data)

    def test_conversion(self, sample_signal) -> None:
        """Input is a list of signals"""

        output = get_z_dim(sample_signal, "X")
        assert are_identical(output[1][-1].data, 2026.0)

    def test_fail(self, sample_signal) -> None:
        """Input is a list of signals"""

        output = get_z_dim(sample_signal, "Z")
        assert are_identical(output[1][-1].data, 5)
