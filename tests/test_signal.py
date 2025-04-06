"""Test module for the functions in the `signal.py` module.

This module contains unit tests for the functions implemented in the `signal.py` module. The purpose of these tests is to
ensure the correct functionality of each function in different scenarios and to validate that the expected outputs are
returned.

Tests should cover various edge cases, valid inputs, and any other conditions that are necessary to confirm the
robustness of the functions."""

import plotly.subplots as ps
import pytest

from app.signal_data import *
from app.utils import are_close, are_identical


class TestGetLabel:

    def setup_method(self) -> None:
        """Set the quantities_label and units_label dicts"""

        # Mock for pc.quantities_label dictionary based on examples
        self.quantities_label = {
            "max. wavelength": r"\lambda_{max}",
            "int. max.": "Int. max.",
        }

        # Mock for pc.units_label dictionary
        self.units_label = {"mum": r"\mum"}

    def test_raises_assertion_error_when_question_mark_in_string(self) -> None:
        with pytest.raises(AssertionError, match="\\? should not be in the string"):
            get_label("test?", {})

    def test_superscripts_and_subscripts(self) -> None:
        result = get_label("sub_sub^1 max. wavelength^2 sup^sup1 sup^-3", self.quantities_label)
        assert result == r"Sub<sub>sub</sub><sup>1</sup> \lambda_{max}<sup>2</sup> sup<sup>sup1</sup> sup<sup>-3</sup>"

    def test_multiple_dictionary_replacements(self) -> None:
        result = get_label("int. max. max. wavelength^2", self.quantities_label)
        assert result == r"Int. max. \lambda_{max}<sup>2</sup>"

    def test_no_capitalization(self) -> None:
        custom_dict = {"rap": r"\tau", "taureau": "t|rap"}
        result = get_label("taureau", custom_dict, False)
        assert result == "t rap"

    def test_with_units_label(self) -> None:
        result = get_label("test mum_test^2", self.units_label, False)
        assert result == r"test \mum<sub>test</sub><sup>2</sup>"

    def test_capitalize_true(self) -> None:
        result = get_label("test string", {}, True)
        assert result == "Test string"

    def test_capitalize_short_word(self) -> None:
        # Words with length 1 should not be capitalized even if capitalize=True
        result = get_label("a test string", {}, True)
        assert result == "A test string"

    def test_dictionary_replacement_conditions(self) -> None:
        # Test that replacements only occur when surrounded by whitespace, underscore, or caret
        test_dict = {"word": "REPLACED"}

        # Should be replaced
        assert "REPLACED" in get_label(" word ", test_dict)
        assert "REPLACED" in get_label("_word ", test_dict)
        assert "REPLACED" in get_label(" word_", test_dict)
        assert "REPLACED" in get_label("^word ", test_dict)
        assert "REPLACED" in get_label(" word^", test_dict)

        # Should not be replaced
        assert "REPLACED" not in get_label("aword", test_dict)
        assert "REPLACED" not in get_label("worda", test_dict)
        assert "REPLACED" not in get_label("awordb", test_dict)

    def test_sorted_dictionary_keys(self) -> None:
        # Test that longer keys are replaced first
        test_dict = {
            "short": "SHORT",
            "very long key": "LONG",
        }
        result = get_label("test very long key short test", test_dict)
        assert result == "Test LONG SHORT test"

    def test_pipe_replacement(self) -> None:
        test_dict = {"test": "replacement|with|pipes"}
        result = get_label("test", test_dict)
        assert result == "replacement with pipes"

    def test_complex_formatting(self) -> None:
        test_dict = {"complex": r"\lambda_{α}^{β}"}
        result = get_label("complex_value^2", test_dict)
        assert result == r"\lambda_{α}^{β}<sub>value</sub><sup>2</sup>"

    def test_empty_string(self) -> None:
        with pytest.raises(IndexError):
            get_label("", {})


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
        assert np.array_equal(dim.data, data)
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

    def test_get_value_label_html(self) -> None:
        """Test get_value_label_html method"""

        # Test with scalar value
        dim = Dimension(5, "length", "m")
        assert dim.get_value_label_html() == "Length: 5 m"

    def test_get_label_raw(self) -> None:
        """Test get_label_raw method"""

        # Test with both quantity and unit
        dim = Dimension(5, "length", "m")
        assert dim.get_label_raw() == "length (m)"

        # Test with only quantity
        dim = Dimension(5, "length")
        assert dim.get_label_raw() == "length"

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

    @pytest.fixture
    def sample_dimensions(self) -> tuple[Dimension, Dimension]:
        """Fixture that returns sample x and y dimensions for testing"""

        x_data = np.linspace(0, 10, 21)
        y_data = np.exp(- (x_data - 3) ** 2 / 2)
        x_dim = Dimension(x_data, "time", "s")
        y_dim = Dimension(y_data, "amplitude", "V")
        return x_dim, y_dim

    @pytest.fixture
    def sample_signal(self, sample_dimensions) -> SignalData:
        """Fixture that returns a sample SignalData object for testing"""

        x_dim, y_dim = sample_dimensions
        return SignalData(x_dim, y_dim, "Test Signal", "TS", {"param1": "value1"})

    def test_init(self, sample_dimensions) -> None:
        """Test initialization of SignalData object"""

        x_dim, y_dim = sample_dimensions

        # Test with all parameters
        signal = SignalData(x_dim, y_dim, "Test Signal", "TS", {"param1": "value1"})
        assert signal.x == x_dim
        assert signal.y == y_dim
        assert signal.name == "Test Signal"
        assert signal.shortname == "TS"
        assert signal.z_dict == {"param1": "value1"}

        # Test with default parameters
        signal = SignalData(x_dim, y_dim)
        assert signal.x == x_dim
        assert signal.y == y_dim
        assert signal.name == ""
        assert signal.shortname == ""
        assert signal.z_dict == {}

    def test_get_name(self, sample_signal) -> None:
        """Test get_name method"""

        # With condition=True
        assert sample_signal.get_name(True) == "Test Signal: TS"

        # With condition=False
        assert sample_signal.get_name(False) == "TS"

        # When shortname is empty
        sample_signal.shortname = ""
        assert sample_signal.get_name(True) == "Test Signal"
        assert sample_signal.get_name(False) == "Test Signal"

    def test_plot(self, sample_signal) -> None:
        """Test plot method"""

        # Test without position
        figure = sample_signal.plot()
        trace = figure.data[0]
        assert isinstance(trace, go.Scatter)
        assert np.array_equal(trace.x, sample_signal.x.data)
        assert np.array_equal(trace.y, sample_signal.y.data)
        assert trace.name == sample_signal.get_name(False)
        assert trace.showlegend is True

        # Test with position
        figure = ps.make_subplots(1, 2)
        sample_signal.plot(figure, [1, 2], True)
        trace = figure.data[0]
        # noinspection PyUnresolvedReferences
        assert trace.name == sample_signal.get_name(True)

    def test_smooth(self, sample_signal) -> None:
        """Test smooth method"""

        smoothed = sample_signal.smooth(5, 3)
        assert isinstance(smoothed, SignalData)
        assert are_identical(smoothed.x.data, sample_signal.x.data)
        assert are_close(smoothed.y.data[:3], np.array([0.01174766, 0.04138229, 0.13916725]))
        assert smoothed.name == sample_signal.name
        assert smoothed.shortname == sample_signal.shortname + " (smoothed)"
        assert smoothed.z_dict == sample_signal.z_dict

        # Should return original signal on error
        result = sample_signal.smooth(5, 10001)
        assert result == sample_signal

    def test_reduce_range(self, sample_signal) -> None:
        """Test reduce_range method"""

        # Set up test range
        x_range = [2.0, 8.0]

        # Find expected indices
        index1 = 4
        index2 = 16

        # Call reduce_range
        reduced = sample_signal.reduce_range(x_range)

        # Verify results
        assert isinstance(reduced, SignalData)
        assert np.array_equal(reduced.x.data, sample_signal.x.data[index1:index2])
        assert np.array_equal(reduced.y.data, sample_signal.y.data[index1:index2])
        assert reduced.name == sample_signal.name
        assert reduced.shortname == sample_signal.shortname
        assert reduced.z_dict == sample_signal.z_dict

    def test_get_point(self, sample_signal) -> None:
        """Test _get_point method"""

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

    def test_get_max(self, sample_signal) -> None:
        """Test get_max method"""

        result = sample_signal.get_max()
        assert are_close(result[0].data, 3.0)
        assert result[0].quantity == 'max. time'
        assert result[0].unit == 's'
        assert are_close(result[1].data, 1.0)
        assert result[1].quantity == 'max. amplitude'
        assert result[1].unit == 'V'
        assert are_close(result[2].data, 6)
        assert result[2].quantity == ''
        assert result[2].unit == ''

    def test_get_min(self, sample_signal) -> None:
        """Test get_min method"""

        result = sample_signal.get_min()
        assert are_close(result[0].data, 10.0)
        assert result[0].quantity == 'min. time'
        assert result[0].unit == 's'
        assert are_close(result[1].data, 2.289734845645553e-11)
        assert result[1].quantity == 'min. amplitude'
        assert result[1].unit == 'V'
        assert are_close(result[2].data, 20)
        assert result[2].quantity == ''
        assert result[2].unit == ''

    def test_get_extrema(self, sample_signal, sample_dimensions) -> None:
        """Test get_extrema method"""

        result = sample_signal.get_extrema()
        assert are_close(result[0].data, 3.0)
        assert result[0].quantity == 'extremum time'
        assert result[0].unit == 's'
        assert are_close(result[1].data, 1.)
        assert result[1].quantity == 'extremum amplitude'
        assert result[1].unit == 'V'
        assert are_close(result[2].data, 6)
        assert result[2].quantity == ''
        assert result[2].unit == ''

        x_dim, y_dim = sample_dimensions
        signal = SignalData(x_dim, Dimension(-y_dim.data))
        result = signal.get_extrema()
        assert are_close(result[0].data, 3.)
        assert result[0].quantity == 'extremum time'
        assert result[0].unit == 's'
        assert are_close(result[1].data, -1.)
        assert result[1].quantity == 'extremum '
        assert result[1].unit == ''
        assert are_close(result[2].data, 6)
        assert result[2].quantity == ''
        assert result[2].unit == ''

    def test_get_fwhm(self, sample_signal) -> None:
        """Test get_fwhm method"""

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
        assert fwhm_dim.data == 0
        assert x_left.data is None
        assert y_left.data is None
        assert x_right.data == 1
        assert y_right.data == 1

        # Test edge case: m == 0
        ext_i = Dimension(np.array([0, 1]))
        signal = SignalData(ext_i, ext_i)
        fwhm_dim, x_left, y_left, x_right, y_right = signal.get_fwhm()
        assert fwhm_dim.data == 0
        assert x_left.data == 0
        assert y_left.data == 0
        assert x_right.data is None
        assert y_right.data is None

    def test_get_half_int(self, sample_signal) -> None:

        # Test the function with a peak pointing up and no interpolation.
        x_half, y_half = sample_signal.get_halfint_point(sample_signal.x.data, sample_signal.y.data, False)
        assert are_close(x_half, 2.0)
        assert are_close(y_half, 0.606)

        # Test the function with a peak pointing down and no interpolation
        x_half, y_half = sample_signal.get_halfint_point(sample_signal.x.data, - sample_signal.y.data, False)
        assert are_close(x_half, 2.0)
        assert are_close(y_half, -0.606)

        # Test the function with a provided half intensity value
        x_half, y_half = sample_signal.get_halfint_point(sample_signal.x.data, sample_signal.y.data, False, 0.8)
        assert are_close(x_half, 3.5)
        assert are_close(y_half, 0.8825)

        # Test the function with flat data
        signal = SignalData(Dimension(np.linspace(0, 10, 21)), Dimension(np.ones(21)))
        x_half, y_half = signal.get_halfint_point(signal.x.data, signal.y.data, interpolation=False)
        assert are_close(x_half, 0.0)
        assert are_close(y_half, 1.0)

        # Test that interpolation is called correctly
        x_half, y_half = sample_signal.get_halfint_point(sample_signal.x.data, sample_signal.y.data, True)
        assert are_close(x_half, 1.810810810810811)
        assert are_close(y_half, 0.49987404638939104)
