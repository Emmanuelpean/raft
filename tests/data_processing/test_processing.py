"""Test module for the functions in the `data_processing/processing.py` module.

This module contains unit tests for the functions implemented in the `data_processing/processing.py` module. The purpose
of these tests is to ensure the correct functionality of each function in different scenarios and to validate that the
expected outputs are returned.

Tests should cover various edge cases, valid inputs, and any other conditions that are necessary to confirm the
robustness of the functions."""

import numpy as np
import pytest
import scipy.interpolate as sci

from data_processing.processing import (
    normalise,
    feature_scale,
    interpolate_data,
    get_derivative,
    interpolate_point,
    finite_argm,
    get_area,
)
from utils.checks import are_close


class TestNormalise:
    """Test class for the normalise function."""

    def test_basic_normalisation(self) -> None:
        """Test basic normalisation with regular values."""

        input_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        result = normalise(input_array)
        are_close(result, expected)

    def test_normalise_zero_array(self) -> None:
        """Test normalisation with an array of zeros."""

        input_array = np.array([0.0, 0.0, 0.0])
        # When all values are zero, division by zero may result in NaN values
        result = normalise(input_array)
        assert np.isnan(result).all()

    def test_normalise_other(self) -> None:
        """Test feature scaling with custom min and max range."""

        input_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        input_array2 = np.array([0.14285714, 0.28571429, 0.42857143, 0.57142857, 0.71428571])
        expected = np.array([10.0, 11.66666667, 13.33333333, 15.0, 16.66666667])
        result = normalise(input_array, other=input_array2)
        are_close(result, expected)

    def test_normalise_negative_values(self) -> None:
        """Test normalisation with negative values."""
        input_array = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        expected = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        result = normalise(input_array)
        are_close(result, expected)

    def test_normalise_single_value(self) -> None:
        """Test normalisation with a single value."""
        input_array = np.array([42.0])
        expected = np.array([1.0])
        result = normalise(input_array)
        are_close(result, expected)

    def test_normalise_with_nan_values(self) -> None:
        """Test normalisation with NaN values."""

        input_array = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        expected = np.array([0.2, 0.4, np.nan, 0.8, 1.0])
        result = normalise(input_array)
        # Check non-NaN values
        mask = ~np.isnan(result)
        are_close(result[mask], expected[mask])
        # Check NaN values are preserved
        assert np.isnan(result[2])

    def test_normalise_integer_array(self) -> None:
        """Test normalisation with integer arrays."""

        input_array = np.array([10, 20, 30, 40, 50])
        expected = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        result = normalise(input_array)
        are_close(result, expected)

    def test_normalise_multidimensional_array(self) -> None:
        """Test normalisation with a multidimensional array."""

        input_array = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = np.array([[0.25, 0.5], [0.75, 1.0]])
        result = normalise(input_array)
        are_close(result, expected)


class TestFeatureScale:
    """Test class for the feature_scale function."""

    def test_basic_feature_scaling(self) -> None:
        """Test basic feature scaling with default parameters."""

        input_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = feature_scale(input_array)
        are_close(result, expected)

    def test_feature_scale_other(self) -> None:
        """Test feature scaling with custom min and max range."""

        input_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        input_array2 = np.array([1.0, 2.0, 7.0, 4.0, 5.0])
        expected = np.array([10.0, 11.66666667, 13.33333333, 15.0, 16.66666667])  # Scaled to [10, 20]
        result = feature_scale(input_array, other=input_array2, b=10.0, a=20.0)
        are_close(result, expected)

    def test_feature_scale_custom_range(self) -> None:
        """Test feature scaling with custom min and max range."""

        input_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.array([10.0, 12.5, 15.0, 17.5, 20.0])  # Scaled to [10, 20]
        result = feature_scale(input_array, b=10.0, a=20.0)
        are_close(result, expected)

    def test_feature_scale_negative_range(self) -> None:
        """Test feature scaling with a negative range."""

        input_array = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        expected = np.array([-5.0, -2.5, 0.0, 2.5, 5.0])  # Scaled to [-5, 5]
        result = feature_scale(input_array, b=-5.0, a=5.0)
        are_close(result, expected)

    def test_feature_scale_reversed_range(self) -> None:
        """Test feature scaling with a reversed range (max < min)."""

        input_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.array([1.0, 0.75, 0.5, 0.25, 0.0])  # Scaled to [1, 0]
        result = feature_scale(input_array, b=1.0, a=0.0)
        are_close(result, expected)

    def test_feature_scale_with_nan_values(self) -> None:
        """Test feature scaling with NaN values."""
        input_array = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        expected = np.array([0.0, 0.25, np.nan, 0.75, 1.0])
        result = feature_scale(input_array)
        # Check non-NaN values
        mask = ~np.isnan(result)
        are_close(result[mask], expected[mask])
        # Check NaN values are preserved
        assert np.isnan(result[2])

    def test_feature_scale_same_value_array(self) -> None:
        """Test feature scaling with an array of identical values."""
        input_array = np.array([7.0, 7.0, 7.0])
        # When all values are the same, division by zero may result in NaN values
        result = feature_scale(input_array)
        assert np.isnan(result).all()

    def test_feature_scale_integer_array(self) -> None:
        """Test feature scaling with integer arrays."""
        input_array = np.array([10, 20, 30, 40, 50])
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = feature_scale(input_array)
        are_close(result, expected)

    def test_feature_scale_multidimensional_array(self) -> None:
        """Test feature scaling with a multidimensional array."""
        input_array = np.array([[1.0, 3.0], [5.0, 7.0]])
        expected = np.array([[0.0, 1 / 3], [2 / 3, 1.0]])
        result = feature_scale(input_array)
        are_close(result, expected)


class TestInterpolateData:

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample data"""

        x_data = np.array([1.0, 2.0, 3.0, 3.5])
        y_data = x_data**2
        return x_data, y_data

    def test_float(self, sample_data) -> None:

        x_data, y_data = interpolate_data(*sample_data, dx=0.3)
        assert are_close(x_data, [1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 3.1, 3.4])
        assert are_close(y_data, [1.0, 1.9, 2.8, 3.7, 5.0, 6.5, 8.0, 9.65, 11.6])

    def test_int(self, sample_data) -> None:

        x_data, y_data = interpolate_data(*sample_data, dx=2)
        assert are_close(x_data, [1.0, 3.5])
        assert are_close(y_data, [1.0, 12.25])

    def test_negative_dx(self, sample_data) -> None:

        with pytest.raises(AssertionError):
            interpolate_data(*sample_data, dx=-2)


class TestDerivative:

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample data"""

        x_data = np.array([1.0, 2.0, 3.0, 3.5])
        y_data = x_data**3
        return x_data, y_data

    def test_first_derivative(self, sample_data) -> None:

        output = get_derivative(*sample_data)
        assert are_close(output[0], [1.5, 2.5, 3.25])
        assert are_close(output[1], [7.0, 19.0, 31.75])

    def test_second_derivative(self, sample_data) -> None:

        output = get_derivative(*sample_data, n=2)
        assert are_close(output[0], [2.0, 2.875])
        assert are_close(output[1], [12.0, 17.0])


class TestInterpolatePoint:

    def test_linear_interpolation(self) -> None:
        x = np.array([-10, -7, -5, -1, 5, 7, 10])
        y = x**2
        x_int, y_int = interpolate_point(x, y, 3, kind="linear")
        assert len(x_int) == 1000
        assert len(y_int) == 1000

        interp = sci.interp1d(x[1:6], y[1:6], kind="linear")
        are_close(y_int, interp(x_int))

    def test_cubic_interpolation(self) -> None:
        x = np.array([-10, -7, -5, -1, 5, 7, 10])
        y = x**2
        x_int, y_int = interpolate_point(x, y, 3, kind="cubic")
        assert len(x_int) == 1000
        assert len(y_int) == 1000

        interp = sci.interp1d(x[1:6], y[1:6], kind="cubic")
        are_close(y_int, interp(x_int))

    def test_default_nb_point(self) -> None:
        x = np.array([-10, -7, -5, -1, 5, 7, 10])
        y = x**2
        x_int, y_int = interpolate_point(x, y, 3)
        assert len(x_int) == 1000
        assert len(y_int) == 1000

    def test_index_near_start(self) -> None:
        x = np.array([-10, -7, -5, -1, 5, 7, 10])
        y = x**2
        x_int, y_int = interpolate_point(x, y, 0, kind="linear")
        assert len(x_int) == 1000
        assert len(y_int) == 1000

    def test_index_near_end(self) -> None:
        x = np.array([-10, -7, -5, -1, 5, 7, 10])
        y = x**2
        x_int, y_int = interpolate_point(x, y, 6, kind="linear")
        assert len(x_int) == 1000
        assert len(y_int) == 1000


class TestFiniteArgm:

    def test_argmax_basic(self) -> None:

        data = np.array([1, 3, np.nan, np.inf, -np.inf, 2])
        result = finite_argm("argmax", data)
        assert result == 1  # 3 is at index 1

    def test_argmin_basic(self) -> None:

        data = np.array([np.nan, 3, np.inf, 1, -np.inf, 2])
        result = finite_argm("argmin", data)
        assert result == 3  # 1 is at index 3

    def test_all_invalid(self) -> None:

        data = np.array([np.nan, np.inf, -np.inf])
        with pytest.raises(ValueError):
            finite_argm("argmax", data)

    def test_no_invalid(self) -> None:

        data = np.array([10, 5, 20, 15])
        assert finite_argm("argmax", data) == 2  # 20 at index 2
        assert finite_argm("argmin", data) == 1  # 5 at index 1

    def test_method_not_found(self) -> None:

        data = np.array([1, 2, 3])
        with pytest.raises(AttributeError):
            finite_argm("not_a_method", data)

    def test_original_array_not_modified(self) -> None:

        data = np.array([1, 2, np.inf])
        data_copy = data.copy()
        try:
            finite_argm("argmax", data)
        except Exception:
            pass
        assert np.array_equal(data, data_copy), "Original data should not be modified"


class TestGetArea:

    def test_simple_rectangle(self) -> None:
        """Test area calculation for a simple rectangle."""
        x = np.array([0, 1, 2, 3])
        y = np.array([2, 2, 2, 2])
        expected_area = 6.0  # Width (3) * Height (2)
        assert get_area(x, y) == expected_area

    def test_triangle(self) -> None:
        """Test area calculation for a triangle."""
        x = np.array([0, 4])
        y = np.array([0, 3])
        expected_area = 6.0  # 1/2 * base * height
        assert get_area(x, y) == expected_area

    def test_parabola(self) -> None:
        """Test area calculation for a parabola y = x^2 from 0 to 1."""
        x = np.linspace(0, 1, 100)
        y = x**2
        expected_area = 1 / 3  # Integral of x^2 from 0 to 1 = 1/3
        are_close(get_area(x, y), expected_area)

    def test_sine_function(self) -> None:
        """Test area calculation for sine function from 0 to pi."""
        x = np.linspace(0, np.pi, 1000)
        y = np.sin(x)
        expected_area = 2.0  # Integral of sin(x) from 0 to pi = 2

        are_close(get_area(x, y), expected_area)

    def test_non_uniform_x(self) -> None:
        """Test with non-uniform x-spacing."""
        x = np.array([0, 1, 3, 7, 10])
        y = np.array([5, 5, 5, 5, 5])
        expected_area = 50.0  # 5 * (10-0)
        assert get_area(x, y) == expected_area

    def test_negative_values(self) -> None:
        """Test with negative y values."""
        x = np.array([0, 1, 2, 3])
        y = np.array([-1, -2, -1, 0])
        # Area is negative because it's below x-axis
        expected_area = -3.5
        assert get_area(x, y) == expected_area

    def test_empty_arrays(self) -> None:
        """Test with empty arrays which should raise an error."""
        x = np.array([])
        y = np.array([])
        assert get_area(x, y) == 0.0

    def test_single_point(self) -> None:
        """Test with just one point which should return 0."""
        x = np.array([5])
        y = np.array([10])
        assert get_area(x, y) == 0.0
