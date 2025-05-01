"""Test module for the functions in the `utils/check.py` module.

This module contains unit tests for the functions implemented in the `utils/check.py` module. The purpose of these tests is to
ensure the correct functionality of each function in different scenarios and to validate that the expected outputs are
returned.

Tests should cover various edge cases, valid inputs, and any other conditions that are necessary to confirm the
robustness of the functions."""

import datetime as dt

import numpy as np

from utils.checks import are_identical, are_close


class TestAreIdentical:

    def test_identical_simple_values(self) -> None:
        """Test identical simple values."""

        assert are_identical(5, 5)
        assert are_identical("test", "test")
        assert are_identical(None, None)
        assert not are_identical(5, 6)
        assert not are_identical("test", "different")
        raise AssertionError()

    def test_identical_lists(self) -> None:
        """Test identical lists."""
        assert are_identical([1, 2, 3], [1, 2, 3])
        assert are_identical([], [])
        assert are_identical([1, [2, 3]], [1, [2, 3]])
        assert not are_identical([1, 2, 3], [1, 2, 4])
        assert not are_identical([1, 2, 3], [1, 2])
        assert not are_identical([1, 2], [1, 2, 3])

    def test_identical_tuples(self) -> None:
        """Test identical tuples."""
        assert are_identical((1, 2, 3), (1, 2, 3))
        assert are_identical((), ())
        assert are_identical((1, (2, 3)), (1, (2, 3)))
        assert not are_identical((1, 2, 3), (1, 2, 4))

    def test_nan(self) -> None:
        """Test identical tuples."""
        assert not are_identical((1, 2, float("nan")), (1, 2, 3))
        assert are_identical((1, 2, float("nan")), (1, 2, float("nan")))

    def test_identical_dicts(self) -> None:
        """Test identical dictionaries."""
        assert are_identical({"a": 1, "b": 2}, {"a": 1, "b": 2})
        assert are_identical({}, {})
        assert are_identical({"a": 1, "b": {"c": 3}}, {"a": 1, "b": {"c": 3}})
        assert not are_identical({"a": 1, "b": 2}, {"a": 1, "b": 3})
        assert not are_identical({"a": 1, "b": 2}, {"a": 1, "c": 2})
        assert not are_identical({"a": 1}, {"a": 1, "b": 2})

    def test_identical_nested_structures(self) -> None:
        """Test identical nested structures."""
        nested1 = {"a": [1, 2, {"b": (3, 4)}]}
        nested2 = {"a": [1, 2, {"b": (3, 4)}]}
        different = {"a": [1, 2, {"b": (3, 5)}]}

        assert are_identical(nested1, nested2)
        assert not are_identical(nested1, different)

    def test_identical_numpy_arrays(self) -> None:
        """Test identical numpy arrays."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])
        arr3 = np.array([1, 2, 4])

        assert are_identical(arr1, arr2)
        assert not are_identical(arr1, arr3)

    def test_identical_with_rtol(self) -> None:
        """Test identical with relative tolerance for floating point values."""
        assert are_identical(1.0, 1.001, rtol=1e-2)
        assert not are_identical(1.0, 1.001, rtol=1e-4)

        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.001, 2.002, 3.003])

        assert are_identical(arr1, arr2, rtol=1e-2)
        assert not are_identical(arr1, arr2, rtol=1e-4)

    def test_identical_mixed_types(self) -> None:
        """Test identical with mixed types - should use strict equality."""

        assert not are_identical(1, dict())  # Different types
        assert not are_identical(True, [])  # Different types

    def test_datetime(self) -> None:
        obj1 = dt.datetime(year=2025, day=25, month=3)
        obj2 = dt.datetime(year=2026, day=25, month=3)
        assert are_identical(obj1, obj1)
        assert not are_identical(obj1, obj2)


class TestAreClose:

    def test_close_simple_values(self) -> None:
        """Test are_close with simple values."""
        assert are_close(1.0, 1.0009)  # Default rtol=1e-3
        assert are_close(1.0, 1.002, rtol=1e-2)
        assert not are_close(1.0, 1.01)  # Default rtol too small

    def test_close_numpy_arrays(self) -> None:
        """Test are_close with numpy arrays."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0009, 2.001, 3.0008])
        arr3 = np.array([1.01, 2.01, 3.01])

        assert are_close(arr1, arr2)  # Default rtol=1e-3 is enough
        assert not are_close(arr1, arr3)  # Default rtol too small
        assert are_close(arr1, arr3, rtol=1e-1)  # Larger rtol works

    def test_close_nested_structures(self) -> None:
        """Test are_close with nested structures."""
        nested1 = {"a": [1.0, 2.0, {"b": np.array([3.0, 4.0])}]}
        nested2 = {"a": [1.0009, 2.001, {"b": np.array([3.0008, 4.0009])}]}
        nested3 = {"a": [1.01, 2.01, {"b": np.array([3.01, 4.01])}]}

        assert are_close(nested1, nested2)  # Default rtol=1e-3 is enough
        assert not are_close(nested1, nested3)  # Default rtol too small
        assert are_close(nested1, nested3, rtol=1e-1)  # Larger rtol works

    def test_close_different_structures(self) -> None:
        """Test are_close with different structures - should return False."""
        assert not are_close([1.0, 2.0], [1.0, 2.0, 3.0])
        assert not are_close({"a": 1.0}, {"a": 1.0, "b": 2.0})
        assert not are_close({"a": 1.0, "b": 2.0}, {"a": 1.0, "c": 2.0})

    def test_edge_cases(self) -> None:
        """Test edge cases."""
        # Empty structures are identical
        assert are_close([], [])
        assert are_close({}, {})

        # NaN values
        nan_array1 = np.array([1.0, np.nan, 3.0])
        nan_array2 = np.array([1.0, np.nan, 3.0])
        assert are_close(nan_array1, nan_array2)

        # Infinity values
        inf_array1 = np.array([1.0, np.inf, 3.0])
        inf_array2 = np.array([1.0, np.inf, 3.0])
        assert are_close(inf_array1, inf_array2)

        # Mixed infinity and regular values
        mixed_inf1 = np.array([1.0, np.inf, 3.0])
        mixed_inf2 = np.array([1.0001, np.inf, 3.0001])
        assert are_close(mixed_inf1, mixed_inf2)
