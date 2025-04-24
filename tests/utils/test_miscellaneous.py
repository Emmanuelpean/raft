"""Test module for the functions in the `utils/miscellaneous.py` module.

This module contains unit tests for the functions implemented in the `utils/miscellaneous.py` module. The purpose of these tests is to
ensure the correct functionality of each function in different scenarios and to validate that the expected outputs are
returned.

Tests should cover various edge cases, valid inputs, and any other conditions that are necessary to confirm the
robustness of the functions."""

import numpy as np

from utils.checks import are_identical
from utils.miscellaneous import sort_lists, normalise_list, make_unique, merge_dicts


class TestSortLists:

    def test_basic_sorting(self) -> None:

        alist = [np.array([1, 4, 3, 5]), np.array([2, 1, 3, 3])]
        expected = np.array([[4, 1, 3, 5], [1, 2, 3, 3]])
        result = sort_lists(alist, 1)
        are_identical(result, expected)

    def test_sorting_multiple_indices(self) -> None:

        alist = [np.array([1, 4, 3, 5]), np.array([2, 1, 3, 3]), np.array([7, 1, 6, 3])]
        expected = np.array([[4, 1, 5, 3], [1, 2, 3, 3], [1, 7, 3, 6]])
        result = sort_lists(alist, 1, 2)
        are_identical(result, expected)

    def test_no_indices_provided(self) -> None:

        alist = [np.array([4, 1, 3, 5]), np.array([2, 1, 3, 3])]
        expected = np.array([[1, 3, 4, 5], [1, 3, 2, 3]])  # Default sorting by first index
        result = sort_lists(alist)
        are_identical(result, expected)

    def test_empty_input(self) -> None:

        alist = []
        expected = np.array([])
        result = sort_lists(alist)
        assert result.shape == expected.shape

    def test_single_element_lists(self) -> None:

        alist = [np.array([5]), np.array([3])]
        expected = np.array([[5], [3]])
        result = sort_lists(alist, 1)
        are_identical(result, expected)


class TestNormaliseInput:

    def test_list_objects(self) -> None:

        output = normalise_list([1, 3, 5, 76])
        assert are_identical(output, [1, 3, 5, 76])

    def test_list_lists_objects(self) -> None:

        output = normalise_list([[1, 3, 5, 76], [1, 3, 5, 76]])
        assert are_identical(output, [1, 3, 5, 76, 1, 3, 5, 76])

    def test_list_dicts_objects(self) -> None:

        output = normalise_list([dict(a=1, b=2), dict(a=4, b=6)])
        assert are_identical(output, {"b": [2, 6], "a": [1, 4]})

    def test_list_dicts_lists_objects(self) -> None:

        output = normalise_list([dict(a=[1, 2, 3], b=[2, 3, 4]), dict(a=[4, 5, 6], b=[5, 6, 7])])
        assert are_identical(output, {"b": [2, 3, 4, 5, 6, 7], "a": [1, 2, 3, 4, 5, 6]})


class TestMakeUnique:

    def test_all_unique_strings(self) -> None:

        input_list = ["apple", "banana", "cherry"]
        expected = ["apple", "banana", "cherry"]
        assert are_identical(make_unique(input_list), expected)

    def test_with_duplicates(self) -> None:

        input_list = ["apple", "banana", "apple", "banana", "apple"]
        expected = ["apple", "banana", "apple (1)", "banana (1)", "apple (2)"]
        assert are_identical(make_unique(input_list), expected)

    def test_with_preexisting_suffixes(self) -> None:

        input_list = ["name", "name (1)", "name"]
        expected = ["name", "name (1)", "name (2)"]
        assert are_identical(make_unique(input_list), expected)

    def test_mixed_case_duplicates(self) -> None:

        input_list = ["Test", "test", "Test"]
        expected = ["Test", "test", "Test (1)"]
        assert are_identical(make_unique(input_list), expected)

    def test_empty_list(self) -> None:

        assert make_unique([]) == []

    def test_single_entry(self) -> None:

        assert are_identical(["only"], ["only"])

    def test_duplicate_suffix_collision(self) -> None:

        input_list = ["file", "file (1)", "file", "file (1)"]
        expected = ["file", "file (1)", "file (2)", "file (1) (1)"]
        assert are_identical(make_unique(input_list), expected)


class TestMergeDicts:

    def test_empty_input(self) -> None:
        """Test with empty input."""

        assert are_identical(merge_dicts(), {})

    def test_single_dict(self) -> None:
        """Test with a single dictionary."""

        assert are_identical(merge_dicts({"a": 1, "b": 2}), {"a": 1, "b": 2})

    def test_two_dicts_no_overlap(self) -> None:
        """Test with two dictionaries that don't have overlapping keys."""

        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3, "d": 4}
        assert are_identical(merge_dicts(dict1, dict2), {"a": 1, "b": 2, "c": 3, "d": 4})

    def test_two_dicts_with_overlap(self) -> None:
        """Test with two dictionaries that have overlapping keys."""

        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 3, "c": 4}
        assert are_identical(merge_dicts(dict1, dict2), {"a": 1, "b": 2, "c": 4})

    def test_nested_dictionaries(self) -> None:
        """Test with nested dictionaries."""

        dict1 = {"a": {"nested": 1}}
        dict2 = {"a": {"different": 2}, "b": 3}
        # The entire nested dict should be kept from dict1
        assert are_identical(merge_dicts(dict1, dict2), {"a": {"nested": 1, "different": 2}, "b": 3})
