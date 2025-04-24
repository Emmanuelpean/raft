"""Test module for the functions in the `data_extraction.py` module.

This module contains unit tests for the functions implemented in the `data_extraction.py` module. The purpose of these tests is to
ensure the correct functionality of each function in different scenarios and to validate that the expected outputs are
returned.

Tests should cover various edge cases, valid inputs, and any other conditions that are necessary to confirm the
robustness of the functions."""

import numpy as np
import pytest

from data_files.data_extraction import get_data_index, get_precision, get_header_as_dicts, grep, stringlist_to_matrix
from utils.checks import are_identical


class TestStringListToMatrix:

    def test_basic_conversion(self) -> None:

        raw_data = ["1 2 3", "4 5 6", "7 8 9"]
        expected = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        result = stringlist_to_matrix(raw_data)
        assert are_identical(result, expected)

    def test_comma_delimiter(self) -> None:

        raw_data = ["1,2,3", "4,5,6", "7,8,9"]
        expected = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        result = stringlist_to_matrix(raw_data, delimiter=",")
        assert are_identical(result, expected)

    def test_uneven_rows(self) -> None:

        raw_data = ["1,2,", "3,4,5", "6,,"]
        expected = np.array([[1, 3, 6], [2, 4, np.nan], [np.nan, 5, np.nan]])
        result = stringlist_to_matrix(raw_data, ",")
        assert are_identical(result, expected)

    def test_non_numeric_values(self) -> None:

        raw_data = ["1 2 5", "4 b 6", "c 8 9"]
        expected = np.array([[1, 4, np.nan], [2, np.nan, 8], [5, 6, 9]])
        result = stringlist_to_matrix(raw_data)
        assert are_identical(result, expected)

    def test_empty_input(self) -> None:

        raw_data = []
        expected = np.array([])
        result = stringlist_to_matrix(raw_data)
        assert result.shape == expected.shape

    def test_uneven_lines(self) -> None:

        raw_data = ["1998.97,-70.38,,", "1999.49,-77.46,,", "2000.00,-84.53,"]
        result = stringlist_to_matrix(raw_data, ",")
        assert are_identical(result[0], [1998.97, 1999.49, 2000.0])
        assert are_identical(result[-1], [np.nan, np.nan, np.nan])


class TestGetHeaderAsDicts:

    def test_multiple_values(self) -> None:

        header = ["Key1\tVal1\tValA", "Key2\tVal2\tValB"]
        expected = [{"Key1": "Val1", "Key2": "Val2"}, {"Key1": "ValA", "Key2": "ValB"}]
        assert are_identical(get_header_as_dicts(header), expected)

    def test_different_delimiter(self) -> None:

        header = ["Name 1,A,B", "Unit,a,b"]
        expected = [{"Name 1": "A", "Unit": "a"}, {"Name 1": "B", "Unit": "b"}]
        assert are_identical(get_header_as_dicts(header, ","), expected)

    def test_empty_header(self) -> None:

        with pytest.raises(ValueError):
            get_header_as_dicts([])


class TestGetDataIndex:

    def test_no_delimiter(self) -> None:
        """Test without a delimiter (default None)"""
        content = ["header", "data starts here", "1 2 3", "4 5 6"]
        result = get_data_index(content)
        assert result == 2  # the first line with float data is at index 2

    def test_with_delimiter(self) -> None:
        """Test with a specified delimiter"""
        content = ["header", "data starts here", "1,2,3", "4,5,6"]
        result = get_data_index(content, delimiter=",")
        assert result == 2  # the first line with float data is at index 2

    def test_no_data(self) -> None:
        """Test case when there are no float data lines"""
        content = ["header", "some text", "more text"]
        result = get_data_index(content)
        assert result is None  # No line contains float data

    def test_empty_list(self) -> None:
        """Test with an empty list"""
        content = []
        result = get_data_index(content)
        assert result is None  # No data in the list

    def test_mixed_data(self) -> None:
        """Test with mixed data (some numeric and some non-numeric)"""
        content = ["header", "text", "1 2 3", "text again", "4 5 6"]
        result = get_data_index(content)
        assert result == 2  # the first line with numeric data is at index 2

    def test_non_matching_delimiter(self) -> None:
        """Test with a delimiter that doesn't match any line"""
        content = ["header", "text", "1 2 3", "4 5 6"]
        result = get_data_index(content, delimiter=",")
        assert result is None  # no lines with comma as delimiter


class TestGrep:

    def setup_method(self) -> None:
        """Sample data to use across tests"""

        self.sample_content = [
            "name: John",
            "age: 30",
            "height: 175.5",
            "occupation: developer",
            "name: Jane",
            "location: London",
        ]

    def test_basic_string_search(self) -> None:
        result = grep(self.sample_content, "name:")
        expected = [["name: John", 0], ["name: Jane", 4]]
        assert result == expected

    def test_list_string_search(self) -> None:
        result = grep(self.sample_content, ["age: 30"])
        expected = [["age: 30", 1]]
        assert result == expected

    def test_no_match_returns_none(self) -> None:
        result = grep(self.sample_content, "salary:")
        assert result is None

    def test_line_number_extraction(self) -> None:
        result = grep(self.sample_content, "name:", line_nb=0)
        assert result == "John"

    def test_line_number_extraction_with_string2(self) -> None:
        # Create content with data between two markers
        content = ["data: 123 end", "info: abc end"]
        result = grep(content, "data: ", line_nb=0, string2=" end")
        assert result == "123"

    def test_float_data_type(self) -> None:
        result = grep(self.sample_content, "height: ", line_nb=0, data_type="float")
        assert result == 175.5
        assert isinstance(result, float)

    def test_int_data_type(self) -> None:
        content = ["count: 42", "total: 100"]
        result = grep(content, "count: ", line_nb=0, data_type="int")
        assert result == 42
        assert isinstance(result, int)

    def test_nb_expected_success(self) -> None:
        result = grep(self.sample_content, "name:", nb_expected=2)
        expected = [["name: John", 0], ["name: Jane", 4]]
        assert result == expected

    def test_nb_expected_failure(self) -> None:
        with pytest.raises(AssertionError):
            grep(self.sample_content, "name:", nb_expected=3)

    def test_complex_extraction(self) -> None:
        content = ["log: error [code=404] message"]
        result = grep(content, "[code=", line_nb=0, string2="]")
        assert result == "404"

    def test_multiple_occurrences_line_selection(self) -> None:
        result = grep(self.sample_content, "name:", line_nb=1)
        assert result == "Jane"

    def test_empty_content(self) -> None:
        result = grep([], "anything")
        assert result is None

    def test_whitespace_handling(self) -> None:
        content = ["key:   value  "]
        result = grep(content, "key:", line_nb=0)
        assert result == "value"  # Should strip whitespace


class TestGetPrecision:

    def test_int(self) -> None:

        assert are_identical(get_precision("5"), (5.0, 0))

    def test_float(self) -> None:

        assert are_identical(get_precision("3.14"), (3.14, 2))
