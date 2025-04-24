"""Test module for the functions in the `utils/string.py` module.

This module contains unit tests for the functions implemented in the `utils/string.py` module. The purpose of these tests is to
ensure the correct functionality of each function in different scenarios and to validate that the expected outputs are
returned.

Tests should cover various edge cases, valid inputs, and any other conditions that are necessary to confirm the
robustness of the functions."""

import pytest
import numpy as np
import pandas as pd
import bs4
from utils.string import matrix_to_string, generate_html_table, dedent, number_to_str


class TestMatrixToString:

    def test_basic_conversion(self) -> None:

        arrays = [np.array([1.2, 2.0, 5.0]), np.array([1.6, 2.0, 5.0])]
        header = ["A", "B"]
        result = matrix_to_string(arrays, header)
        expected = "A,B\n1.20000E+00,1.60000E+00\n2.00000E+00,2.00000E+00\n5.00000E+00,5.00000E+00"
        assert result == expected

    def test_no_header(self) -> None:

        arrays = [np.array([1.2, 2, 5]), np.array([1.6, 2])]
        result = matrix_to_string(arrays)
        expected = "1.20000E+00,1.60000E+00\n2.00000E+00,2.00000E+00\n5.00000E+00,"
        assert result == expected

    def test_single_column(self) -> None:

        arrays = [np.array([1.2, 2, 5])]
        result = matrix_to_string(arrays, ["A"])
        expected = "A\n1.20000E+00\n2.00000E+00\n5.00000E+00"
        assert result == expected

    def test_mixed_lengths(self) -> None:

        arrays = [np.array([1.2, 2.0, 5.0]), np.array([1.6, 2.0])]
        header = ["A", "B"]
        result = matrix_to_string(arrays, header)
        expected = "A,B\n1.20000E+00,1.60000E+00\n2.00000E+00,2.00000E+00\n5.00000E+00,"
        assert result == expected

        arrays = [np.array([1.2, 2.0, 6.0]), np.array([1.2, 2.0]), np.array([1.6, 2.0, 5.0])]
        header = ["A", "B"]
        result = matrix_to_string(arrays, header)
        expected = (
            "A,B\n1.20000E+00,1.20000E+00,1.60000E+00\n2.00000E+00,2.00000E+00,2.00000E+00\n6.00000E+00,,5.00000E+00"
        )
        assert result == expected

        arrays = [np.array([1.2, 2.0]), np.array([1.6, 2.0, 5.0])]
        header = ["A", "B"]
        result = matrix_to_string(arrays, header)
        expected = "A,B\n1.20000E+00,1.60000E+00\n2.00000E+00,2.00000E+00\n,5.00000E+00"
        assert result == expected

    def test_all_empty(self) -> None:

        arrays = [np.array([]), np.array([])]
        header = ["A", "B"]
        result = matrix_to_string(arrays, header)
        assert result == "A,B\n"


class TestGenerateHtmlTable:

    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
        """Create a sample dataframe for testing."""

        data = {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8], "C": [9, 10, 11, 12]}
        return pd.DataFrame(data, index=["row1", "row2", "row3", "row4"])

    @pytest.fixture
    def dataframe_with_column_name(self) -> pd.DataFrame:
        """Create a dataframe with a column name."""

        data = {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}
        df = pd.DataFrame(data, index=["row1", "row2", "row3"])
        df.columns.filename = "Categories"
        return df

    def test_basic_table_generation(self, sample_dataframe) -> None:
        """Test basic HTML table generation."""

        html = generate_html_table(sample_dataframe)

        # Verify structure using BeautifulSoup
        soup = bs4.BeautifulSoup(html, "html.parser")

        # Check table exists
        table = soup.find("table")
        assert table is not None

        # Check number of rows (header + data rows)
        rows = table.find_all("tr")
        assert len(rows) == 5  # 1 header + 4 data rows

        # Check header row
        header_cells = rows[0].find_all(["th"])
        assert len(header_cells) == 4  # corner cell + 3 columns

        # Check data cells
        data_rows = rows[1:]
        for i, row in enumerate(data_rows):
            cells = row.find_all("td")
            assert cells[0].text == f"row{i + 1}"  # Check row name

    def test_column_name_in_corner(self, dataframe_with_column_name) -> None:
        """Test that the column name appears in the corner cell."""

        html = generate_html_table(dataframe_with_column_name)

        soup = bs4.BeautifulSoup(html, "html.parser")
        corner_cell = soup.find("tr").find("th")

        assert corner_cell.text == "Categories"

    def test_empty_corner_cell_with_no_column_name(self, sample_dataframe) -> None:
        """Test that the corner cell is empty when no column name is provided."""

        html = generate_html_table(sample_dataframe)

        soup = bs4.BeautifulSoup(html, "html.parser")
        corner_cell = soup.find("tr").find("th")

        assert corner_cell.text == ""

    def test_div_wrapper(self, sample_dataframe) -> None:
        """Test that the table is wrapped in a div with correct styling."""

        html = generate_html_table(sample_dataframe)

        soup = bs4.BeautifulSoup(html, "html.parser")
        div = soup.find("div")

        assert div is not None
        assert div.has_attr("style")
        assert "margin: auto" in div["style"]
        assert "display: table" in div["style"]


class TestDedent:

    def test_empty_string(self) -> None:

        assert dedent("") == ""

    def test_single_line_no_indent(self) -> None:
        assert dedent("NoIndent") == "NoIndent"

    def test_single_line_with_indent(self) -> None:
        assert dedent("    Indented") == "Indented"

    def test_multiline_indented(self) -> None:
        input_str = "    Line1\n    Line2\n    Line3"
        expected = "Line1\nLine2\nLine3"
        assert dedent(input_str) == expected

    def test_multiline_mixed_indent(self) -> None:
        input_str = "    Line1\n  Line2\n\tLine3"
        expected = "Line1\nLine2\nLine3"
        assert dedent(input_str) == expected

    def test_lines_with_only_spaces(self) -> None:
        input_str = "    \n  \n\t\nContent"
        expected = "\n\n\nContent"
        assert dedent(input_str) == expected

    def test_preserve_line_breaks(self) -> None:
        input_str = "    First\n\n    Second"
        expected = "First\n\nSecond"
        assert dedent(input_str) == expected

    def test_only_whitespace_lines(self) -> None:
        input_string = "    \n\t\n  "
        expected = "\n\n"
        assert dedent(input_string) == expected


class TestNumberToString:
    """Test cases for the to_scientific function"""

    def test_none_input(self) -> None:
        """Test None input"""

        assert number_to_str(None) == ""

    def test_edge_case_zero(self) -> None:
        """Test zero as an edge case"""

        assert number_to_str(0) == "0"

    def test_numbers_default(self) -> None:
        """Test various numbers with default settings"""

        # < 1e-2
        assert number_to_str(1.4647336e-4) == "1.4647E-4"

        # 1e-2 < value < 1
        assert number_to_str(1.4647336e-2) == "0.01465"

        # 1e4 > value > 1
        assert number_to_str(153.43433) == "153.4"
        assert number_to_str(153) == "153"  # trailing zeros

        # value > 1e4
        assert number_to_str(1536.43433) == "1.5364E3"
        assert number_to_str(1536) == "1.536E3"  # trailing zeros

    def test_numbers_n(self) -> None:
        """Test various numbers with default settings"""

        # < 1e-2
        assert number_to_str(1.4647336e-4, n=5) == "1.46473E-4"

        # 1e-2 < value < 1
        assert number_to_str(1.4647336e-2, n=5) == "0.014647"

        # 1e4 > value > 1
        assert number_to_str(153.43433, n=5) == "153.43"
        assert number_to_str(153, n=5) == "153"  # trailing zeros

        # value > 1e4
        assert number_to_str(1536.43433, n=5) == "1.53643E3"
        assert number_to_str(1536, n=5) == "1.536E3"  # trailing zeros

    def test_numbers_display(self) -> None:
        """Test various numbers with default settings"""

        # < 1e-2
        assert number_to_str(1.4647336e-4, display=True) == "1.4647 &#10005; 10<sup>-4</sup>"

        # 1e-2 < value < 1
        assert number_to_str(1.4647336e-2, display=True) == "0.01465"

        # 1e4 > value > 1
        assert number_to_str(153.43433, display=True) == "153.4"
        assert number_to_str(153, display=True) == "153.0"  # trailing zeros

        # value > 1e4
        assert number_to_str(1536.43433, display=True) == "1.5364 &#10005; 10<sup>3</sup>"
        assert number_to_str(1536, display=True) == "1.5360 &#10005; 10<sup>3</sup>"  # trailing zeros

    def test_negative_medium_numbers(self) -> None:
        """Test single float to scientific notation"""

        assert number_to_str(-1.46473e-2) == "-0.01465"

    def test_list_of_floats(self) -> None:
        """Test list of floats to scientific notation"""

        result = number_to_str([1e-4, 1e-5])
        assert result == "1E-4, 1E-5"
