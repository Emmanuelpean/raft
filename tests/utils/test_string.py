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
from utils.strings import matrix_to_string, generate_html_table, dedent, number_to_str, get_label_html


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

    def test_nan(self) -> None:
        """Test zero as an edge case"""

        assert number_to_str(float("nan")) == ""

    def test_5_f_True_True(self) -> None:
        """Test various numbers with default settings"""

        kwargs = dict(precision=5, format_str="f", html=True, auto_exponent=True)

        # Medium value
        assert number_to_str(45.25452732, **kwargs) == "45.25453"
        assert number_to_str(45.336, **kwargs) == "45.33600"

        # Low/High value
        assert number_to_str(0.00025452732, **kwargs) == "2.54527 &#10005; 10<sup>-4</sup>"
        assert number_to_str(0.000336, **kwargs) == "3.36000 &#10005; 10<sup>-4</sup>"

    def test_5_f_True_False(self) -> None:
        """Test various numbers with default settings"""

        kwargs = dict(precision=5, format_str="f", html=True, auto_exponent=False)

        # Medium value
        assert number_to_str(45.25452732, **kwargs) == "45.25453"
        assert number_to_str(45.336, **kwargs) == "45.33600"

        # Low/High value
        assert number_to_str(0.00025452732, **kwargs) == "0.00025"
        assert number_to_str(0.000336, **kwargs) == "0.00034"

    def test_5_f_False_False(self) -> None:
        """Test various numbers with default settings"""

        kwargs = dict(precision=5, format_str="f", html=False, auto_exponent=False)

        # Medium value
        assert number_to_str(45.25452732, **kwargs) == "45.25453"
        assert number_to_str(45.336, **kwargs) == "45.33600"

        # Low/High value
        assert number_to_str(0.00025452732, **kwargs) == "0.00025"
        assert number_to_str(0.000336, **kwargs) == "0.00034"

    @pytest.mark.parametrize(
        "value, precision,format_str,html,auto_exponent,expected",
        [
            # Medium value (precision low)
            (45.25452732, 5, "f", True, True, "45.25453"),
            (45.25452732, 5, "f", True, False, "45.25453"),
            (45.25452732, 5, "f", False, False, "45.25453"),
            (45.25452732, 5, "f", False, True, "45.25453"),
            (45.25452732, 5, "g", True, True, "45.255"),
            (45.25452732, 5, "g", True, False, "45.255"),
            (45.25452732, 5, "g", False, False, "45.255"),
            (45.25452732, 5, "g", False, True, "45.255"),
            # Medium value (precision high)
            (45.254, 5, "f", True, True, "45.25400"),
            (45.254, 5, "f", True, False, "45.25400"),
            (45.254, 5, "f", False, False, "45.25400"),
            (45.254, 5, "f", False, True, "45.25400"),
            (45.254, 5, "g", True, True, "45.254"),
            (45.254, 5, "g", True, False, "45.254"),
            (45.254, 5, "g", False, False, "45.254"),
            (45.254, 5, "g", False, True, "45.254"),
            # Low value (precision low)
            (1.325224262e-5, 5, "f", True, True, "1.32522 &#10005; 10<sup>-5</sup>"),
            (1.325224262e-5, 5, "f", True, False, "0.00001"),
            (1.325224262e-5, 5, "f", False, False, "0.00001"),
            (1.325224262e-5, 5, "f", False, True, "1.32522E-5"),
            (1.325224262e-5, 5, "g", True, True, "1.3252 &#10005; 10<sup>-5</sup>"),
            (1.325224262e-5, 5, "g", True, False, "1.3252e-05"),
            (1.325224262e-5, 5, "g", False, False, "1.3252e-05"),
            (1.325224262e-5, 5, "g", False, True, "1.3252E-5"),
            # Low value (precision high)
            (1.325e-5, 5, "f", True, True, "1.32500 &#10005; 10<sup>-5</sup>"),
            (1.325e-5, 5, "f", True, False, "0.00001"),
            (1.325e-5, 5, "f", False, False, "0.00001"),
            (1.325e-5, 5, "f", False, True, "1.32500E-5"),
            (1.325e-5, 5, "g", True, True, "1.325 &#10005; 10<sup>-5</sup>"),
            (1.325e-5, 5, "g", True, False, "1.325e-05"),
            (1.325e-5, 5, "g", False, False, "1.325e-05"),
            (1.325e-5, 5, "g", False, True, "1.325E-5"),
        ],
    )
    def test_number(self, value, precision, format_str, html, auto_exponent, expected):

        assert (
            number_to_str(
                value=value,
                precision=precision,
                format_str=format_str,
                html=html,
                auto_exponent=auto_exponent,
            )
            == expected
        )

    def test_negative_medium_numbers(self) -> None:
        """Test single float to scientific notation"""

        assert number_to_str(-1.46473e-2) == "-0.01465"

    def test_list_of_floats(self) -> None:
        """Test list of floats to scientific notation"""

        result = number_to_str([1e-4, 1e-5])
        assert result == "1E-4, 1E-5"


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
            get_label_html("test?", {})

    def test_superscripts_and_subscripts(self) -> None:
        result = get_label_html("sub_sub^1 max. wavelength^2 sup^sup1 sup^-3", self.quantities_label)
        assert result == r"Sub<sub>sub</sub><sup>1</sup> \lambda_{max}<sup>2</sup> sup<sup>sup1</sup> sup<sup>-3</sup>"

    def test_multiple_dictionary_replacements(self) -> None:
        result = get_label_html("int. max. max. wavelength^2", self.quantities_label)
        assert result == r"Int. max. \lambda_{max}<sup>2</sup>"

    def test_no_capitalization(self) -> None:
        custom_dict = {"rap": r"\tau", "taureau": "t|rap"}
        result = get_label_html("taureau", custom_dict, False)
        assert result == "t rap"

    def test_with_units_label(self) -> None:
        result = get_label_html("test mum_test^2", self.units_label, False)
        assert result == r"test \mum<sub>test</sub><sup>2</sup>"

    def test_capitalize_true(self) -> None:
        result = get_label_html("test string", {}, True)
        assert result == "Test string"

    def test_capitalize_short_word(self) -> None:
        # Words with length 1 should not be capitalized even if capitalize=True
        result = get_label_html("a test string", {}, True)
        assert result == "A test string"

    def test_dictionary_replacement_conditions(self) -> None:
        # Test that replacements only occur when surrounded by whitespace, underscore, or caret
        test_dict = {"word": "REPLACED"}

        # Should be replaced
        assert "REPLACED" in get_label_html(" word ", test_dict)
        assert "REPLACED" in get_label_html("_word ", test_dict)
        assert "REPLACED" in get_label_html(" word_", test_dict)
        assert "REPLACED" in get_label_html("^word ", test_dict)
        assert "REPLACED" in get_label_html(" word^", test_dict)

        # Should not be replaced
        assert "REPLACED" not in get_label_html("aword", test_dict)
        assert "REPLACED" not in get_label_html("worda", test_dict)
        assert "REPLACED" not in get_label_html("awordb", test_dict)

    def test_sorted_dictionary_keys(self) -> None:
        # Test that longer keys are replaced first
        test_dict = {
            "short": "SHORT",
            "very long key": "LONG",
        }
        result = get_label_html("test very long key short test", test_dict)
        assert result == "Test LONG SHORT test"

    def test_pipe_replacement(self) -> None:
        test_dict = {"test": "replacement|with|pipes"}
        result = get_label_html("test", test_dict)
        assert result == "replacement with pipes"

    def test_complex_formatting(self) -> None:
        test_dict = {"complex": r"\lambda_{α}^{β}"}
        result = get_label_html("complex_value^2", test_dict)
        assert result == r"\lambda_{α}^{β}<sub>value</sub><sup>2</sup>"

    def test_empty_string(self) -> None:
        with pytest.raises(IndexError):
            get_label_html("", {})
