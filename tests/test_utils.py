"""Test module for the functions in the `utils.py` module.

This module contains unit tests for the functions implemented in the `utils.py` module. The purpose of these tests is to
ensure the correct functionality of each function in different scenarios and to validate that the expected outputs are
returned.

Tests should cover various edge cases, valid inputs, and any other conditions that are necessary to confirm the
robustness of the functions."""

import datetime as dt
import os
from unittest.mock import mock_open, patch

import pytest
from bs4 import BeautifulSoup

from app.utils import *


# -------------------------------------------------- STRING CONVERSION -------------------------------------------------


class TestStringListToMatrix:

    def test_basic_conversion(self) -> None:

        raw_data = ["1 2 3", "4 5 6", "7 8 9"]
        expected = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        result = stringlist_to_matrix(raw_data)
        assert are_close(result, expected)

    def test_comma_delimiter(self) -> None:

        raw_data = ["1,2,3", "4,5,6", "7,8,9"]
        expected = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        result = stringlist_to_matrix(raw_data, delimiter=",")
        assert are_close(result, expected)

    def test_uneven_rows(self) -> None:

        raw_data = ["1,2,", "3,4,5", "6,,"]
        expected = np.array([[1, 3, 6], [2, 4, np.nan], [np.nan, 5, np.nan]])
        result = stringlist_to_matrix(raw_data, ",")
        assert are_close(result, expected)

    def test_non_numeric_values(self) -> None:

        raw_data = ["1 2 5", "4 b 6", "c 8 9"]
        expected = np.array([[1, 4, np.nan], [2, np.nan, 8], [5, 6, 9]])
        result = stringlist_to_matrix(raw_data)
        assert are_close(result, expected)

    def test_empty_input(self) -> None:

        raw_data = []
        expected = np.array([])
        result = stringlist_to_matrix(raw_data)
        assert result.shape == expected.shape

    def test_uneven_lines(self) -> None:

        raw_data = ["1998.97,-70.38,,", "1999.49,-77.46,,", "2000.00,-84.53,"]
        result = stringlist_to_matrix(raw_data, ",")
        assert are_close(result[0], [1998.97, 1999.49, 2000.0])
        assert are_close(result[-1], [np.nan, np.nan, np.nan])


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


# --------------------------------------------------- DATA CONVERSION --------------------------------------------------


class TestRenderImage:

    def setup_method(self) -> None:
        """Sample SVG content for testing"""

        self.sample_svg_content = b'<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"></svg>'
        self.encoded_svg = base64.b64encode(self.sample_svg_content).decode()

    @patch("builtins.open", new_callable=mock_open)
    def test_render_image_default_params(self, mock_file) -> None:
        """Test render_image with default parameters."""

        # Configure the mock to return our sample SVG content
        mock_file.return_value.read.return_value = self.sample_svg_content

        # Call the function with just the file path
        result = render_image("test.svg")

        # Verify the file was opened correctly
        mock_file.assert_called_once_with("test.svg", "rb")

        # Check the returned HTML string
        expected_html = f'<center><img src="data:image/svg+xml;base64,{self.encoded_svg}" id="responsive-image" width="100%"/></center>'
        assert result == expected_html

    @patch("builtins.open", new_callable=mock_open)
    def test_render_image_custom_width(self, mock_file) -> None:
        """Test render_image with custom width."""

        mock_file.return_value.read.return_value = self.sample_svg_content

        # Call with custom width
        result = render_image("test.svg", width=50)

        # Check the width in the resulting HTML
        expected_html = f'<center><img src="data:image/svg+xml;base64,{self.encoded_svg}" id="responsive-image" width="50%"/></center>'
        assert result == expected_html

    @patch("builtins.open", new_callable=mock_open)
    def test_render_image_custom_itype(self, mock_file) -> None:
        """Test render_image with custom image type."""

        mock_file.return_value.read.return_value = self.sample_svg_content

        # Call with custom image type
        result = render_image("test.svg", itype="png")

        # Check the image type in the resulting HTML
        expected_html = f'<center><img src="data:image/png+xml;base64,{self.encoded_svg}" id="responsive-image" width="100%"/></center>'
        assert result == expected_html

    @patch("builtins.open", new_callable=mock_open)
    def test_render_image_all_custom_params(self, mock_file) -> None:
        """Test render_image with all custom parameters."""

        mock_file.return_value.read.return_value = self.sample_svg_content

        # Call with all custom parameters
        result = render_image("test.svg", width=75, itype="jpeg")

        # Check all parameters in the resulting HTML
        expected_html = f'<center><img src="data:image/jpeg+xml;base64,{self.encoded_svg}" id="responsive-image" width="75%"/></center>'
        assert result == expected_html

    def test_render_image_file_not_found(self) -> None:
        """Test render_image with non-existent file."""

        # Test with a file that doesn't exist
        with pytest.raises(FileNotFoundError):
            render_image("nonexistent.svg")

    @patch("builtins.open")
    def test_render_image_read_error(self, mock_file) -> None:
        """Test render_image with file read error."""

        # Simulate a read error
        mock_file.return_value.__enter__.return_value.read.side_effect = IOError("Read error")

        with pytest.raises(IOError):
            render_image("error.svg")

    @pytest.mark.parametrize(
        "width,expected_width",
        [
            (0, "0%"),
            (100, "100%"),
            (200, "200%"),
        ],
    )
    @patch("builtins.open", new_callable=mock_open)
    def test_render_image_various_widths(self, mock_file, width, expected_width) -> None:
        """Test render_image with various width values."""

        mock_file.return_value.read.return_value = self.sample_svg_content

        result = render_image("test.svg", width=width)

        assert f'width="{expected_width}"' in result

    @pytest.mark.parametrize("itype", ["svg", "png", "jpeg", "gif", "webp"])
    @patch("builtins.open", new_callable=mock_open)
    def test_render_image_various_types(self, mock_file, itype) -> None:
        """Test render_image with various image types."""

        mock_file.return_value.read.return_value = self.sample_svg_content

        result = render_image("test.svg", itype=itype)

        assert f"data:image/{itype}+xml;base64" in result


class TestGenerateDownloadLink:

    def test_basic_functionality(self) -> None:
        """Test basic functionality of generate_download_link."""

        x_data = [np.array([1, 2, 3])]
        y_data = [np.array([4, 5, 6])]
        header = ["X", "Y"]
        text = "Download Data"
        result = generate_download_link((x_data, y_data), header, text)
        assert '<a href="data:text/csv;base64,' in result
        assert "Download Data" in result

    def test_no_header(self) -> None:
        """Test when no header is provided."""

        x_data = [np.array([1, 2, 3])]
        y_data = [np.array([4, 5, 6])]
        text = "Download Data"
        result = generate_download_link((x_data, y_data), None, text)
        assert '<a href="data:text/csv;base64,' in result
        assert "Download Data" in result

    def test_with_special_characters(self) -> None:
        """Test if the function handles special characters in the header and text."""

        x_data = [np.array([1, 2])]
        y_data = [np.array([3, 4])]
        header = ["Col@1", "Col#2"]
        text = "Download with Special Characters"
        result = generate_download_link((x_data, y_data), header, text)

        # Extract the base64 string from the result
        base64_string = result.split("base64,")[1].split('"')[0]

        # Decode the base64 string to get the original string
        decoded_string = base64.b64decode(base64_string).decode()

        # Now check if the decoded string contains the header with special characters
        assert "Col@1" in decoded_string
        assert "Col#2" in decoded_string
        assert "Download with Special Characters" in result

    def test_no_text_provided(self) -> None:
        """Test if no text is provided (empty string)."""

        x_data = [np.array([1, 2, 3])]
        y_data = [np.array([4, 5, 6])]
        header = ["X", "Y"]
        result = generate_download_link((x_data, y_data), header, "")
        assert '<a href="data:text/csv;base64,' in result
        assert 'href="data:text/csv;base64,' in result
        assert "Download" not in result  # Should not have any text if empty


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
        df.columns.name = "Categories"
        return df

    def test_basic_table_generation(self, sample_dataframe) -> None:
        """Test basic HTML table generation."""

        html = generate_html_table(sample_dataframe)

        # Verify structure using BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

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

        soup = BeautifulSoup(html, "html.parser")
        corner_cell = soup.find("tr").find("th")

        assert corner_cell.text == "Categories"

    def test_empty_corner_cell_with_no_column_name(self, sample_dataframe) -> None:
        """Test that the corner cell is empty when no column name is provided."""

        html = generate_html_table(sample_dataframe)

        soup = BeautifulSoup(html, "html.parser")
        corner_cell = soup.find("tr").find("th")

        assert corner_cell.text == ""

    def test_div_wrapper(self, sample_dataframe) -> None:
        """Test that the table is wrapped in a div with correct styling."""

        html = generate_html_table(sample_dataframe)

        soup = BeautifulSoup(html, "html.parser")
        div = soup.find("div")

        assert div is not None
        assert div.has_attr("style")
        assert "margin: auto" in div["style"]
        assert "display: table" in div["style"]


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


# --------------------------------------------------- DATA EXTRACTION --------------------------------------------------


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


# -------------------------------------------------- DATA MANIPULATION -------------------------------------------------


class TestMergeDicts:

    def test_empty_input(self) -> None:
        """Test with empty input."""
        assert merge_dicts() == {}

    def test_single_dict(self) -> None:
        """Test with a single dictionary."""
        assert merge_dicts({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_two_dicts_no_overlap(self) -> None:
        """Test with two dictionaries that don't have overlapping keys."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3, "d": 4}
        assert merge_dicts(dict1, dict2) == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_two_dicts_with_overlap(self) -> None:
        """Test with two dictionaries that have overlapping keys."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 3, "c": 4}
        assert merge_dicts(dict1, dict2) == {"a": 1, "b": 2, "c": 4}

    def test_nested_dictionaries(self) -> None:
        """Test with nested dictionaries."""
        dict1 = {"a": {"nested": 1}}
        dict2 = {"a": {"different": 2}, "b": 3}
        # The entire nested dict should be kept from dict1
        assert merge_dicts(dict1, dict2) == {"a": {"nested": 1}, "b": 3}


class TestSortLists:

    def test_basic_sorting(self) -> None:
        alist = [np.array([1, 4, 3, 5]), np.array([2, 1, 3, 3])]
        expected = np.array([[4, 1, 3, 5], [1, 2, 3, 3]])
        result = sort_lists(alist, 1)
        are_close(result, expected)

    def test_sorting_multiple_indices(self) -> None:
        alist = [np.array([1, 4, 3, 5]), np.array([2, 1, 3, 3]), np.array([7, 1, 6, 3])]
        expected = np.array([[4, 1, 5, 3], [1, 2, 3, 3], [1, 7, 3, 6]])
        result = sort_lists(alist, 1, 2)
        are_close(result, expected)

    def test_no_indices_provided(self) -> None:
        alist = [np.array([4, 1, 3, 5]), np.array([2, 1, 3, 3])]
        expected = np.array([[1, 3, 4, 5], [1, 3, 2, 3]])  # Default sorting by first index
        result = sort_lists(alist)
        are_close(result, expected)

    def test_empty_input(self) -> None:
        alist = []
        expected = np.array([])
        result = sort_lists(alist)
        assert result.shape == expected.shape

    def test_single_element_lists(self) -> None:
        alist = [np.array([5]), np.array([3])]
        expected = np.array([[5], [3]])
        result = sort_lists(alist, 1)
        are_close(result, expected)


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


class TestNormaliseInput:

    def test_list_objects(self) -> None:

        output = normalise_list([1, 3, 5, 76])
        assert output == [1, 3, 5, 76]

    def test_list_lists_objects(self) -> None:

        output = normalise_list([[1, 3, 5, 76], [1, 3, 5, 76]])
        assert output == [1, 3, 5, 76, 1, 3, 5, 76]

    def test_list_dicts_objects(self) -> None:

        output = normalise_list([dict(a=1, b=2), dict(a=4, b=6)])
        assert output == {"b": [2, 6], "a": [1, 4]}

    def test_list_dicts_lists_objects(self) -> None:

        output = normalise_list([dict(a=[1, 2, 3], b=[2, 3, 4]), dict(a=[4, 5, 6], b=[5, 6, 7])])
        assert output == {"b": [2, 3, 4, 5, 6, 7], "a": [1, 2, 3, 4, 5, 6]}


# ---------------------------------------------------- DATA CHECKING ---------------------------------------------------


class TestComparisonFunctions:

    def test_identical_simple_values(self) -> None:
        """Test identical simple values."""

        assert are_identical(5, 5)
        assert are_identical("test", "test")
        assert are_identical(None, None)
        assert not are_identical(5, 6)
        assert not are_identical("test", "different")

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

    def test_datetime(self) -> None:
        obj1 = dt.datetime(year=2025, day=25, month=3)
        obj2 = dt.datetime(year=2026, day=25, month=3)
        assert are_identical(obj1, obj1)
        assert not are_identical(obj1, obj2)


class TestReadTxtFile:
    """Test class for the read_txt_file function."""

    # File path for temporary test file
    TEMP_FILE = "_temp.txt"

    def teardown_method(self) -> None:
        """Teardown method that runs after each test."""

        # Clean up test file after each test
        if os.path.exists(self.TEMP_FILE):
            os.remove(self.TEMP_FILE)

        # Clear the cache
        st.cache_resource.clear()

    def test_read_existing_file(self) -> None:
        """Test reading from an existing file with valid content."""

        # Create file with some content
        with open(self.TEMP_FILE, "w") as f:
            f.write("Hello, World!")

        # Read the content using our function
        content = read_file(self.TEMP_FILE)

        # Assert the content matches what we wrote
        assert content == "Hello, World!"

    def test_read_multiline_file(self) -> None:
        """Test reading from a file with multiple lines."""

        # Create a file with multiline content
        with open(self.TEMP_FILE, "w") as f:
            f.write("Line 1\nLine 2\nLine 3")

        # Read the content using our function
        content = read_file(self.TEMP_FILE)

        # Assert the content matches what we wrote
        assert content == "Line 1\nLine 2\nLine 3"
        # Additional check for line count
        assert len(content.splitlines()) == 3

    def test_nonexistent_file(self) -> None:
        """Test that trying to read a nonexistent file raises an error."""

        # Check that the function raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            read_file(self.TEMP_FILE)
