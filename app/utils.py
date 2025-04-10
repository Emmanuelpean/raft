"""utils module"""

import base64
import math
from operator import itemgetter

import numpy as np
import pandas as pd
import scipy.interpolate as sci
import streamlit as st


@st.cache_resource
def read_file(path: str) -> str:
    """Read the content of a file and store it as a resource.
    :param path: file path"""

    with open(path, encoding="utf-8") as ofile:
        return ofile.read()


# -------------------------------------------------- STRING CONVERSION -------------------------------------------------


def stringlist_to_matrix(
    raw_data: list[str],
    delimiter: str | None = None,
) -> np.ndarray:
    """Quickly convert a list of strings to ndarrays
    :param raw_data: list of strings where each string contains as many numbers as there are columns
    :param delimiter: delimiter separating two data in the strings"""

    lines = []
    for line in raw_data:
        line_float = []

        # Split the line and try to convert the numbers
        for string in line.split(delimiter):
            try:
                line_float.append(float(string))
            except ValueError:
                line_float.append(float("nan"))

        lines.append(line_float)

    # Check the number of elements in all lines are consistent
    for line in lines:
        while len(line) < len(lines[0]):
            line.append(float("nan"))

    return np.transpose(lines)


def matrix_to_string(
    arrays: np.ndarray | list[np.ndarray],
    header: None | list[str] | np.ndarray = None,
) -> str:
    """Convert a matrix to a string
    :param arrays: list of ndarrays
    :param header: header"""

    max_rows = np.max([len(array) for array in arrays])
    rows = []
    delimiter = ","

    for i in range(max_rows):
        row_values = []
        for array in arrays:
            if i < len(array):
                row_values.append(f"{array[i]:.5E}")
            else:
                row_values.append("")
        rows.append(delimiter.join(row_values))

    string = "\n".join(rows)

    # Add the header
    if header is not None:
        string = delimiter.join(header) + "\n" + string

    return string


def get_header_as_dicts(
    header: list[str],
    delimiter: str | None = "\t",
) -> list[dict]:
    """Transform a list or numpy array of string corresponding to a file header to a list of dictionaries
    whose keys are the first column of the header and values are the header of each column
    :param header: list of strings each one corresponding to a line of the header
    :param delimiter: string delimiting two headers"""

    keys, *values = np.transpose([line.split(delimiter) for line in header])  # split headers
    return [dict(zip(keys, v)) for v in values]  # store in dictionaries


# --------------------------------------------------- DATA CONVERSION --------------------------------------------------


@st.cache_resource
def render_image(
    svg_file: str,
    width: int = 100,
    itype: str = "svg",
) -> str:
    """Render a svg file.
    :param str svg_file: file path
    :param int width: width in percent
    :param str itype: image type"""

    with open(svg_file, "rb") as ofile:
        svg = base64.b64encode(ofile.read()).decode()
        return (
            f'<center><img src="data:image/{itype}+xml;base64,{svg}" id="responsive-image" width="{width}%"/></center>'
        )


@st.cache_resource
def generate_download_link(
    data: tuple,
    header: None | list[str] | np.ndarray = None,
    text: str = "",
    name: str | None = None,
) -> str:
    """Generate a download link from a matrix and a header
    :param data: tuple containing x-axis and y-axis data
    :param header: list of strings corresponding to the header of each column
    :param text: text to be displayed instead of the link
    :param name: name of the file"""

    if name is None:
        name = text
    data = np.concatenate([[data[0][0]], data[1]])
    string = matrix_to_string(data, header)
    b64 = base64.b64encode(string.encode()).decode()
    return rf'<a href="data:text/csv;base64,{b64}" download="{name}.csv">{text}</a>'


def number_to_str(
    value: float | int | list[float | int] | None,
    n: int = 4,
    display: bool = False,
) -> str:
    """Convert a number to scientific notation without trailing zeros
    :param value: value to convert
    :param n: if the value is outside the range -1000 - 1000, number of floating digits
    :param display: if True, do not remove trailing zeros and use html"""

    if isinstance(value, (list, tuple, np.ndarray)):
        return ", ".join(number_to_str(v, n, display) for v in value)

    if isinstance(value, (int, np.integer)):
        return str(value)

    if value is None:
        return ""

    if value == 0:
        return "0"

    abs_val = abs(value)

    if 1e-2 <= abs_val <= 1000:
        if display:
            return f"{value:.{n}f}"
        else:
            return f"{value:.{n}g}"

    # Scientific notation
    order = int(math.floor(math.log10(abs_val)))
    mantissa = value / 10**order

    if display:
        mantissa_str = f"{mantissa:.{n}f}"
        return f"{mantissa_str} &#10005; 10<sup>{order}</sup>"
    else:
        mantissa_str = f"{mantissa:.{n}g}".rstrip("0").rstrip(".")
        return f"{mantissa_str}E{order}"


def generate_html_table(df: pd.DataFrame) -> str:
    """Generate an HTML table from a pandas DataFrame with merged cells for rows
    where all values are identical. Includes row names (index), column names,
    and displays the columns name in the upper-left corner cell.
    :param df: pandas DataFrame to convert to HTML"""

    html = ['<table border="1" style="border-collapse: collapse; text-align: center;">']

    # Add header row with columns name in the corner cell
    corner_cell_content = df.columns.name if df.columns.name else ""
    header = f'<tr><th style="padding: 8px; text-align: center;">{corner_cell_content}</th>'
    for col in df.columns:
        header += f'<th style="padding: 8px; text-align: center;">{col}</th>'
    header += "</tr>"
    html.append(header)

    # Process each row
    for idx, row in df.iterrows():
        values = row.tolist()

        row_html = f'<tr><td style="padding: 8px; font-weight: bold; text-align: center;">{idx}</td>'
        for val in values:
            row_html += f'<td style="padding: 8px; text-align: center;">{val}</td>'
        row_html += "</tr>"
        html.append(row_html)

    html.append("</table>")
    return '<div style="margin: auto; display: table;">' + "\n".join(html) + "</div>"


# ------------------------------------------------- DATA NORMALISATION -------------------------------------------------


def normalise(
    ndarray: np.ndarray,
    other: None | np.ndarray = None,
) -> np.ndarray:
    """Normalise a numpy array
    :param ndarray: ndarray of floats or ints
    :param other: if provided, normalise ndarray with respect to a"""

    if other is None:
        other = ndarray
    return ndarray / np.nanmax(other)


def feature_scale(
    ndarray: np.ndarray,
    a: float = 1.0,
    b: float = 0.0,
    other: None | np.ndarray = None,
) -> np.ndarray:
    """Feature scale a numpy array ndarray
    :param ndarray: ndarray of floats or ints
    :param other: if provided, normalise ndarray with respect to a
    :param a: minimum value
    :param b: maximum value"""

    if other is None:
        other = ndarray
    return b + (ndarray - np.nanmin(other)) * (a - b) / (np.nanmax(other) - np.nanmin(other))


# --------------------------------------------------- DATA EXTRACTION --------------------------------------------------


def get_data_index(
    content: list[str],
    delimiter: str | None = None,
) -> int | None:
    """Retrieve the index of the line where the data starts
    :param content: list of strings
    :param delimiter: delimiter of the float data"""

    for index, line in enumerate(content):

        if line != "" and line != delimiter and not line.isspace():
            try:
                [float(f) for f in line.split(delimiter) if not f.isspace() and f != ""]
                return index
            except ValueError:
                continue


def grep(
    content: list[str],
    string: str | list[str],
    line_nb: int | None = None,
    string2: str | None = None,
    data_type: str = "str",
    nb_expected: int | None = None,
) -> float | int | str | list | None:
    """Similar function to the bash grep
    :param content: list of strings where to search the data in
    :param string: string after which the data is located. if the string is in a list, only locate this specific string
    :param line_nb: amongst the lines which contains 'thing', number of the line to be considered
    :param string2: string before which the data is located
    :param data_type: type of the data to be returned
    :param nb_expected: exact number of 'string1' to be found"""

    if isinstance(string, list):
        found = [[f, g] for f, g in zip(content, range(len(content))) if string[0] == f]
    else:
        found = [[f, g] for f, g in zip(content, range(len(content))) if string in f]

    if len(found) == 0:
        return None  # Return None if nothing was found

    if isinstance(nb_expected, int):
        if len(found) != nb_expected:
            raise AssertionError()

    if line_nb is None:
        return found
    else:
        line = found[line_nb][0]

        if string2 is None:
            value = line[line.find(string) + len(string) :]
        else:
            index_beg = line.find(string) + len(string)
            index_end = line[index_beg:].find(string2) + index_beg
            value = line[index_beg:index_end]

        if data_type == "float":
            return float(value)
        if data_type == "int":
            return int(value)
        else:
            return value.strip()


# -------------------------------------------------- DATA MANIPULATION -------------------------------------------------


def sort_lists(
    alist: list | tuple | np.ndarray,
    *indices: int,
) -> np.ndarray:
    """Sort a list/tuple of np.ndarrays with respect to the ndarrays corresponding to the indices given
    :param alist: list/tuple of ndarrays of the same size
    :param indices: indices"""

    if not indices:
        indices = (0,)
    # noinspection PyArgumentList
    data_sorted = sorted(zip(*alist), key=itemgetter(*indices))
    return np.transpose(data_sorted)


def merge_dicts(*dictionaries: dict) -> dict:
    """Merge multiple dictionaries, keeping the first occurrence of each key.
    If a key appears in more than one dictionary, the value from the first dictionary containing the key will be used.
    """

    merged_dictionary = {}
    for dictionary in dictionaries:
        for key in dictionary.keys():
            if key not in merged_dictionary:
                merged_dictionary[key] = dictionary[key]

    return merged_dictionary


def interpolate_point(
    x: np.ndarray,
    y: np.ndarray,
    index: int,
    nb_point: int = 2,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate data around a given index
    :param x: ndarray corresponding to the x-axis
    :param y: ndarray corresponding to the y-axis
    :param index: index of the point
    :param nb_point: number of point to consider for the interpolation"""

    index1 = max([0, index - nb_point])
    index2 = min([len(x), index + nb_point + 1])
    x = x[index1:index2]
    y = y[index1:index2]

    # Interpolate the ys with the new x
    new_x = np.linspace(np.min(x), np.max(x), 1000)
    interp = sci.interp1d(x, y, **kwargs)
    return new_x, interp(new_x)


def interpolate_data(
    x: np.ndarray,
    y: np.ndarray,
    dx: int | float,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the y data of a signal
    :param x: x data
    :param y: y data
    :param dx: step (float) or array or number of points (int) of the new x data.
    :param kwargs: keyword arguments passed to the scipy interp1d function"""

    if dx <= 0:
        raise AssertionError("dx cannot be negative")

    if isinstance(dx, int):
        new_x = np.linspace(np.min(x), np.max(x), dx)
    else:
        new_x = np.arange(np.min(x), np.max(x), dx)

    # Interpolate the ys with the new x
    interp = sci.interp1d(x, y, **kwargs)
    new_y = interp(new_x)

    return new_x, new_y


def get_derivative(
    x: np.ndarray,
    y: np.ndarray,
    n: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the nth derivative of a ndarray
    :param np.ndarray x: x values
    :param np.ndarray y: y values (float or SignalData)
    :param int n: order of the derivative"""

    for i in range(n):
        dx = np.diff(x)
        dy = np.diff(y)
        y = dy / dx
        x = (x[1:] + x[:-1]) / 2

    return x, y


# ---------------------------------------------------- DATA CHECKING ---------------------------------------------------


def are_identical(
    obj1: any,
    obj2: any,
    rtol: float | None = None,
) -> bool:
    """Check if two objects have identical values. This does not check if the types are the same
    :param obj1: list or dictionary to compare.
    :param obj2: list or dictionary to compare.
    :param rtol: Relative tolerance for floating-point comparisons using np.allclose."""

    if isinstance(obj1, dict) and isinstance(obj2, dict):
        if obj1.keys() != obj2.keys():
            return False
        else:
            return all(are_identical(obj1[k], obj2[k], rtol) for k in obj1)

    elif isinstance(obj1, (list, tuple, np.ndarray)) and isinstance(obj2, (list, tuple, np.ndarray)):
        if len(obj1) != len(obj2):
            return False
        else:
            return all(are_identical(i1, i2, rtol) for i1, i2 in zip(obj1, obj2))

    else:
        if isinstance(obj1, float) and isinstance(obj2, float):
            if rtol is not None:
                return np.allclose(obj1, obj2, rtol=rtol, equal_nan=True)
            else:
                return np.array_equal(obj1, obj2, equal_nan=True)

        else:
            return obj1 == obj2


def are_close(*args, rtol=1e-3) -> bool:
    """Check if two objects are similar"""

    return are_identical(*args, rtol=rtol)
