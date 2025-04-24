"""Module containing functions for generating or modifying strings"""

import math

import numpy as np
import pandas as pd


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
    corner_cell_content = df.columns.filename if df.columns.filename else ""
    header = (
        f'<tr><th style="padding: 8px; text-align: center;">{corner_cell_content}</th>'
    )
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


def dedent(string: str) -> str:
    """Dedent a string
    :param string: string to be dedented"""

    return "\n".join([m.lstrip() for m in string.split("\n")])
