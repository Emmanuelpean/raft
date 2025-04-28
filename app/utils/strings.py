"""Module containing functions for generating or modifying strings"""

import datetime as dt
import math
import re

import numpy as np
import pandas as pd


def matrix_to_string(
    arrays: np.ndarray | list[np.ndarray],
    header: None | list[str] | list[list[str]] = None,
) -> str:
    """Convert a matrix to a string
    :param arrays: list of ndarrays
    :param header: header"""

    # # Faster version
    # buffer = io.StringIO()
    # if isinstance(header, list) and isinstance(header[0], str):
    #     header = [header]
    # if isinstance(header, list):
    #     buffer.write("\n".join([",".join(head) for head in header]) + "\n")
    # pd.DataFrame(np.transpose(arrays)).to_csv(buffer, index=False)
    # return buffer.getvalue()

    max_rows = np.max([len(array) for array in arrays])
    rows = []
    delimiter = ","

    for i in range(max_rows):
        row_values = []
        for array in arrays:
            if i < len(array):
                if isinstance(array[i], dt.datetime):
                    row_values.append(str(array[i]))
                else:
                    row_values.append(f"{array[i]:.5E}")
            else:
                row_values.append("")
        rows.append(delimiter.join(row_values))

    string = "\n".join(rows)

    # Add the header
    if isinstance(header, list) and isinstance(header[0], str):
        header = [header]
    if isinstance(header, list):
        header_string = "\n".join([delimiter.join(head) for head in header]) + "\n"
    else:
        header_string = ""

    return header_string + string


def number_to_str(
    value: float | int | list[float | int] | None,
    precision: int = 5,
    format_str: str = "f",
    html: bool = False,
    auto_exponent: bool = True,
) -> str:
    """Convert a number to scientific notation without trailing zeros
    :param value: value to convert
    :param format_str: string format
    :param precision: if the value is outside the range -1000 - 1000, number of floating digits
    :param html: if True, generate html string
    :param auto_exponent: if True, automatically switch to exponent notation if the abs value is higher than 1e4 or
                          lower than 1e-4"""

    if isinstance(value, (list, tuple, np.ndarray)):
        return ", ".join(number_to_str(v, precision, format_str, html, auto_exponent) for v in value)

    elif value is None:
        return ""

    elif np.isnan(value):
        return ""

    elif value == 0:
        return "0"

    else:

        string = f"{value:.{precision}{format_str}}"

        if not auto_exponent:
            return string

        else:
            abs_val = abs(value)
            if 1e-2 <= abs_val <= 1000:
                return string

            # Exponent notation
            order = int(math.floor(math.log10(abs_val)))
            mantissa = value / 10**order
            mantissa_str = f"{mantissa:.{precision}{format_str}}"

            if html:
                return f"{mantissa_str} &#10005; 10<sup>{order}</sup>"
            else:
                return f"{mantissa_str}E{order}"


def generate_html_table(df: pd.DataFrame) -> str:
    """Generate an HTML table from a pandas DataFrame with merged cells for rows
    where all values are identical. Includes row names (index), column names,
    and displays the columns name in the upper-left corner cell.
    :param df: pandas DataFrame to convert to HTML"""

    html = ['<table border="1" style="border-collapse: collapse; text-align: center;">']

    # Add header row with columns name in the corner cell
    corner_cell_content = df.columns.filename if df.columns.filename else ""
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


def dedent(string: str) -> str:
    """Dedent a string
    :param string: string to be dedented"""

    return "\n".join([m.lstrip() for m in string.split("\n")])


def get_label_html(
    string: str,
    dictionary: dict[str, str],
    capitalize: bool = True,
) -> str:
    """Create a label from a string to display it with matplotlib
    :param string: string
    :param dictionary: dictionary containing conversion for specific strings
    :param capitalize: if True, try to capitalize the label
    Note: The string should not contain the symbol '?'"""

    if string == "":
        return string

    if "?" in string:
        raise AssertionError("? should not be in the string")

    label = " " + string + " "  # initialise the label

    # Replace the strings that have a dictionary entry only if they start and end with a white space or _ or ^
    for item in sorted(dictionary.keys(), key=lambda a: -len(a)):
        condition_start = any(x + item in label for x in [" ", "_", "^"])
        condition_end = any(item + x in label for x in [" ", "_", "^"])
        if condition_start and condition_end:
            label = label.replace(item, "?" + dictionary[item])

    # Superscripts
    label = re.sub(r"\^([0-9a-zA-Z+-]+)", r"<sup>\1</sup>", label)

    # Subscripts
    label = re.sub(r"_([0-9a-zA-Z+-]+)", r"<sub>\1</sub>", label)

    # Remove whitespaces
    label = label.strip()

    # Capitalize
    if label[0] != "?" and capitalize and len(label.split("_")[0]) > 1:
        label = label[0].capitalize() + label[1:]

    label = label.replace("?", "")
    label = label.replace("|", " ")

    return label
