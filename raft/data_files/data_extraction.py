"""Module containing functions for extracting information from data files"""

import numpy as np


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


def get_precision(string) -> tuple[float, int]:
    """Get a number string precision.
    :param string: float or int string"""

    if "." in string:
        value, decimals = string.split(".")
        precision = len(decimals)
        return float(string), precision
    else:
        return float(string), 0


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
    return None
