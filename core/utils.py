""" utils module """

import numpy as np
from operator import itemgetter
import streamlit as st
import base64


def matrix_to_string(arrays, header=None):
    """ Convert a matrix array to a string
    :param np.ndarray, list arrays: list of ndarrays
    :param np.ndarray list header: header

    Example
    -------
    >>> arrays1 = [np.array([1, 2, 5]), np.array([1, 2])]
    >>> header1 = ['A', 'B']
    >>> matrix_to_string(arrays1, header1)
    'A\tB\n1.00000E+00\t1.00000E+00\t\n2.00000E+00\t2.00000E+00\t\n5.00000E+00\t\t'"""

    n = np.max([len(array) for array in arrays])
    rows = []
    for i in range(n):
        row = ''
        for array in arrays:
            if i < len(array):
                row += '%.5E\t' % array[i]
            else:
                row += '\t'
        rows.append(row)
    string = '\n'.join(rows)

    if header is not None:
        string = '\t'.join(header) + '\n' + string

    return string


@st.cache
def render_image(svg_file, width=100, itype='svg'):
    """ Render a svg file
    :param str svg_file: file path
    :param int width: width in percent
    :param str itype: image type"""

    with open(svg_file, "rb") as ofile:
        svg = base64.b64encode(ofile.read()).decode()
        return '<center><img src="data:image/%s+xml;base64,%s" id="responsive-image" width="%s%%"/></center>' % (itype, svg, width)


@st.cache
def generate_downloadlink(array, header=None, text=''):
    """ Generate a download link from a matrix and a header
    :param np.ndarray, list array: matrix
    :param None, list None header: list of strings corresponding to the header of each column
    :param str text: text to be displayed instead of the link """
    
    string = matrix_to_string(array, header)
    b64 = base64.b64encode(string.encode()).decode()
    return r'<a href="data:file/csv;base64,%s" width="30">' % b64 + text + '</a>'


def stringcolumn_to_array(raw_data, delimiter=None):
    """ Quickly convert a list of strings to ndarrays
    :param list of str raw_data: list of strings where each string contains as many numbers as there are columns
    :param str or None delimiter: delimiter separating two data in the strings

    Examples
    --------
    >>> stringcolumn_to_array(['1 2', '3 4'])
    array([[1., 3.],
           [2., 4.]])

    >>> stringcolumn_to_array(['1,2', '3,4'], ',')
    array([[1., 3.],
           [2., 4.]]) """

    lines = []
    for line in raw_data:
        line_float = []
        for string in line.split(delimiter):
            try:
                a = float(string)
            except ValueError:
                a = float('nan')
            line_float.append(a)
        lines.append(line_float)

    # Check the length of the lines matches
    for line in lines:
        while len(line) < len(lines[0]):
            line.append(float('nan'))

    return np.transpose(lines)


def get_data_index(content, delimiter=None):
    """ Retrieve the index of the line where the data starts
    :param list of str content: list of strings
    :param str or None delimiter: delimiter of the float data

    Example
    -------
    >>> get_data_index(['first line', 'second line', '1 2 3'])
    2"""

    for index, line in enumerate(content):

        if line != '':
            try:
                [float(f) for f in line.split(delimiter)]
                return index
            except ValueError:
                continue


def get_header_as_dicts(header, delimiter='\t'):
    """ Transform a list or numpy array of string corresponding to a file header to a list of dictionaries
    whose keys are the first column of the header and values are the header of each column
    :param list of str header: list of strings each one corresponding to a line of the header
    :param None, str delimiter: string delimiting two headers
    :return: list of dict

    Example
    -------
    >>> get_header_as_dicts(['Name 1,A,B', 'Unit,a,b'], ',')
    [{'Name 1': 'A', 'Unit': 'a'}, {'Name 1': 'B', 'Unit': 'b'}]

    >> get_header_as_dicts(['Name 1\tA\tB', 'Unit\ta\tb'], '\t')  # doctest does not work
    [{'Name 1': 'A', 'Unit': 'a'}, {'Name 1': 'B', 'Unit': 'b'}]"""

    keys, *values = np.transpose([line.split(delimiter) for line in header])  # split headers
    return [dict(zip(keys, v)) for v in values]  # store in dictionaries


def grep(content, string1, line_nb=False, string2=None, data_type='str', nb_found=None):
    """ Similar function to the bash grep
    :param list of str content: list of strings where to search the data in
    :param str or list string1: string after which the data is located. if the string is in a list, only locate this specific string
    :param int line_nb: amongst the lines which contains 'thing', number of the line to be considered
    :param str or bool string2: string before which the data is located
    :param str data_type: type of the data to be returned
    :param int or None nb_found: exact number of 'string1' to be found

    Examples
    --------
    >>> content1 = ['my first line', 'my second line', 'this number is 3']

    >>> grep(content1, 'first')
    [['my first line', 0]]

    >>> grep(content1, 'first', 0)
    'line'

    >>> grep(content1, 'my', 1, 'line', nb_found=2)
    'second'

    >>> grep(content1, 'number is', 0, data_type='int', nb_found=1)
    3"""

    if isinstance(string1, list):
        found = [[f, g] for f, g in zip(content, range(len(content))) if string1[0] == f]
    else:
        found = [[f, g] for f, g in zip(content, range(len(content))) if string1 in f]

    if len(found) == 0:
        return None  # Return None if nothing was found

    if isinstance(nb_found, int):
        if len(found) != nb_found:
            raise AssertionError()

    if line_nb is False:
        return found
    else:
        line = found[line_nb][0]

        if string2 is None:
            value = line[line.find(string1) + len(string1):]
        else:
            index_beg = line.find(string1) + len(string1)
            index_end = line[index_beg:].find(string2) + index_beg
            value = line[index_beg: index_end]

        if data_type == 'float':
            return float(value)
        if data_type == 'int':
            return int(value)
        else:
            return value.strip()


def sort(alist, *indices):
    """ Sort a list/tuple of np.ndarrays with respect to the ndarrays corresponding to the indices given
    :param list, tuple alist: list/tuple of ndarrays of the same size
    :param int indices: indices
    :return: np.ndarray of np.ndarray

    Example
    -------
    >>> sort([np.array([1, 4, 3, 5]), [2, 1, 3, 3]], 1)  # sort with respect to second ndarray
    array([[4, 1, 3, 5],
           [1, 2, 3, 3]])

    >>> sort([np.array([1, 4, 3, 5]), [2, 1, 3, 3], [7, 1, 6, 3]], 1, 2)
    array([[4, 1, 5, 3],
           [1, 2, 3, 3],
           [1, 7, 3, 6]])"""

    if not indices:
        indices = (0, )
    data_sorted = sorted(zip(*alist), key=itemgetter(*indices))
    return np.transpose(data_sorted)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
