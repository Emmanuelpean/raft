""" utils module """

import numpy as np
from operator import itemgetter
import streamlit as st
import base64
import scipy.interpolate as sci


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

        if line != '' and line != delimiter and not line.isspace():
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


def merge_dicts(*dictionaries):
    """ Merge multiple dictionaries. Last argument can be a boolean determining if the dicts can erase each other's content

    Examples
    --------
    >>> dict1 = {'label': 'string'}
    >>> dict2 = {'c': 'r', 'label': 'something'}

    >>> merge_dicts(dict1, dict2, False)
    {'label': 'string', 'c': 'r'}

    >>> merge_dicts(dict1, dict2, True)
    {'label': 'something', 'c': 'r'}

    >>> merge_dicts(dict1, dict2)
    {'label': 'string', 'c': 'r'}"""

    # Argument for force rewriting the dictionaries
    if isinstance(dictionaries[-1], bool):
        force = dictionaries[-1]
        dictionaries = dictionaries[:-1]
    else:
        force = False

    merged_dictionary = {}
    for dictionary in dictionaries:
        if force:
            merged_dictionary.update(dictionary)
        else:
            for key in dictionary.keys():
                if key not in merged_dictionary:
                    merged_dictionary[key] = dictionary[key]

    return merged_dictionary


def interpolate_point(x, y, index, nb_point=2, **kwargs):
    """ Interpolate data around a given index
    :param np.ndarray x: ndarray corresponding to the x axis
    :param np.ndarray y: ndarray corresponding to the y axis
    :param int index: index of the point
    :param int nb_point: number of point to consider for the interpolation

    Example
    -------
    >>> import matplotlib
    >>> matplotlib.use('TkAgg')
    >>> import matplotlib.pyplot as plt
    >>> x1 = np.array([-10, -7, -5, -1, 5, 7, 10])
    >>> y1 = x1 ** 2
    >>> x_int, y_int = interpolate_point(x1, y1, 3, kind='cubic')
    >>> ax = plt.figure().add_subplot(111)
    >>> l1 = ax.plot(x1, y1, 'x', ms=30, mew=3)
    >>> l2 = ax.plot(x_int, y_int, lw=5)
    >>> t1 = ax.set_title('interpolate_point') """

    index1 = max([0, index - nb_point])
    index2 = min([len(x), index + nb_point + 1])
    x = x[index1: index2]
    y = y[index1: index2]
    # Interpolate the ys with the new x
    new_x = np.linspace(np.min(x), np.max(x), 1000)
    # noinspection PyArgumentList
    interp = sci.interp1d(x, y, **kwargs)
    return new_x, interp(new_x)


def concatenate_data(data):
    """ Concatenate a list containing either lists, dicts or others """

    # List of dicts
    if isinstance(data[0], dict):
        keys = np.unique(np.concatenate([list(s.keys()) for s in data]))
        return {key: concatenate_data([s[key] for s in data if key in s]) for key in keys}

    # List of list
    elif isinstance(data[0], list):
        return list(np.concatenate(data))

    else:
        return data


if __name__ == '__main__':
    import doctest
    doctest.testmod()
