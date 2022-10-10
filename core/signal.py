import numpy as np
import re
import plotly.graph_objects as go
import core.utils as utils
import scipy.signal as ss
import core.constants as pc
np.set_printoptions(threshold=7)


def get_label(string, dictionary, capitalize=True):
    """ Create a label from a string to display it with matplotlib
    :param str string: string
    :param dictionary: dictionary containing conversion for specific strings
    :param bool capitalize: if True, try to capitalize the label
    :return: str
    Note: The string should not contain the symbol '?'

    Examples
    --------
    >>> print(get_label('sub_sub^1 max. wavelength^2 sup^sup1 sup^{-3}', pc.quantities_label))
    Sub<sub>sub<\sub>^1 λ_{max}^2 sup^sup1 sup^{-3}

    >>> print(get_label('int. max. max. wavelength^2', pc.quantities_label))
    Int. max. λ_{max}^2

    >>> print(get_label('taureau', {'rap': r'\tau', 'taureau': 't|rap'}, False))
    t rap

    >>> print(get_label('test mum_test^2', pc.units_label, False))
    test mum<sub>test<\sub>^2"""

    label = ' ' + string + ' '  # initialise the label

    # Replace the strings that have a dictionary entry only if they start and end with a white space or _ or ^
    for item in sorted(dictionary.keys(), key=lambda a: - len(a)):
        condition_start = any(x + item in label for x in [' ', '_', '^'])
        condition_end = any(item + x in label for x in [' ', '_', '^'])
        if condition_start and condition_end:
            label = label.replace(item, '?' + dictionary[item])

    # Superscripts
    label = re.sub(r'\^([0-9a-zA-Z+-]+)', r'<sup>\1</sup>', label)

    # Subscripts
    label = re.sub(r'_([0-9a-zA-Z+-]+)', r'<sub>\1</sub>', label)

    # Remove whitespaces
    label = label.strip()

    # Capitalize
    if label[0] != '?' and capitalize and len(label.split('_')[0]) > 1:
        label = label[0].capitalize() + label[1:]

    label = label.replace('?', '')
    label = label.replace('|', ' ')

    return label


class Dimension(object):
    """ Dimension class

    Examples
    --------
    >>> dimension_B = Dimension([1, 2, 3])
    >>> dimension_C = Dimension(1)
    >>> dimension_D = Dimension(None)"""

    ARGKEYS = ('data', 'quantity', 'unit')

    def __init__(self, data, quantity='', unit=''):
        """ Object initialisation
        :param data: value or np.ndarray of values
        :param str quantity: quantity
        :param str unit: unit """

        if not isinstance(data, np.ndarray):
            self.data = np.array([data])
        else:
            self.data = data
        self.quantity = quantity
        self.unit = unit

    @property
    def kwargs(self):
        """ Return the keyword arguments of the Dimension """

        return dict(data=self.data, quantity=self.quantity, unit=self.unit)

    def __call__(self, *args, **kwargs):
        """ __call__
        NOTE: THIS IS 3 TIMES SLOWER THAN CALLING DIMENSION CLASS

        Examples
        --------
        >>> = dimension3 = Dimension([-1., 1., 2., 3., 3.5], 'quantity', 'unit')

        >>> dimension3([1, 3], unit='unit2')
        Dimension([1 3], quantity, unit2)

        >>> dimension3([1, 3], data=[1, 4], unit='unit2')
        Dimension([1 3], quantity, unit2)"""

        args = dict(zip(self.ARGKEYS, args))
        return Dimension(**utils.merge_dicts(args, kwargs, self.kwargs))

    def get_quantity_label(self):
        """ Get the quantity label """

        if self.quantity:
            if sum([c.isalpha() for c in self.quantity]) == 1:
                capitalise = False
            else:
                capitalise = True

            return get_label(self.quantity, pc.quantities_label, capitalise)
        else:
            return ''

    def get_unit_label(self):
        """ Get the unit label"""

        if self.unit:
            return get_label(self.unit, pc.units_label, False)
        else:
            return ''

    def get_axis_label_html(self):
        """ Return the quantity and unit as a string

        Example
        -------
        >>> Dimension([-1., 1., 2., 3., 3.5], 'quantity', 'unit').get_axis_label_html()
        'Quantity (unit)'"""

        if self.unit:
            return self.get_quantity_label() + ' (' + self.get_unit_label() + ')'
        else:
            return self.get_quantity_label()

    def get_value_label_html(self):
        """ Return the value label for display

        Example
        -------
        >>> Dimension([-1., 1., 2., 3., 3.5], 'quantity', 'unit').get_value_label_html()
        'Quantity: -1 unit'"""

        return self.get_quantity_label() + ': ' + '%g' % self.data[0] + ' ' + self.get_unit_label()

    def get_label(self):
        """ Get a simple label of the dimension """

        if self.unit:
            return f'{self.quantity} ({self.unit})'
        else:
            return self.quantity

    def get_value_label(self):

        return self.quantity + ': ' + '%g' % self.data[0] + ' ' + self.unit

    def __repr__(self):
        """ __repr__

        Examples
        --------
        >>> Dimension(3.)
        Dimension(3.0)

        >>> Dimension(3., 'quantity', 'unit')
        Dimension(3.0, quantity, unit)

        >>> Dimension(np.array([1., 2., 3., 3.5]), '', 'unit')
        Dimension([1.  2.  3.  3.5], unit)"""

        # Data
        string = 'Dimension(%s' % self.data

        # Quantity, unit, name and z_dict
        if self.quantity:
            string += ', %s' % self.quantity
        if self.unit:
            string += ', %s' % self.unit

        return string + ')'


class SignalData(object):

    def __init__(self, x, y, name='', z_dict=None):
        """
        :param Dimension x: x dimension
        :param Dimension y: y dimension
        :param str name: name
        :param None, dict z_dict: information about the signal """

        self.x = x
        self.y = y
        self.name = name
        if z_dict is None:
            self.z_dict = {}
        else:
            self.z_dict = z_dict

    def print(self):
        """ Print the signal dimensions """

        print(self.x)
        print(self.y)

    # noinspection PyArgumentList
    def plot(self, figure, position=None):
        """ Plot the signal data in a plotly figure
        :param figure: plotly figure object
        :param None, tuple, list position: subplot position """

        trace = go.Scatter(x=self.x.data, y=self.y.data, name=self.name, showlegend=True)

        if position:
            kwargs = dict(row=position[0], col=position[1])
        else:
            kwargs = dict()

        figure.add_trace(trace, **kwargs)

        figure.update_xaxes(title_text=self.x.get_axis_label_html(), tickformat=',', **kwargs)
        figure.update_yaxes(title_text=self.y.get_axis_label_html(), **kwargs)

    def smooth(self, *args):
        """ Smooth the y data """

        # noinspection PyBroadException
        try:
            y = ss.savgol_filter(self.y.data, *args)
            return SignalData(self.x, self.y(data=y), self.name, self.z_dict)
        except:
            return self

    def reduce_range(self, xrange):
        """ Reduce the x range """

        index1 = np.abs(self.x.data - xrange[0]).argmin()
        index2 = np.abs(self.x.data - xrange[1]).argmin()
        return SignalData(self.x(data=self.x.data[index1: index2]), self.y(data=self.y.data[index1: index2]), self.name, self.z_dict)

    def get_max(self):
        """ Get the maximum intensity signal """

        return get_point(self.x, self.y, 'max')

    def get_min(self):
        """ Get the minimum intensity signal """

        return get_point(self.x, self.y, 'min')

    def get_extrema(self):
        """ Get the extremum intensity signal """

        M = self.get_max()
        m = self.get_min()
        if abs(M[1].data) < abs(m[1].data):  # accessing data avoid pycharm from highlighting error
            x_ext = m[0](quantity=m[0].quantity.replace(pc.min_qt, 'extremum'))
            y_ext = m[1](quantity=m[1].quantity.replace(pc.min_qt, 'extremum'))
            i_ext = m[2]
        else:
            x_ext = M[0](quantity=M[0].quantity.replace(pc.max_qt, 'extremum'))
            y_ext = M[1](quantity=M[1].quantity.replace(pc.max_qt, 'extremum'))
            i_ext = M[2]
        return x_ext, y_ext, i_ext

    def get_fwhm(self, interpolation=False):
        """ Get the signal FWHM. Calculate the maximum point if not already stored """

        x_ext, y_ext, i_ext = self.get_extrema()

        m = i_ext.data[0]
        if m != len(self.y.data):
            x_right, y_right = get_halfint_point(self.x.data[m:], self.y.data[m:], interpolation)
        else:
            x_right, y_right = None, None

        if m != 0:
            x_left, y_left = get_halfint_point(self.x.data[:m + 1], self.y.data[:m + 1], interpolation)
        else:
            x_left, y_left = None, None

        if x_left and x_right:
            fwhm = x_right - x_left
        else:
            fwhm = 0

        return self.x(data=np.abs(fwhm), quantity=pc.fwhm_qt), self.x(data=x_left), self.y(data=y_left), self.x(data=x_right), self.y(data=y_right)


def get_point(x, y, point):
    """ Get the extrema point of a signal
    :param Dimension x: x dimension
    :param Dimension y: y dimension
    :param str point: 'max', 'min'

    Example
    -------
    >>> x1 = Dimension(np.arange(-10, 10, 0.32), 'X quantity', 'X unit')
    >>> gaussian = lambda x_, height, mean, fwhm: height * np.exp(- (x_ - mean)**2 / (2 * (fwhm / (2 * np.sqrt(2 * np.log(2))))**2))
    >>> y1 = Dimension(gaussian(x1.data, 10., 0., 3.), 'Y quantity', 'Y unit')
    >>> print(*get_point(x1, y1, 'max'))
    Dimension(-0.07999999999999119, max. X quantity, X unit) Dimension(9.98030323716376, max. Y quantity, Y unit) Dimension(31)"""

    if point == 'min':
        method, quantity = 'argmin', pc.min_qt
    else:
        method, quantity = 'argmax', pc.max_qt
    index = getattr(y.data, method)()

    # Point dimensions
    x_e = x(data=x.data[index], quantity=quantity + ' ' + x.quantity)
    y_e = y(data=y.data[index], quantity=quantity + ' ' + y.quantity)

    return x_e, y_e, Dimension(index)


def get_halfint_point(x, y, interpolation=True, halfint=None):
    """ Calculate the half intensity point
    :param np.ndarray x: x dimension
    :param np.ndarray y: y dimension
    :param bool interpolation: if True, use interpolation to improve the calculation
    :param None, float halfint: half intensity

    Example
    -------
    >>> x1 = np.linspace(-1, 10, 101)[:19]
    >>> gaussian = lambda x_, height, mean, fwhm: height * np.exp(- (x_ - mean)**2 / (2 * (fwhm / (2 * np.sqrt(2 * np.log(2))))**2))
    >>> y1 = gaussian(x1, 10., 1, 3.)[:19]
    >>> get_halfint_point(x1, y1)
    (-0.22999999999999998, 6.27462002114742)"""

    # Calculate the half intensity
    if not halfint:
        y_min = np.min(y)
        y_max = np.max(y)

        # Determine if the peak is upside down
        if np.abs(y_min) > np.abs(y_max):
            halfint = (y_min - y_max) / 2. + y_max
        else:
            halfint = (y_max - y_min) / 2. + y_min

    # Find the closest points to the half intensity
    index = np.abs(y - halfint).argsort()[0]

    # Interpolation
    if interpolation:
        try:
            x_int, y_int = utils.interpolate_point(x, y, index)
            return get_halfint_point(x_int, y_int, False, halfint)
        except IndexError:
            print('Interpolation failed')
            pass

    return x[index], y[index]


if __name__ == '__main__':
    import doctest
    doctest.testmod()
