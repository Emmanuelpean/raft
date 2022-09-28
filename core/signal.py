import numpy as np
import re
import plotly.graph_objects as go

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

        self.data = data
        self.quantity = quantity
        self.unit = unit

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

    def get_axis_label(self):
        """ Return the quantity and unit as a string

        Example
        -------
        >>> Dimension([-1., 1., 2., 3., 3.5], 'quantity', 'unit').get_axis_label()
        'Quantity (unit)'"""

        if self.unit:
            return self.get_quantity_label() + ' (' + self.get_unit_label() + ')'
        else:
            return self.get_quantity_label()

    def get_value_label(self):
        """ Return the value label for display

        Example
        -------
        >>> Dimension([-1., 1., 2., 3., 3.5], 'quantity', 'unit').get_value_label()
        'Quantity: -1 unit'"""

        return self.get_quantity_label() + ': ' + '%g' % self.data[0] + ' ' + self.get_unit_label()

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

        figure.update_xaxes(title_text=self.x.get_axis_label(), tickformat=',', **kwargs)
        figure.update_yaxes(title_text=self.y.get_axis_label(), **kwargs)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
