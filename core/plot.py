""" plot module """

import plotly.graph_objects as go
import plotly.subplots as ps

import numpy as np
import math
import itertools


def subplots(n, m=None, **kwargs):
    """ Create n subplots
    :param int n: number of subplots
    :param None, int m: if int, maximum number of columns
    :param kwargs: keyword arguments passed to plotly.subplots.make_subplots

    Example
    -------
    >>> subplots(3)[1]
    [(1, 1), (2, 1), (3, 1)]
    >>> subplots(9, 2)[1]
    [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1)]"""

    nb_cols = int(np.sqrt(n))
    if isinstance(m, int) and nb_cols > m:
        nb_cols = m
    nb_rows = int(math.ceil(n / nb_cols))
    positions = list(itertools.product(range(1, nb_rows + 1), range(1, nb_cols + 1)))[:n]
    return ps.make_subplots(rows=nb_rows, cols=nb_cols, **kwargs), positions


def plot(signals, position=None, figure=None):
    """ Plot signals """

    # Dict: data type is different, plot in multiple subplots
    if isinstance(signals, dict):
        figure, positions = subplots(len(signals))
        for key, position in zip(signals, positions):
            plot(signals[key], position, figure)

    else:

        if figure is None:
            figure = go.Figure()

        # List/tuple of signals, plot all in the same figure
        if isinstance(signals, (list, tuple)):
            for signal in signals:
                signal.plot(figure, position)

        else:
            signals.plot(figure, position)

    figure.update_layout(height=800)
    return figure


if __name__ == '__main__':
    import doctest
    doctest.testmod()
