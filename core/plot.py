""" plot module """

import itertools
import math

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as ps

from core.signal import SignalData


def subplots(
        n: int,
        m: int | None = None,
        **kwargs,
) -> tuple[go.Figure, list[tuple[int, int]]]:
    """Create n subplots
    :param n: number of subplots
    :param m: if int, maximum number of columns
    :param kwargs: keyword arguments passed to plotly.subplots.make_subplots"""

    nb_cols = int(np.sqrt(n))
    if isinstance(m, int) and nb_cols > m:
        nb_cols = m
    nb_rows = int(math.ceil(n / nb_cols))
    positions = list(itertools.product(range(1, nb_rows + 1), range(1, nb_cols + 1)))[:n]
    return ps.make_subplots(rows=nb_rows, cols=nb_cols, **kwargs), positions


def plot(signals: dict[str, list[SignalData]] | dict[str, SignalData] | list[SignalData] | SignalData,
         position: list | None = None,
         figure: go.Figure | None = None,
         *args,
         **kwargs):
    """Plot signals in a Plotly figure.
    :param signals: Signal data to plot. Can be:
            - A single SignalData object
            - A list of SignalData objects (plotted in the same subplot)
            - A dictionary of signal names mapping to SignalData objects or lists of SignalData objects
              (each key will be plotted in its own subplot)
    :param position: Position in the figure to plot at, format [row, col]. If None and
        plotting a single signal or list, plots in the main figure.
    :param figure: Existing figure to plot on. If None, creates a new figure.
    :param args: Additional positional arguments passed to the SignalData.plot method
    :param kwargs: Additional keyword arguments passed to the SignalData.plot method"""

    # Dict: data type is different, plot in multiple subplots
    if isinstance(signals, dict):
        figure, positions = subplots(len(signals))
        for key, position in zip(signals, positions):
            plot(signals[key], position, figure, *args, **kwargs)

    else:

        if figure is None:
            figure = go.Figure()

        # List/tuple of signals, plot all in the same figure
        if isinstance(signals, (list, tuple)):
            for signal in signals:
                signal.plot(figure, position, *args, **kwargs)

        else:
            signals.plot(figure, position, *args, **kwargs)

    figure.update_layout(height=800)
    return figure
