"""plot module"""

import itertools
import math

import plotly.graph_objects as go
import plotly.subplots as ps

from signaldata import SignalData
from utils import merge_dicts


def subplots(
    nb_rows: int,
    nb_cols: int = 0,
    specs: None | list = None,
    **kwargs,
) -> tuple:
    """Create n subplots
    :param nb_rows: number of subplots
    :param nb_cols: number of columns
    :param specs:
    :param kwargs: keyword arguments passed to plotly.subplots.make_subplots"""

    if not nb_cols:
        n = math.isqrt(nb_rows)
        m = math.ceil(nb_rows / n)
    else:
        n, m = nb_rows, nb_cols
    if specs:
        specs = [specs[i : i + n] for i in range(0, len(specs), n)]
    positions = list(itertools.product(range(1, n + 1), range(1, m + 1)))
    return ps.make_subplots(rows=n, cols=m, specs=specs, **kwargs), positions


def plot_signals(
    signals: SignalData | list[SignalData],
    figure: go.Figure | None = None,
    *args,
    **kwargs,
) -> go.Figure:
    """Plot signals in a Plotly figure.
    :param signals: Signal data to plot. Can be:
            - A single SignalData object
            - A list of SignalData objects (plotted in the same subplot)
            - A dictionary of signal names mapping to SignalData objects or lists of SignalData objects
              (each key will be plotted in its own subplot)
    :param figure: Existing figure to plot on. If None, creates a new figure.
    :param args: Additional positional arguments passed to the SignalData.plot method
    :param kwargs: Additional keyword arguments passed to the SignalData.plot method"""

    # List/tuple of signals, plot all in the same figure
    if isinstance(signals, (list, tuple)):
        for signal in signals:
            signal.plot(figure, *args, **kwargs)

    else:
        signals.plot(figure, *args, **kwargs)

    return figure


def scatter_plot(
    figure: go.Figure,
    x_data: list | float,
    y_data: list | float,
    label: str,
    marker: dict | None = None,
    **kwargs,
) -> None:
    """Simple scatter plot
    :param figure: plotly figure
    :param x_data: x-axis data
    :param y_data: y-axis data
    :param label: data label
    :param marker: marker settings
    :param kwargs: keyword arguments passed to Scatter"""

    if isinstance(x_data, float):
        x_data = [x_data]
    if isinstance(y_data, float):
        y_data = [y_data]
    if len(x_data) == 1:
        mode = "markers"
    else:
        mode = "markers+lines"

    if marker is None:
        marker = dict()
    marker = merge_dicts(marker, dict(size=15))
    trace = go.Scattergl(
        x=x_data,
        y=y_data,
        mode=mode,
        name=label,
        marker=marker,
        **kwargs,
    )
    figure.add_trace(trace)
