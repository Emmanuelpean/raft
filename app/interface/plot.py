"""Module containing functions for plotting data"""

import plotly.colors as pc
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots as ps

from utils.miscellaneous import merge_dicts

# Define your custom colour list
PASTEL_COLORS = [
    "#1F78B4",  # stronger blue
    "#66C2A5",  # greenish pastel
    "#FC8D62",  # coral pastel
]

# Set it globally for all future plots
pio.templates["plotly"].layout.colorway = PASTEL_COLORS


def make_figure(secondary_y: bool = True) -> go.Figure:
    """Create a plotly figure
    :param secondary_y: if True, add a secondary y-axis"""

    return ps.make_subplots(1, 1, specs=[[{"secondary_y": secondary_y}]])


def plot_signals(
    signals: "SignalData | list[SignalData]",
    figure: go.Figure,
    colorscale: str = "viridis",
    *args,
    **kwargs,
) -> go.Figure:
    """Plot signals in a Plotly figure.
    :param signals: signal or list of signals to plot.
    :param figure: existing figure to plot on
    :param colorscale: colorscale to use if a list of signals is plotted
    :param args: positional arguments passed to the SignalData.plot method
    :param kwargs: keyword arguments passed to the SignalData.plot method"""

    # List/tuple of signals, plot all in the same figure
    if isinstance(signals, (list, tuple)):

        # Generate the colours
        if len(signals) > 1:
            values = [i / len(signals) for i in range(len(signals))]
            colours = pc.sample_colorscale(colorscale, values)

            # Plot the signals
            for signal, colour in zip(signals, colours):
                signal.plot(figure, *args, **merge_dicts(kwargs, dict(line=dict(color=colour))))

        else:
            signals[0].plot(figure, *args, **merge_dicts(kwargs, dict(line=dict(width=2.6))))

    else:
        signals.plot(figure, *args, **kwargs)

    return figure


def scatter_plot(
    figure: go.Figure,
    x_data: list | float,
    y_data: list | float,
    label: str,
    **kwargs,
) -> None:
    """Simple scatter plot
    :param figure: plotly figure
    :param x_data: x-axis data
    :param y_data: y-axis data
    :param label: data label
    :param kwargs: keyword arguments passed to Scatter"""

    if isinstance(x_data, float):
        x_data = [x_data]
    if isinstance(y_data, float):
        y_data = [y_data]
    if len(x_data) == 1:
        mode = "markers"
    else:
        mode = "markers+lines"

    kwargs = merge_dicts(kwargs, dict(marker=dict(size=15)))
    trace = go.Scattergl(
        x=x_data,
        y=y_data,
        mode=mode,
        name=label,
        **kwargs,
    )
    figure.add_trace(trace)
