"""Test module for the functions in the `plot.py` module.

This module contains unit tests for the functions implemented in the `plot.py` module. The purpose of these tests is to
ensure the correct functionality of each function in different scenarios and to validate that the expected outputs are
returned.

Tests should cover various edge cases, valid inputs, and any other conditions that are necessary to confirm the
robustness of the functions."""

import numpy as np
import plotly.graph_objs as go
import plotly.subplots as ps
import pytest

from data_files.signal_data import Dimension, SignalData
from interface.plot import plot_signals, make_figure, scatter_plot


@pytest.fixture
def signal_data() -> SignalData:
    """Fixture to provide a SignalData object."""

    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([10, 20, 30, 40, 50])
    x_dim = Dimension(x_data, quantity="Time", unit="s")
    y_dim = Dimension(y_data, quantity="Intensity", unit="a.u.")
    return SignalData(x_dim, y_dim, filename="Signal1")


class TestPlot:

    def test_plot_single_signal(self, signal_data) -> None:
        """Test plotting a single SignalData object."""

        figure = ps.make_subplots(1, 1)
        plot_signals(signal_data, figure)
        assert isinstance(figure, go.Figure)
        assert len(figure.data) == 1  # Should contain exactly one trace

    def test_plot_multiple_signals(self, signal_data) -> None:
        """Test plotting multiple SignalData objects."""

        figure = ps.make_subplots(1, 1)
        plot_signals([signal_data, signal_data], figure)
        assert isinstance(figure, go.Figure)
        assert len(figure.data) == 2  # Should contain exactly two traces


class TestMakeFigure:

    def test_yaxis(self) -> None:

        figure = make_figure()
        assert isinstance(figure, go.Figure)

    def test_yaxis2(self, signal_data) -> None:

        figure = make_figure()
        signal_data.plot(figure=figure, secondary_y=True)
        layout = figure.layout
        assert layout["yaxis2"]


class TestScatterPlot:

    def setup_method(self) -> None:
        """Create a new figure for each test"""

        self.fig = go.Figure()

    def test_single_point_defaults(self) -> None:

        scatter_plot(self.fig, 1.0, 2.0, "test_label")
        trace = self.fig.data[0]

        assert isinstance(trace, go.Scattergl)
        assert trace.x == (1.0,)
        assert trace.y == (2.0,)
        assert trace.mode == "markers"
        assert trace.name == "test_label"
        assert trace.marker.size == 15

    def test_multiple_points_with_marker(self) -> None:

        x = [1, 2, 3]
        y = [4, 5, 6]
        custom_marker = {"color": "red", "size": 10}

        scatter_plot(self.fig, x, y, "series", marker=custom_marker)
        trace = self.fig.data[0]

        assert trace.mode == "markers+lines"
        assert trace.marker.color == "red"
        assert trace.marker.size == 10

    def test_kwargs_passed_to_trace(self) -> None:

        scatter_plot(self.fig, [1, 2], [3, 4], "label", line=dict(dash="dash"))
        trace = self.fig.data[0]

        assert trace.line.dash == "dash"

    def test_float_list_handling(self) -> None:

        scatter_plot(self.fig, 5.0, [10.0], "mixed_type")
        trace = self.fig.data[0]

        assert trace.x == (5.0,)
        assert trace.y == (10.0,)
        assert trace.mode == "markers"
