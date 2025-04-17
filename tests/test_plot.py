"""Test module for the functions in the `plot.py` module.

This module contains unit tests for the functions implemented in the `plot.py` module. The purpose of these tests is to
ensure the correct functionality of each function in different scenarios and to validate that the expected outputs are
returned.

Tests should cover various edge cases, valid inputs, and any other conditions that are necessary to confirm the
robustness of the functions."""

import numpy as np
import pytest

from app.plot import *
from app.signaldata import Dimension


class TestSubplots:

    def test_subplots_shape(self) -> None:
        """Test the shape of the subplots returned by subplots function."""

        figure, positions = subplots(6)
        assert len(positions) == 6
        assert len(positions[0]) == 2  # (row, col)

    def test_subplots_with_max_columns(self) -> None:
        """Test subplots when max columns are specified."""

        figure, positions = subplots(6, m=1)
        assert len(positions) == 6


class TestPlot:

    @pytest.fixture
    def signal_data(self) -> SignalData:
        """Fixture to provide a SignalData object."""

        x_data = np.array([1, 2, 3, 4, 5])
        y_data = np.array([10, 20, 30, 40, 50])
        x_dim = Dimension(x_data, quantity="Time", unit="s")
        y_dim = Dimension(y_data, quantity="Intensity", unit="a.u.")
        return SignalData(x_dim, y_dim, name="Signal1")

    def test_plot_single_signal(self, signal_data) -> None:
        """Test plotting a single SignalData object."""

        figure = plot_signals(signal_data)
        assert isinstance(figure, go.Figure)
        assert len(figure.data) == 1  # Should contain exactly one trace

    def test_plot_multiple_signals(self, signal_data) -> None:
        """Test plotting multiple SignalData objects."""

        figure = plot_signals([signal_data, signal_data])
        assert isinstance(figure, go.Figure)
        assert len(figure.data) == 2  # Should contain exactly two traces

    def test_plot_signal_dict(self, signal_data) -> None:
        """Test plotting a dictionary of SignalData objects."""

        signals = {"Signal1": signal_data, "Signal2": signal_data}
        figure = plot_signals(signals)
        assert isinstance(figure, go.Figure)
        assert len(figure.data) == 2  # Should contain two traces

    def test_plot_with_existing_figure(self, signal_data) -> None:
        """Test plotting onto an existing figure."""

        existing_figure = go.Figure()
        figure = plot_signals(signal_data, figure=existing_figure)
        assert isinstance(figure, go.Figure)
        assert len(figure.data) == 1  # Should contain exactly one trace, as we added one signal

    def test_plot_signal_dict_list(self, signal_data) -> None:
        """Test plotting a dictionary of SignalData objects."""

        signals = {"Signal1": [signal_data, signal_data], "Signal2": signal_data}
        figure = plot_signals(signals)
        assert isinstance(figure, go.Figure)
        assert len(figure.data) == 3  # Should contain two traces
