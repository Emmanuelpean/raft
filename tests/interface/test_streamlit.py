"""Test module for the functions in the `interface/streamlit.py` module.

This module contains unit tests for the functions implemented in the `interface/streamlit.py` module. The purpose of these tests is to
ensure the correct functionality of each function in different scenarios and to validate that the expected outputs are
returned.

Tests should cover various edge cases, valid inputs, and any other conditions that are necessary to confirm the
robustness of the functions."""

from unittest.mock import patch, MagicMock

import numpy as np
import plotly.graph_objects as go
import pytest

from data_files.signal_data import Dimension, SignalData
from interface.streamlit import tab_bar, display_data


class TestTabBar:
    """Tests for the tab_bar function."""

    @patch("streamlit.radio")
    def test_tab_bar_returns_active_tab(self, mock_radio) -> None:
        """Test that tab_bar returns the value of the active tab."""
        # Setup
        mock_radio.return_value = "Tab2"
        values = ["Tab1", "Tab2", "Tab3"]

        # Execute
        result = tab_bar(values)

        # Assert
        assert result == "Tab2"
        mock_radio.assert_called_once_with("tab_bar_label", values, label_visibility="collapsed")

    @patch("streamlit.radio")
    def test_tab_bar_passes_kwargs(self, mock_radio) -> None:
        """Test that tab_bar passes kwargs to st.radio."""

        # Setup
        mock_radio.return_value = "Tab1"
        values = ["Tab1", "Tab2"]
        kwargs = {"horizontal": True, "key": "test_tabs"}

        # Execute
        tab_bar(values, **kwargs)

        # Assert
        mock_radio.assert_called_once_with(
            "tab_bar_label", values, label_visibility="collapsed", horizontal=True, key="test_tabs"
        )

    @patch("streamlit.radio")
    @patch("streamlit.get_option", return_value="#FF4B4B")
    @patch("streamlit.html")
    def test_tab_bar_html_generation(self, mock_html, mock_get_option, mock_radio) -> None:
        """Test that tab_bar generates HTML with the correct active tab index."""

        # Setup
        mock_radio.return_value = "Tab2"  # Second tab selected
        values = ["Tab1", "Tab2", "Tab3"]

        # Execute
        tab_bar(values)

        # Assert
        mock_get_option.assert_called_once_with("theme.primaryColor")
        # Check that the HTML contains the correct child number (2 for the second tab)
        html_arg = mock_html.call_args[0][0]
        assert "label:nth-child(2)" in html_arg
        assert "#FF4B4B" in html_arg  # Check primary color is included


class TestDisplayData:
    """Tests for the display_data function."""

    @pytest.fixture
    def signal_data(self) -> list[SignalData]:
        """Create mock SignalData objects."""

        x = Dimension(np.array([1, 2, 3, 4, 5]), "time", "s")
        y1 = Dimension(np.array([10, 20, 30, 40, 50]))
        y2 = Dimension(np.array([5, 15, 25, 35, 45]))

        signal1 = SignalData(x, y1, "Signal 1")
        signal2 = SignalData(x, y2, "Signal 2")

        return [signal1, signal2]

    @patch("streamlit.tabs")
    def test_display_data_creates_tabs(self, mock_tabs, signal_data) -> None:
        """Test that display_data creates the Graph and Data tabs."""

        # Setup
        mock_tabs.return_value = [MagicMock(), MagicMock()]

        # Execute
        display_data(go.Figure(), signal_data, key=1, filename=True)

        # Assert
        mock_tabs.assert_called_once_with(["Graph", "Data"])

    @patch("streamlit.tabs")
    @patch("utils.miscellaneous.make_unique")
    @patch("pandas.DataFrame")
    def test_display_data_shows_figure(self, mock_dataframe, mock_make_unique, mock_tabs, signal_data) -> None:
        """Test that display_data displays the figure in the first tab."""

        # Setup
        mock_tab1 = MagicMock()
        mock_tab2 = MagicMock()
        mock_tabs.return_value = [mock_tab1, mock_tab2]
        mock_make_unique.return_value = ["Time (s)", "Signal 1", "Signal 2"]
        mock_dataframe.return_value.style.format.return_value = "formatted_df"

        figure = go.Figure()

        # Execute
        display_data(figure, signal_data, key=1, filename=True)

        # Assert
        mock_tab1.plotly_chart.assert_called_once_with(figure, use_container_width=True, key="figure_1")

    @patch("streamlit.tabs")
    @patch("utils.miscellaneous.make_unique")
    @patch("pandas.DataFrame")
    def test_display_data_shows_dataframe(self, mock_dataframe, mock_make_unique, mock_tabs, signal_data) -> None:
        """Test that display_data displays a dataframe in the second tab."""

        # Setup
        mock_tab1 = MagicMock()
        mock_tab2 = MagicMock()
        mock_tabs.return_value = [mock_tab1, mock_tab2]
        mock_make_unique.return_value = ["Time (s)", "Signal 1", "Signal 2"]
        mock_df_style = MagicMock()
        mock_df_style.format.return_value = "formatted_df"
        mock_dataframe.return_value.style = mock_df_style

        figure = go.Figure()

        # Execute
        display_data(figure, signal_data, key=1, filename=True)

        # Assert
        mock_tab2.dataframe.assert_called_once_with("formatted_df", use_container_width=True, hide_index=True)

    @patch("streamlit.tabs")
    @patch("utils.miscellaneous.make_unique")
    @patch("pandas.DataFrame")
    def test_display_data_creates_correct_dataframe(
        self, mock_dataframe, mock_make_unique, mock_tabs, signal_data
    ) -> None:
        """Test that display_data creates a dataframe with the correct data."""

        # Setup
        mock_tab1 = MagicMock()
        mock_tab2 = MagicMock()
        mock_tabs.return_value = [mock_tab1, mock_tab2]
        mock_make_unique.return_value = ["Time (s)", "Signal 1", "Signal 2"]
        mock_df_style = MagicMock()
        mock_df_style.format.return_value = "formatted_df"
        mock_dataframe.return_value.style = mock_df_style

        # Execute
        display_data(go.Figure(), signal_data, key=1, filename=True)

        # Assert
        # Check that DataFrame was created with correct data
        x_data = signal_data[0].x.data
        y1_data = signal_data[0].y.data
        y2_data = signal_data[1].y.data
        expected_data = {"Time (s)": x_data, "Signal 1": y1_data, "Signal 2": y2_data}
        mock_dataframe.assert_called_once()
        # Check data passed to DataFrame constructor matches expected
        actual_data = mock_dataframe.call_args[0][0]
        assert list(actual_data.keys()) == list(expected_data.keys())
