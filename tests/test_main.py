"""Test module for the functions in the `main.py` module.

This module contains unit tests for the functions implemented in the `main.py` module. The purpose of these tests is to
ensure the correct functionality of each function in different scenarios and to validate that the expected outputs are
returned.

Tests should cover various edge cases, valid inputs, and any other conditions that are necessary to confirm the
robustness of the functions."""

import os.path
from io import BytesIO
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

import resources


class TestApp:
    main_path = "app/main.py"

    # def teardown_method(self) -> None:
    #     """Teardown method that runs after each test."""
    #
    #     # Make sure that no exception happened
    #     assert len(self.at.exception) == 0

    def test_default(self) -> None:

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()
        assert len(self.at.exception) == 0
        assert self.at.expander[-2].label == "Changelog"
        assert self.at.expander[-1].label == "License & Disclaimer"

    def get_widget_by_key(
        self,
        widget: str,
        key: str,
        verbose: bool = False,
    ) -> any:
        """Get a widget given its key
        :param widget: widget type
        :param key: key
        :param verbose: if True, print the index of the widget"""

        keys = [wid.key for wid in getattr(self.at, widget)]
        if verbose:
            print(keys)  # pragma: no cover
        index = keys.index(key)
        if verbose:
            print(index)  # pragma: no cover
        return getattr(self.at, widget)[index]

    def get_type_select(self) -> any:
        """Get the type select selectbox"""

        return self.get_widget_by_key("selectbox", "type_select")

    def get_data_select(self) -> any:
        """Get the data select selectbox"""

        return self.get_widget_by_key("selectbox", "data_select")

    def get_range_select(self, n: int) -> any:
        """Get the data range text input
        :param n: integer"""

        return self.get_widget_by_key("text_input", f"range_input{n}")

    def toggle_max_point(self, value: bool) -> None:
        """Toggle the display max checkbox"""

        self.get_widget_by_key("checkbox", "max_button").set_value(value).run()

    def toggle_min_point(self, value: bool) -> None:
        """Toggle the display min checkbox"""

        self.get_widget_by_key("checkbox", "min_button").set_value(value).run()

    def toggle_fwhm(self, value: bool) -> None:
        """Toggle the display fwhm checkbox"""

        self.get_widget_by_key("checkbox", "fwhm_button").set_value(value).run()

    def toggle_fwhm_interp(self, value: bool) -> None:
        """Toggle the display fwhm checkbox"""

        self.get_widget_by_key("checkbox", "fwhm_button_interp").set_value(value).run()

    @staticmethod
    def create_mock_file(
        mock_file_uploader: MagicMock,
        path: str,
    ) -> None:
        """Create a temporary CSV file with uneven columns and mock file upload.
        :param mock_file_uploader: MagicMock
        :param path: data to be uploaded"""

        # Load the data to the mock file
        with open(path, "rb") as f:
            file = BytesIO(f.read())
            file.name = os.path.basename(path)
            mock_file_uploader.return_value = file

    @patch("streamlit.sidebar.file_uploader")
    def test_upload(self, mock_file_uploader: MagicMock) -> None:

        for path in resources.FILE_TYPE_DICT:

            self.create_mock_file(mock_file_uploader, path)

            # Start the app and run it
            self.at = AppTest(self.main_path, default_timeout=100).run()

            # Assert filetype
            assert self.at.sidebar.markdown[1].value == f"Detected: {resources.FILE_TYPE_DICT[path]}"

    @patch("streamlit.sidebar.file_uploader")
    def test_fixed_read(self, mock_file_uploader: MagicMock) -> None:

        self.create_mock_file(mock_file_uploader, resources.SPECTRASUITE_HEADER_PATH)

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()

        self.get_widget_by_key("selectbox", "filetype_select").set_value("SpectraSuite (.txt)").run()

    @patch("streamlit.sidebar.file_uploader")
    def test_fixed_read_incorrect(self, mock_file_uploader: MagicMock) -> None:

        self.create_mock_file(mock_file_uploader, resources.SPECTRASUITE_HEADER_PATH)

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()

        self.get_widget_by_key("selectbox", "filetype_select").set_value("EasyLog (.txt)").run()

        assert self.at.warning[0].value == "Unable to read that file"

    @patch("streamlit.sidebar.file_uploader")
    def test_range_incorrect(self, mock_file_uploader: MagicMock):

        self.create_mock_file(mock_file_uploader, resources.SPECTRASUITE_HEADER_PATH)

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()

        self.get_range_select(1).set_value("f").run()

    # ---------------------------------------------- DIFFERENT DATA TYPES ----------------------------------------------

    @patch("streamlit.sidebar.file_uploader")
    def test_dict_file(self, mock_file_uploader: MagicMock):

        self.create_mock_file(mock_file_uploader, resources.EASYLOG_PATH)

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()

        assert self.get_type_select().options == ["All", "Temperature", "Humidity"]
        self.get_type_select().select("Humidity").run()

        assert self.get_range_select(1).value == "0.0"
        assert self.get_range_select(2).value == "253080.0"

    @patch("streamlit.sidebar.file_uploader")
    def test_list_file(self, mock_file_uploader: MagicMock):

        self.create_mock_file(mock_file_uploader, resources.FLUORESSENCE_MULTIPLE_PATH)

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()

        assert len(self.get_data_select().options) == 11
        self.get_data_select().select("05/12/2017 17:24:47").run()

        assert self.get_range_select(1).value == "650.0"
        assert self.get_range_select(2).value == "850.0"

    @patch("streamlit.sidebar.file_uploader")
    def test_dict_list_file(self, mock_file_uploader: MagicMock):

        self.create_mock_file(mock_file_uploader, resources.PRODATA_TAS_3PROP_PATH)  # wrong detect

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()

        # Type select
        assert len(self.get_type_select().options) == 4
        self.get_type_select().select("dT/T").run()

        # Data select
        assert len(self.get_data_select().options) == 4
        self.get_data_select().select("Repeats: 1").run()

        assert self.get_range_select(1).value == "-9.89e-07"
        assert self.get_range_select(2).value == "9.93e-07"

    # --------------------------------------------- EXTRACTING INFORMATION ---------------------------------------------

    @patch("streamlit.sidebar.file_uploader")
    def test_extract(self, mock_file_uploader: MagicMock):

        self.create_mock_file(mock_file_uploader, resources.SPECTRASUITE_HEADER_PATH)

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()

        # Toggle the min/max and fwhm buttons
        self.toggle_max_point(True)
        self.toggle_min_point(True)
        self.toggle_fwhm(True)
        assert len(self.at.markdown) == 14
        assert self.at.markdown[3].value == "Max. wavelength: 763.01 nm"
        assert self.at.markdown[4].value == "Max. intensity: 3363.64 counts"
        assert self.at.markdown[6].value == "Min. wavelength: 343.1 nm"
        assert self.at.markdown[7].value == "Min. intensity: 653.66 counts"
        assert self.at.markdown[9].value == "FWHM: 41.38 nm"

        # Try interpolation
        self.toggle_fwhm_interp(True)
        assert self.at.markdown[9].value == "FWHM: 41.1896 nm"

    @patch("streamlit.sidebar.file_uploader")
    def test_range_extract(self, mock_file_uploader: MagicMock):

        self.create_mock_file(mock_file_uploader, resources.SPECTRASUITE_HEADER_PATH)

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()

        # Change the range
        self.get_range_select(1).set_value("600").run()
        self.get_range_select(2).set_value("900").run()

        # Toggle the min/max and fwhm buttons
        self.toggle_max_point(True)
        self.toggle_min_point(True)
        self.toggle_fwhm(True)
        assert len(self.at.markdown) == 14
        assert self.at.markdown[3].value == "Max. wavelength: 763.01 nm"
        assert self.at.markdown[4].value == "Max. intensity: 3363.64 counts"
        assert self.at.markdown[6].value == "Min. wavelength: 856.63 nm"
        assert self.at.markdown[7].value == "Min. intensity: 805.85 counts"
        assert self.at.markdown[9].value == "FWHM: 40.04 nm"

        # Try interpolation
        self.toggle_fwhm_interp(True)
        assert self.at.markdown[9].value == "FWHM: 40.0205 nm"

    @patch("streamlit.sidebar.file_uploader")
    def test_smoothing_extract(self, mock_file_uploader: MagicMock):

        self.create_mock_file(mock_file_uploader, resources.SPECTRASUITE_HEADER_PATH)

        # Start the app and run it
        self.at = AppTest(self.main_path, default_timeout=100).run()

        # Set smoothing
        self.get_widget_by_key("number_input", "sg_fw").set_value(101).run()
        self.get_widget_by_key("number_input", "sg_po").set_value(4).run()

        # Toggle the min/max and fwhm buttons
        self.toggle_max_point(True)
        self.toggle_min_point(True)
        self.toggle_fwhm(True)
        assert len(self.at.markdown) == 14
        assert self.at.markdown[3].value == "Max. wavelength: 764 nm"
        assert self.at.markdown[4].value == "Max. intensity: 3337.52 counts"
        assert self.at.markdown[6].value == "Min. wavelength: 343.1 nm"
        assert self.at.markdown[7].value == "Min. intensity: 695.209 counts"
        assert self.at.markdown[9].value == "FWHM: 41.38 nm"

        # Try interpolation
        self.toggle_fwhm_interp(True)
        assert self.at.markdown[9].value == "FWHM: 41.1594 nm"
