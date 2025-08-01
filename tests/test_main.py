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

import main
from config import resources


class TestApp:
    main_path = "app/main.py"

    AVERAGING_INDEX = 0
    BCKG_EXP_INDEX = 0
    RANGE_EXP_INDEX = 1
    INTERP_EXP_INDEX = 2
    DERIVATIVE_EXP_INDEX = 3
    SMOOTHING_EXP_INDEX = 4
    NORM_EXP_INDEX = 5
    FITTING_EXP_INDEX = 6

    def setup_method(self) -> None:

        self.at = AppTest(self.main_path, default_timeout=100).run()

    def teardown_method(self) -> None:
        """Make sure that no exception happened"""

        assert len(self.at.exception) == 0

    def test_default(self) -> None:

        # Start the app and run it
        assert len(self.at.exception) == 0
        assert self.at.expander[-2].label == "Changelog"
        assert self.at.expander[-1].label == "License & Disclaimer"

    def _get_widget_by_key(
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

    # ------------------------------------------------- DATA SELECTION -------------------------------------------------

    def set_filetype_select(self, value: str) -> None:
        """Set the file type select selectbox"""

        self._get_widget_by_key("selectbox", "filetype_select_input").set_value(value).run()

    def get_type_select(self) -> any:
        """Get the type select selectbox"""

        return self._get_widget_by_key("selectbox", "type_select")

    def get_data_select(self) -> any:
        """Get the data select selectbox"""

        return self._get_widget_by_key("selectbox", "data_select")

    def set_data_sorting(self, value: str) -> None:
        """Set the data sorting key"""

        self._get_widget_by_key("selectbox", "sorting_key").set_value(value).run()

    def set_timestamp_shift(self, value: bool) -> None:
        """Set the status of the time stamp shift checkbox"""

        self._get_widget_by_key("toggle", "timestamp_shift").set_value(value).run()

    # ------------------------------------------------- DATA PROCESSING ------------------------------------------------

    def set_averaging(self, value: int) -> None:
        """Set the averaging number"""

        self._get_widget_by_key("number_input", "averaging_n").set_value(value).run()

    def set_background(self, lower: str = "", upper: str = "") -> None:
        """Set the background range"""

        self._get_widget_by_key("text_input", "bckg_range_input1").set_value(lower).run()
        self._get_widget_by_key("text_input", "bckg_range_input2").set_value(upper).run()

    def set_range(self, lower: str = "", upper: str = "") -> None:
        """Set the data range"""

        self._get_widget_by_key("text_input", "range_input1").set_value(lower).run()
        self._get_widget_by_key("text_input", "range_input2").set_value(upper).run()

    def set_interpolation(self, interp_type: str, value: str) -> any:
        """Set the interpolation type and vaue"""

        self._get_widget_by_key("radio", "interp_type").set_value(interp_type).run()
        if interp_type == "Fixed Step":
            self._get_widget_by_key("text_input", "interp_dx").set_value(value).run()
        elif interp_type == "Point Count":
            self._get_widget_by_key("text_input", "interp_n").set_value(value).run()

    def set_derivative(self, order: int) -> any:
        """Set the derivative order"""

        self._get_widget_by_key("number_input", "derive_order").set_value(order).run()

    def set_smoothing(self, window, order) -> None:
        """Set the smoothing window and order"""

        self._get_widget_by_key("number_input", "sg_fw").set_value(window).run()
        self._get_widget_by_key("number_input", "sg_po").set_value(order).run()

    def set_normalisation(self, norm_type: str, a: str = "1", b: str = "0") -> None:
        """Set the normalisation"""

        self._get_widget_by_key("radio", "norm_type").set_value(norm_type).run()
        if norm_type == "Feature Scaling":
            self._get_widget_by_key("text_input", "norm_a").set_value(a).run()
            self._get_widget_by_key("text_input", "norm_a").set_value(b).run()

    def set_fitting(self, model: str, **kwargs) -> None:
        """Get the data range text input"""

        self._get_widget_by_key("selectbox", "fitting_model").set_value(model).run()
        for key in kwargs:
            self._get_widget_by_key("selectbox", "parameter_model_key").set_value(key).run()
            ss_key = model + "0" + key + "guess_value"
            self._get_widget_by_key("text_input", ss_key, True)
            self._get_widget_by_key("text_input", ss_key).set_value(kwargs[key]).run()

    # ------------------------------------------------- DATA EXTRACTION ------------------------------------------------

    def toggle_max_point(self, value: bool) -> None:
        """Toggle the display max checkbox"""

        self._get_widget_by_key("checkbox", "max_button").set_value(value).run()

    def toggle_max_interp(self, value: bool) -> None:
        """Toggle the display max interpolation checkbox"""

        self._get_widget_by_key("checkbox", "max_interp_button").set_value(value).run()

    def toggle_min_interp(self, value: bool) -> None:
        """Toggle the display min interpolation checkbox"""

        self._get_widget_by_key("checkbox", "min_interp_button").set_value(value).run()

    def toggle_min_point(self, value: bool) -> None:
        """Toggle the display min checkbox"""

        self._get_widget_by_key("checkbox", "min_button").set_value(value).run()

    def toggle_fwhm(self, value: bool) -> None:
        """Toggle the display fwhm checkbox"""

        self._get_widget_by_key("checkbox", "fwhm_button").set_value(value).run()

    def toggle_fwhm_interp(self, value: bool) -> None:
        """Toggle the display fwhm checkbox"""

        self._get_widget_by_key("checkbox", "fwhm_button_interp").set_value(value).run()

    def toggle_area(self, value: bool) -> None:
        """Toggle the display fwhm checkbox"""

        self._get_widget_by_key("checkbox", "area_button").set_value(value).run()

    def create_mock_file(
        self,
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
        self.at.run()

    # ------------------------------------------------ FILE UPLOAD/READ ------------------------------------------------

    @patch("streamlit.sidebar.file_uploader")
    def test_upload(self, mock_file_uploader: MagicMock) -> None:

        for path in resources.FILE_TYPE_DICT:

            self.create_mock_file(mock_file_uploader, path)

            # Start the app and run it (necessary to reset the file uploader)
            self.at = AppTest(self.main_path, default_timeout=100).run()

            # Assert filetype
            assert self.at.sidebar.markdown[1].value == f"Detected: {resources.FILE_TYPE_DICT[path]}"

    @patch("streamlit.sidebar.file_uploader")
    def test_fixed_read(self, mock_file_uploader: MagicMock) -> None:

        # Load test data and start the app
        path = resources.SPECTRASUITE_HEADER_PATH
        self.create_mock_file(mock_file_uploader, path)

        self.set_filetype_select(resources.FILE_TYPE_DICT[path])
        assert len(self.at.warning) == 0

    @patch("streamlit.sidebar.file_uploader")
    def test_fixed_read_incorrect(self, mock_file_uploader: MagicMock) -> None:

        # Load test data and start the app
        self.create_mock_file(mock_file_uploader, resources.SPECTRASUITE_HEADER_PATH)

        self.set_filetype_select("EasyLog (.txt)")

        assert self.at.warning[0].value == "Unable to read the file."

    @patch("streamlit.sidebar.file_uploader")
    def test_data_sorting(self, mock_file_uploader: MagicMock) -> None:
        """Test the data sorting using the datetime"""

        self.create_mock_file(mock_file_uploader, resources.DIFFRAC_TIMELAPSE_PATH)
        self.set_data_sorting("Date & time")
        self.set_averaging(3)
        assert self.at.sidebar.expander[self.AVERAGING_INDEX].label == f"__✔ {main.AVERAGING_LABEL} (Every 3 Signals)__"
        self.set_timestamp_shift(False)
        assert self.at.sidebar.expander[self.AVERAGING_INDEX].label == main.AVERAGING_LABEL

    # ------------------------------------------------- DATA PROCESSING ------------------------------------------------

    @patch("streamlit.sidebar.file_uploader")
    def test_averaging(self, mock_file_uploader: MagicMock) -> None:

        # Load test data and start the app
        self.create_mock_file(mock_file_uploader, resources.DIFFRAC_TIMELAPSE_PATH)

        assert self.at.sidebar.expander[self.AVERAGING_INDEX].label == main.AVERAGING_LABEL
        self.set_averaging(2)
        assert self.at.sidebar.expander[self.AVERAGING_INDEX].label == f"__✔ {main.AVERAGING_LABEL} (Every 2 Signals)__"
        self.set_averaging(1)
        assert self.at.sidebar.expander[self.AVERAGING_INDEX].label == main.AVERAGING_LABEL

    @patch("streamlit.sidebar.file_uploader")
    def test_background_range(self, mock_file_uploader: MagicMock) -> None:

        # Load test data and start the app
        self.create_mock_file(mock_file_uploader, resources.SPECTRASUITE_HEADER_PATH)

        assert self.at.sidebar.expander[self.BCKG_EXP_INDEX].label == main.BACKGROUND_LABEL
        self.set_background("400", "800")
        assert self.at.sidebar.expander[self.BCKG_EXP_INDEX].label == f"__✔ {main.BACKGROUND_LABEL} 400.0 - 800.0__"
        self.set_background("400", "")
        assert self.at.sidebar.expander[self.BCKG_EXP_INDEX].label == main.BACKGROUND_LABEL

    @patch("streamlit.sidebar.file_uploader")
    def test_data_range(self, mock_file_uploader: MagicMock) -> None:

        # Load test data and start the app
        self.create_mock_file(mock_file_uploader, resources.SPECTRASUITE_HEADER_PATH)

        assert self.at.sidebar.expander[self.RANGE_EXP_INDEX].label == main.RANGE_LABEL
        self.set_range("400", "800")
        assert self.at.sidebar.expander[self.RANGE_EXP_INDEX].label == f"__✔ {main.RANGE_LABEL} 400.0 - 800.0__"
        self.set_range("", "")
        assert self.at.sidebar.expander[self.RANGE_EXP_INDEX].label == main.RANGE_LABEL

    @patch("streamlit.sidebar.file_uploader")
    def test_interpolation(self, mock_file_uploader: MagicMock) -> None:

        # Load test data and start the app
        self.create_mock_file(mock_file_uploader, resources.SPECTRASUITE_HEADER_PATH)

        assert self.at.sidebar.expander[self.INTERP_EXP_INDEX].label == main.INTERP_LABEL
        self.set_interpolation("Fixed Step", "2")
        assert self.at.sidebar.expander[self.INTERP_EXP_INDEX].label == f"__✔ {main.INTERP_LABEL} (step = 2 nm)__"
        self.set_interpolation("Point Count", "1000")
        assert self.at.sidebar.expander[self.INTERP_EXP_INDEX].label == f"__✔ {main.INTERP_LABEL} (step = 0.69 nm)__"

    @patch("streamlit.sidebar.file_uploader")
    def test_derivative(self, mock_file_uploader: MagicMock) -> None:

        # Load test data and start the app
        self.create_mock_file(mock_file_uploader, resources.SPECTRASUITE_HEADER_PATH)

        assert self.at.sidebar.expander[self.DERIVATIVE_EXP_INDEX].label == main.DERIVE_LABEL
        self.set_derivative(1)
        assert self.at.sidebar.expander[self.DERIVATIVE_EXP_INDEX].label == f"__✔ {main.DERIVE_LABEL} (1 order)__"

    @patch("streamlit.sidebar.file_uploader")
    def test_smoothing(self, mock_file_uploader: MagicMock) -> None:

        # Load test data and start the app
        self.create_mock_file(mock_file_uploader, resources.SPECTRASUITE_HEADER_PATH)

        assert self.at.sidebar.expander[self.SMOOTHING_EXP_INDEX].label == main.SMOOTHING_LABEL
        self.set_smoothing(101, 4)
        assert self.at.sidebar.expander[self.SMOOTHING_EXP_INDEX].label == f"__✔ {main.SMOOTHING_LABEL} (101, 4)__"

        # Incorrect smoothing
        self.set_smoothing(10001, 4)

    @patch("streamlit.sidebar.file_uploader")
    def test_normalisation(self, mock_file_uploader: MagicMock) -> None:

        # Load test data and start the app
        self.create_mock_file(mock_file_uploader, resources.SPECTRASUITE_HEADER_PATH)

        # No normalisation
        assert self.at.sidebar.expander[self.NORM_EXP_INDEX].label == main.NORM_LABEL

        # Max normalisation
        self.set_normalisation("Max. Normalisation")
        assert self.at.sidebar.expander[self.NORM_EXP_INDEX].label == f"__✔ {main.NORM_LABEL} (Max. Normalisation)__"

        # Feature Scaling
        self.set_normalisation("Feature Scaling", "2", "1")
        assert (
            self.at.sidebar.expander[self.NORM_EXP_INDEX].label
            == f"__✔ {main.NORM_LABEL} (Feature Scaling 1.0 - 0.0)__"
        )

        # Normalisation with smoothing
        self.set_smoothing(101, 4)
        self.set_normalisation("Max. Normalisation")
        self.set_normalisation("Feature Scaling")

        # Failed normalisation
        self.set_normalisation("Feature Scaling", "f", "f")

    @patch("streamlit.sidebar.file_uploader")
    def test_fitting(self, mock_file_uploader: MagicMock) -> None:

        # Load test data and start the app
        self.create_mock_file(mock_file_uploader, resources.SPECTRASUITE_HEADER_PATH)

        assert self.at.sidebar.expander[self.FITTING_EXP_INDEX].label == main.FITTING_LABEL
        self.set_fitting("Gaussian", c="4")
        assert self.at.sidebar.expander[self.FITTING_EXP_INDEX].label == f"__✔ {main.FITTING_LABEL} (Gaussian)__"

        # Test failed guess value
        self._get_widget_by_key("text_input", "Gaussian0cguess_value").set_value("f").run()

    # ---------------------------------------------- DIFFERENT DATA TYPES ----------------------------------------------

    @patch("streamlit.sidebar.file_uploader")
    def test_dict_file(self, mock_file_uploader: MagicMock) -> None:

        self.create_mock_file(mock_file_uploader, resources.EASYLOG_PATH)
        assert sorted(self.get_type_select().options) == sorted(["All", "Temperature", "Humidity"])
        self.get_type_select().select("Humidity").run()
        assert len(self.at.warning) == 0

    @patch("streamlit.sidebar.file_uploader")
    def test_list_file(self, mock_file_uploader: MagicMock) -> None:

        self.create_mock_file(mock_file_uploader, resources.FLUORESSENCE_MULTIPLE_PATH)
        assert len(self.get_data_select().options) == 11
        self.get_data_select().select("2017-05-12 17:28:23").run()
        assert len(self.at.warning) == 0

    @patch("streamlit.sidebar.file_uploader")
    def test_dict_list_file(self, mock_file_uploader: MagicMock) -> None:

        self.create_mock_file(mock_file_uploader, resources.PRODATA_TAS_3PROP_PATH)

        # Type select
        assert len(self.get_type_select().options) == 4
        self.get_type_select().select("dT/T").run()

        # Data select
        assert len(self.get_data_select().options) == 4
        self.get_data_select().select("Repeats: 1").run()

        assert len(self.at.warning) == 0

    # --------------------------------------------- EXTRACTING INFORMATION ---------------------------------------------

    @patch("streamlit.sidebar.file_uploader")
    def test_extract(self, mock_file_uploader: MagicMock) -> None:

        self.create_mock_file(mock_file_uploader, resources.SPECTRASUITE_HEADER_PATH)

        # Toggle the min/max and fwhm buttons
        self.toggle_max_point(True)
        self.toggle_min_point(True)
        self.toggle_fwhm(True)
        self.toggle_area(True)

        # Try interpolation
        self.toggle_fwhm_interp(True)
        self.toggle_max_interp(True)
        self.toggle_min_interp(True)

    # -------------------------------------------------- OVERALL TEST --------------------------------------------------

    @patch("streamlit.sidebar.file_uploader")
    def test_typical_user(self, mock_file_uploader: MagicMock) -> None:

        self.create_mock_file(mock_file_uploader, resources.DIFFRAC_TIMELAPSE_PATH)
        self.set_data_sorting("Date & time")

        # Processing options
        self.set_averaging(2)
        self.set_background("13", "13.2")
        self.set_range("13.2")
        self.set_smoothing(101, 4)

        # Extraction
        self.toggle_max_point(True)
        self.toggle_max_interp(True)
        self.toggle_min_point(True)
        self.toggle_min_interp(True)
        self.toggle_fwhm(True)
        self.toggle_fwhm_interp(True)
        self.toggle_area(True)
