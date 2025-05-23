"""Graphical user interface of Raft"""

import os

import numpy as np
import pandas as pd
import streamlit as st
import argparse

from config.constants import TIMESTAMP_ID
from config.resources import LOGO_PATH, CSS_STYLE_PATH, ICON_PATH, DATA_PROCESSING_PATH, LOGO_TEXT_PATH, FILE_TYPE_DICT
from data_files.data_files import FILETYPES, read_data_file
from data_files.signal_data import Dimension, SignalData, average_signals, get_z_dim
from data_processing.fitting import MODELS, get_model_parameters
from interface.plot import plot_signals, scatter_plot, make_figure
from interface.streamlit import tab_bar, display_data
from utils.file_io import read_file, render_image
from utils.miscellaneous import normalise_list, make_unique
from utils.project import get_pyproject_info, get_last_commit_date_from_github
from utils.session_state import refresh_session_state_widgets, set_session_state_value_function, set_default_widgets
from utils.strings import matrix_to_string, number_to_str, generate_html_table, dedent

__version__ = get_pyproject_info("project", "version")
__name__ = get_pyproject_info("project", "name")
__description__ = get_pyproject_info("project", "description")
__github__ = get_pyproject_info("project", "urls", "repository")
__date__ = get_last_commit_date_from_github(__github__)
__author__ = get_pyproject_info("project", "authors")[0]["name"]

# ------------------------------------------------ ARGUMENTS & PROFILER ------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--profile", default=False, type=bool, help="Switch the profiler on")
args, _ = parser.parse_known_args()

# Profiler
if args.profile:
    from wfork_streamlit_profiler import Profiler

    profiler = Profiler()
    profiler.start()
else:
    profiler = None

# ------------------------------------------------------ CONSTANTS -----------------------------------------------------

# Project path
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Section labels
AVERAGING_LABEL = "Averaging"
BACKGROUND_LABEL = "Background Subtraction"
RANGE_LABEL = "Data Range"
INTERP_LABEL = "Interpolation"
DERIVE_LABEL = "Derivative"
SMOOTHING_LABEL = "Smoothing"
NORM_LABEL = "Normalisation"
FITTING_LABEL = "Fitting"
EXTRACTION_LABEL = "Data Extraction"

# -------------------------------------------------------- SETUP -------------------------------------------------------

# Set the app main and sidebar logos
st.set_page_config(__name__.upper() + " - " + __description__, page_icon=ICON_PATH, layout="wide")
st.logo(LOGO_TEXT_PATH, icon_image=LOGO_PATH)

# Load the custom CSS
css_content = read_file(CSS_STYLE_PATH)
st.html(f"<style>{css_content}</style>")

# ---------------------------------------------------- SESSION STATE ---------------------------------------------------

# Create a shortened filename for the session state
sss = st.session_state

# Dictionary storing the default values of session state values (including widgets and interaction status)
widgets_defaults = {
    "bckg_interacted": False,
    "bckg_range_input1": "",
    "bckg_range_input2": "",
    "range_interacted": False,
    "range_input1": "",
    "range_input2": "",
    "smoothing_interacted": False,
    "sg_fw": 0,
    "sg_po": 0,
    "display_raw": True,
    "fitting_interacted": False,
    "fitting_model": "None",
    "fitting_curve_select": "",
    "fitting_apply_all": False,
    "norm_interacted": False,
    "norm_type": "None",
    "norm_a": "1",
    "norm_b": "0",
    "interp_interacted": False,
    "interp_type": "None",
    "interp_dx": "",
    "interp_n": "",
    "derive_interacted": False,
    "derive_order": 0,
    "guess_values": None,
    "reset_settings": True,
    "precision_input": 4,
    "sorting_key": "Signal name",
    "timestamp_shift": True,
    "averaging_interacted": False,
    "averaging_n": 1,
}

set_default_widgets(widgets_defaults)

# Add data and the file_uploader key to the session state
if "data" not in sss:
    sss.data = None
if "error_txt" not in sss:
    sss.error_txt = ""
if "file_uploader" not in sss:
    sss.file_uploader = "file_uploader_key0"


def reset_data() -> None:
    """Reset the settings and remove the stored data"""

    close_expanders()  # close all expanders
    sss.data = None  # remove the stored data
    if sss.reset_settings:
        set_default_widgets(widgets_defaults, True)  # reset the default settings


def close_expanders() -> None:
    """Reset the interacted session state keys"""

    for key in widgets_defaults:
        if "interacted" in key:
            sss[key] = False


refresh_session_state_widgets(widgets_defaults)


# ---------------------------------------------------- FILE UPLOADER ---------------------------------------------------

st.sidebar.markdown("## Data Upload")
FILE_UPLOADER_MODES = ["Single File", "Multiple Files"]
UPLOADER_KWARGS = dict(
    label="File uploader",
    label_visibility="hidden",
    on_change=reset_data,
)
with st.sidebar:
    tab = tab_bar(FILE_UPLOADER_MODES)

# Single file mode
if tab == FILE_UPLOADER_MODES[0]:
    file = st.sidebar.file_uploader(
        **UPLOADER_KWARGS,
    )
else:
    file = st.sidebar.file_uploader(
        accept_multiple_files=True,
        key=sss.file_uploader,
        **UPLOADER_KWARGS,
    )

    def set_new_uploader_key() -> None:
        """Change the key of the file uploader"""

        string = "file_uploader_key"
        i_old = int(sss.file_uploader.replace(string, ""))
        sss.file_uploader = string + str(i_old + 1)
        reset_data()

    # If files have been uploaded, show the button to remove them
    if file:
        st.sidebar.button(
            label="Clear",
            on_click=set_new_uploader_key,
            use_container_width=True,
        )

# ------------------------------------------------------ FILE TYPE -----------------------------------------------------

filetype_help = "Select the file type. If 'Detect' is selected, the file type will be automatically detected."
filetype_select = st.sidebar.selectbox(
    label="Data File Type",
    options=FILETYPES,
    on_change=reset_data,
    help=filetype_help,
    key="filetype_select_input",
)

# Detect message
detect_placeholder = st.sidebar.empty()

# -------------------------------------------------------- LOGO --------------------------------------------------------

# If no file is provided or no signal is stored
if not file:  # covers None and empty list
    st.html(render_image(LOGO_PATH, 200))  # main logo
    title_string = """<div style="text-align: center; font-family: sans-serif; font-size: 45px; line-height: 1.3;
    color: rgb(108, 65, 39)">RAFT</div>
    <div style="text-align: center; font-family: sans-serif; font-size: 45px; line-height: 1.3;color: rgb(108, 65, 39)">
    Unive<strong>r</strong>sal D<strong>a</strong>ta <strong>F</strong>ile Plo<strong>t</strong>ter</div>"""
    st.html(title_string)


# If file is an uploaded file or a list of uploaded file
else:

    # -------------------------------------------------- DATA LOADING --------------------------------------------------

    # If not data have been stored previously, try loading the files and store them
    if sss.data is None:
        sss.error_txt = ""
        print("Reading data files")

        # Check that the filenames are unique
        if isinstance(file, list) and len(set([f.name.split(".")[0] for f in file])) != len(file):
            signals, filetype = [], None
            sss.error_txt = "Multiple files with the same filename cannot be loaded."

        else:
            signals, filetype = read_data_file(file, filetype_select)

            # If signals have been loaded...
            if signals:

                # Normalise them
                signals = normalise_list(signals)

                # Check data consistency
                if isinstance(signals, list):
                    for signal in signals[1:]:
                        if not np.array_equal(signals[0].x.data, signal.x.data, equal_nan=True):
                            signals, filetype = [], None
                            sss.error_txt = "Data not sharing the same X-axis data cannot be loaded as of now."
                            break

            else:
                if isinstance(file, list):
                    sss.error_txt = "Unable to read the files."
                else:
                    sss.error_txt = "Unable to read the file."

            # Store them in the session state
            sss.data = [signals, filetype]

    # Otherwise load the data from the session state
    else:
        signals, filetype = sss.data

    if filetype_select == FILETYPES[0] and filetype:
        detect_placeholder.markdown(f"Detected: {filetype}")

    # Condition determining if multiple files have been loaded
    multifile_cond = isinstance(file, list) and len(file) > 1

    # If the signal could not be loaded, display a warning message
    if filetype is None:
        st.markdown("###")  # some space on top for the st.logo
        st.warning(sss.error_txt)

    # Show the options for selecting a specific signal
    else:

        # ----------------------------------------------- DATA SELECTION -----------------------------------------------

        # If the signals are stored in a dictionary...
        if isinstance(signals, dict):

            # If the dictionary has only 1 key, use that key
            if len(signals) == 1:
                signals = list(signals.values())[0]

            # Otherwise, allow to select the key to plot if signal is a dictionary
            else:
                selection = st.sidebar.selectbox(
                    label="Data Type",
                    options=["All"] + list(signals.keys()),
                    on_change=close_expanders,
                    key="type_select",
                )
                if selection in signals:
                    signals = signals[selection]

        # Select the signal in a list if multiple signals
        if isinstance(signals, list) and len(signals) > 1:
            names = [signal.get_name(multifile_cond) for signal in signals]
            signal_dict = dict(zip(names, signals))
            col_selection = st.sidebar.selectbox(
                label="Data Selection",
                options=["All"] + sorted(signal_dict.keys()),
                on_change=close_expanders,
                key="data_select",
            )
            if col_selection in signal_dict:
                signals = [signal_dict[col_selection]]

        # ----------------------------------------------- SIGNAL SORTING -----------------------------------------------

        if isinstance(signals, list):

            if len(signals) > 1:
                # List of all z_dict keys
                z_keys = list(set([key for signal in signals for key in signal.z_dict]))

                if sss.sorting_key == TIMESTAMP_ID:
                    col1, col2 = st.sidebar.columns(2, vertical_alignment="bottom")
                else:
                    col1, col2 = st.sidebar, None

                # Select the key used to sort the data
                help_str = """Select the quantity by which to sort the signals. For example:
                * Signal name – sorts the signals alphabetically by their name.
                * Timestamp – sorts the signals according to the timestamps extracted from the raw data.
                * Emission Wavelength - sort the signals according to the emission wavelength extracted from the raw data.
                * ..."""
                col1.selectbox(
                    label="Data Sorting",
                    options=["None", "Signal name"] + sorted(z_keys, key=lambda string: string.lower()),
                    help=dedent(help_str),
                    format_func=lambda string: string.capitalize(),
                    key="sorting_key",
                )

                if col2:
                    col2.toggle(
                        label="Use Elapsed Time",
                        key="timestamp_shift",
                    )

                if sss.sorting_key != "None":
                    signals, z_dims = get_z_dim(signals, sss.sorting_key, sss.timestamp_shift)
                else:
                    z_dims = [Dimension(i) for i in range(len(signals))]

            else:
                signals, z_dims = get_z_dim(signals, "Signal name", sss.timestamp_shift)
        else:
            z_dims = []

        # ----------------------------------------------------------------------------------------------------------
        # ----------------------------------------------- DATA PROCESSING ----------------------------------------------
        # ----------------------------------------------------------------------------------------------------------

        if isinstance(signals, list):

            st.sidebar.markdown("## Data Processing")

            # ------------------------------------------------ AVERAGING -----------------------------------------------

            if len(signals) > 1:
                expander_label = AVERAGING_LABEL
                try:
                    if sss.averaging_n > 1:
                        signals, z_dims = average_signals(signals, z_dims, sss.averaging_n)
                        expander_label = f"__✔ {AVERAGING_LABEL} (Every {sss.averaging_n} Signals)__"
                except Exception as e:
                    print("Averaging failed")
                    print(e)
                    expander_label = AVERAGING_LABEL

                with st.sidebar.expander(expander_label, sss["averaging_interacted"]):

                    st.number_input(
                        label="Average Every X Signals",
                        min_value=1,
                        key="averaging_n",
                        on_change=set_session_state_value_function("averaging_interacted", True),
                    )

            # ----------------------------------------------- BACKGROUND -----------------------------------------------

            try:
                xmin = float(sss.bckg_range_input1)
                xmax = float(sss.bckg_range_input2)
                signals = [signal.remove_background(xmin, xmax) for signal in signals]
                expander_label = f"__✔ {BACKGROUND_LABEL} {xmin} - {xmax}__"
            except:
                expander_label = BACKGROUND_LABEL

            with st.sidebar.expander(expander_label, sss["bckg_interacted"]):

                columns = st.columns(2)

                label1 = signals[0].x(quantity="Lower Range").get_label_raw()
                help_str = f"Lower range of {signals[0].x.quantity} used to calculate the average background signal."
                columns[0].text_input(
                    label=label1,
                    key="bckg_range_input1",
                    help=help_str,
                    on_change=set_session_state_value_function("bckg_interacted", True),
                )

                label2 = signals[0].x(quantity="Upper Range").get_label_raw()
                help_str = f"Upper range of {signals[0].x.quantity} used to calculate the average background signal."
                columns[1].text_input(
                    label=label2,
                    key="bckg_range_input2",
                    help=help_str,
                    on_change=set_session_state_value_function("bckg_interacted", True),
                )

            # ----------------------------------------------- DATA RANGE -----------------------------------------------

            expander_label = RANGE_LABEL

            try:
                try:
                    xmin = float(sss.range_input1)
                except:
                    xmin = "Min"
                try:
                    xmax = float(sss.range_input2)
                except:
                    xmax = "Max"

                if isinstance(xmin, float) or isinstance(xmax, float):
                    signals = [signal.reduce_range(xmin, xmax) for signal in signals]
                    expander_label = f"__✔ {RANGE_LABEL} {xmin} - {xmax}__"
            except Exception as e:
                print("Could not reduce the range")
                print(e)
                expander_label = RANGE_LABEL

            with st.sidebar.expander(expander_label, sss["range_interacted"]):

                columns = st.columns(2)

                label1 = signals[0].x(quantity="Lower Range").get_label_raw()
                help_str = f"Lower range of {signals[0].x.quantity} to display."
                columns[0].text_input(
                    label=label1,
                    key="range_input1",
                    help=help_str,
                    on_change=set_session_state_value_function("range_interacted", True),
                )

                label2 = signals[0].x(quantity="Upper Range").get_label_raw()
                help_str = f"Upper range of {signals[0].x.quantity} to display."
                columns[1].text_input(
                    label=label2,
                    key="range_input2",
                    help=help_str,
                    on_change=set_session_state_value_function("range_interacted", True),
                )

            # ---------------------------------------------- INTERPOLATION ---------------------------------------------

            expander_label = INTERP_LABEL
            INTERP_OPTIONS = ["None", "Fixed Step", "Point Count"]
            try:
                dx = None

                # Fixed step interpolation
                if sss.interp_type == "Fixed Step":
                    dx = float(sss.interp_dx)
                    signals = [signal.interpolate(dx=dx) for signal in signals]

                # Point count interpolation
                elif sss.interp_type == "Point Count":
                    n_count = int(sss.interp_n)
                    signals = [signal.interpolate(dx=n_count) for signal in signals]
                    dx = np.mean(np.diff(signals[0].x.data))

                # Label
                if dx:
                    dx_str = number_to_str(dx, 3, "g", False, True)
                    expander_label = f"__✔ {INTERP_LABEL} (step = {dx_str} {signals[0].x.unit})__"
            except Exception as e:
                print("Interpolation failed")
                print(e)
                expander_label = INTERP_LABEL

            with st.sidebar.expander(expander_label, sss["interp_interacted"]):

                help_str = """Interpolate the data using different methods:
                * __Fixed Step__ interpolation – Data are interpolated using a specified step size.
                * __Point Count__ interpolation – Data are interpolated to fit a specified number of points."""

                st.radio(
                    label="Interpolation Type",
                    options=INTERP_OPTIONS,
                    key="interp_type",
                    on_change=set_session_state_value_function("interp_interacted", True),
                    help=dedent(help_str),
                    horizontal=True,
                )

                if sss["interp_type"] != "None":
                    st.text_input(
                        label="f",
                        key={"Fixed Step": "interp_dx", "Point Count": "interp_n"}[sss.interp_type],
                        label_visibility="collapsed",
                        on_change=set_session_state_value_function("interp_interacted", True),
                    )

            # ----------------------------------------------- DERIVATION -----------------------------------------------

            expander_label = DERIVE_LABEL
            try:
                if sss.derive_order > 0:
                    signals = [signal.derive(n=sss.derive_order) for signal in signals]
                    expander_label = f"__✔ {DERIVE_LABEL} ({sss.derive_order} order)__"
            except:
                expander_label = DERIVE_LABEL

            with st.sidebar.expander(expander_label, sss["derive_interacted"]):

                st.number_input(
                    label="Derivative Order",
                    min_value=0,
                    key="derive_order",
                    help="Calculate the n-th order derivative.",
                    on_change=set_session_state_value_function("derive_interacted", True),
                )

            # ------------------------------------------------ SMOOTHING -----------------------------------------------

            s_signals = None
            expander_label = SMOOTHING_LABEL
            try:
                if sss.sg_fw > 0 and sss.sg_po > 0:
                    s_signals = [signal.smooth(sss.sg_fw, sss.sg_po) for signal in signals]
                    expander_label = f"__✔ {SMOOTHING_LABEL} ({sss.sg_fw}, {sss.sg_po})__"
            except:
                s_signals = None
                expander_label = SMOOTHING_LABEL

            with st.sidebar.expander(expander_label, sss["smoothing_interacted"]):

                smoothing_cols = st.columns(2)

                smoothing_cols[0].number_input(
                    label="Filter Length",
                    min_value=0,
                    help="Length of the Savitzky-Golay filter window",
                    key="sg_fw",
                    on_change=set_session_state_value_function("smoothing_interacted", True),
                )

                smoothing_cols[1].number_input(
                    label="Polynomial Order",
                    min_value=0,
                    help="Order of the polynomial used to fit the samples",
                    key="sg_po",
                    on_change=set_session_state_value_function("smoothing_interacted", True),
                )

                st.toggle(
                    label="Display Raw Data",
                    key="display_raw",
                    help="If toggled, display the raw data as well as the smoothed data.",
                    on_change=set_session_state_value_function("smoothing_interacted", True),
                )

            # ---------------------------------------------- NORMALISATION ---------------------------------------------

            NORM_OPTIONS = ["None", "Max. Normalisation", "Feature Scaling"]
            expander_label = NORM_LABEL
            try:
                # Max normalisation
                if sss["norm_type"] == "Max. Normalisation":
                    if s_signals:
                        signals = [sig.normalise(other=s_sig.y.data) for sig, s_sig in zip(signals, s_signals)]
                        s_signals = [s_sig.normalise() for s_sig in s_signals]
                    else:
                        signals = [signal.normalise() for signal in signals]
                    expander_label = f"__✔ {NORM_LABEL} ({sss['norm_type']})__"

                # Feature scaling
                elif sss["norm_type"] == "Feature Scaling":
                    kwargs = dict(a=float(sss["norm_a"]), b=float(sss["norm_b"]))
                    if s_signals:
                        signals = [
                            sig.feature_scale(other=s_sig.y.data, **kwargs) for sig, s_sig in zip(signals, s_signals)
                        ]
                        s_signals = [s_sig.feature_scale(**kwargs) for s_sig in s_signals]
                    else:
                        signals = [signal.feature_scale(**kwargs) for signal in signals]

                    expander_label = f"__✔ {NORM_LABEL} ({sss['norm_type']} {kwargs['a']} - {kwargs['b']})__"
            except:
                expander_label = NORM_LABEL

            with st.sidebar.expander(expander_label, sss["norm_interacted"]):

                help_str = """Data can be normalised using different methods:
                * __Max. normalisation__ – the y-values are normalised with respect to their maximum value.
                * __Feature scaling__ – the y-values are normalised based on specified minimum and maximum values."""

                st.radio(
                    label="Normalisation Type",
                    options=NORM_OPTIONS,
                    key="norm_type",
                    on_change=set_session_state_value_function("norm_interacted", True),
                    help=dedent(help_str),
                    horizontal=True,
                )

                if sss["norm_type"] == "Feature Scaling":
                    columns = st.columns(2)

                    columns[0].text_input(
                        label="Maximum Value",
                        key="norm_a",
                        on_change=set_session_state_value_function("norm_interacted", True),
                    )

                    columns[1].text_input(
                        label="Minimum Value",
                        key="norm_b",
                        on_change=set_session_state_value_function("norm_interacted", True),
                    )

            # Store the raw signals and change the default signals to the smoothed data if they exists
            raw_signals = signals
            if s_signals:
                signals = s_signals

            # ------------------------------------------------- FITTING ------------------------------------------------

            expander_label = FITTING_LABEL
            fit_signals, fit_params, param_errors, r_squared = [], [], [], []
            equation, parameters = "", []
            rel_error = []

            # Get the fit function, equation and guess function, and the function parameters
            if sss.fitting_model in MODELS:

                fit_function, equation, guess_function = MODELS[sss.fitting_model]
                parameters = get_model_parameters(fit_function)

                # Reset the guess value default dict is set to None
                if sss.guess_values is None:
                    try:
                        guess_values = [guess_function(signal.x.data, signal.y.data) for signal in signals]
                        index = int(len(signals) / 2)
                        guess_values += [guess_function(signals[index].x.data, signals[index].y.data)]
                    except:
                        guess_values = [np.ones(len(parameters)) for i in range(len(signals) + 1)]
                    sss.guess_values = [dict(zip(parameters, guessv)) for guessv in guess_values]
                    sss.default_gvs = [dict(zip(parameters, [True] * len(parameters))) for i in range(len(signals) + 1)]

                # Fitting
                try:
                    if sss.fitting_apply_all:
                        guess_values = [sss.guess_values[-1]] * len(signals)
                    else:
                        guess_values = sss.guess_values[:-1]
                    fits = [signal.fit(fit_function, guessv) for signal, guessv in zip(signals, guess_values)]
                    fit_signals, fit_params, param_errors, r_squared = [list(e) for e in zip(*fits)]
                    rel_error = np.array(param_errors) / np.array(fit_params)
                    expander_label = f"__✔ {FITTING_LABEL} ({sss.fitting_model})__"
                except Exception as e:
                    fit_signals, fit_params, param_errors, r_squared = [], [], [], []
                    equation, parameters = "", []
                    rel_error = []
                    print("Fit failed")
                    print(e)

            with st.sidebar.expander(expander_label, sss["fitting_interacted"]):

                def on_change() -> None:
                    """Set fitting_interacted to True and reset guess_values"""

                    set_session_state_value_function("fitting_interacted", True)()
                    sss.guess_values = None

                st.selectbox(
                    label="Model",
                    options=["None"] + list(MODELS.keys()),
                    on_change=on_change,
                    key="fitting_model",
                    help="Use the selected model to fit the data.",
                )

                if sss.fitting_model in MODELS:

                    # Display the equation
                    st.html("Equation: " + equation)

                    if len(signals) > 1:
                        st.checkbox(
                            label="Apply Guess Values To All Signals",
                            key="fitting_apply_all",
                            help="Apply the same guess values to all signals.",
                        )

                        if not sss.fitting_apply_all:
                            names = [signal.get_name(multifile_cond) for signal in signals]
                            signal_dict = dict(zip(names, signals))
                            if sss.fitting_curve_select not in signal_dict:
                                sss.fitting_curve_select = names[0]
                            st.selectbox(
                                label="Curve",
                                options=names,
                                on_change=set_session_state_value_function("fitting_interacted", True),
                                key="fitting_curve_select",
                            )
                            curve_index = names.index(sss.fitting_curve_select)
                        else:
                            curve_index = len(signals)
                    else:
                        curve_index = 0

                    # Guess parameter and value
                    columns = st.columns(2)

                    parameter = columns[0].selectbox(
                        label="Parameter",
                        options=parameters,
                        key="parameter_model_key",
                        on_change=set_session_state_value_function("fitting_interacted", True),
                    )

                    guess_value_key = sss.fitting_model + str(curve_index) + parameter + "guess_value"

                    # If the guess value has not already been stored in the session state...
                    if guess_value_key not in sss:
                        if sss.default_gvs[curve_index][parameter]:
                            precision = 4  # up to 4 digits for default values
                        else:
                            precision = 10  # up to 10 digits for user set values

                        if isinstance(sss.guess_values[curve_index][parameter], str):
                            sss[guess_value_key] = sss.guess_values[curve_index][parameter]
                        else:
                            sss[guess_value_key] = number_to_str(
                                sss.guess_values[curve_index][parameter], precision, "g"
                            )

                    def store_guess_value() -> None:
                        """Store the guess value as a float in the guess_values dictionary"""

                        set_session_state_value_function("fitting_interacted", True)
                        try:
                            sss.guess_values[curve_index][parameter] = float(sss[guess_value_key])
                            sss.default_gvs[curve_index][parameter] = False  # not default value anymore
                        except:
                            pass

                    columns[1].text_input(
                        label="Guess Value",
                        key=guess_value_key,
                        on_change=store_guess_value,
                    )

            # Data reset toggle
            st.sidebar.toggle(
                label="Reset Data Processing Settings Upon Data Change.",
                help="When toggled, this option will reset the data processing settings every time the data is changed.",
                key="reset_settings",
            )

            # ----------------------------------------------- DATA EXPORT ----------------------------------------------

            # Add raw data
            header_name = [signal.get_name(multifile_cond) for signal in raw_signals]
            header_y_name = [signal.y.get_label_raw() for signal in raw_signals]
            data = [signal.y.data for signal in raw_signals]

            # Add smoothed data if exist
            if s_signals:
                header_name += [signal.get_name(multifile_cond) for signal in s_signals]
                header_y_name += [signal.y.get_label_raw() + " (smoothed)" for signal in s_signals]
                data += [signal.y.data for signal in s_signals]

            # Add fitted data if exist
            if fit_signals:
                header_name += [signal.get_name(multifile_cond) for signal in fit_signals]
                header_y_name += [signal.y.get_label_raw() + " (fit)" for signal in fit_signals]
                data += [signal.y.data for signal in fit_signals]

            data = [raw_signals[0].x.data] + data
            header_name = ["Name"] + header_name
            header_y_name = [raw_signals[0].x.get_label_raw()] + header_y_name

            # Generate the export data and download button
            export_data = matrix_to_string(data, [header_name, header_y_name])
            st.sidebar.download_button(
                label="Download Processed Data",
                data=export_data,
                file_name="data.csv",
                use_container_width=True,
                key="data_download_button",
            )

            if fit_params:

                # Header
                z_label = z_dims[0].get_label_raw()
                header_name = [z_label] + parameters

                # Data
                z_values = [z_dim.data for z_dim in z_dims]
                data = [z_values] + [list(f) for f in np.transpose(fit_params)]

                # Generate the export data and download button
                export_data = matrix_to_string(data, header_name)
                st.sidebar.download_button(
                    label="Download Fitting Parameters",
                    data=export_data,
                    file_name="fit_parameters.csv",
                    use_container_width=True,
                    key="fit_download_button",
                )

            # ----------------------------------------------------------------------------------------------------------
            # --------------------------------------------- DATA EXTRACTION --------------------------------------------
            # ----------------------------------------------------------------------------------------------------------

            st.sidebar.markdown(f"## {EXTRACTION_LABEL}")

            # ------------------------------------------------ MAX POINT -----------------------------------------------

            x_max, y_max = None, None

            columns = st.sidebar.columns(2)
            max_button = columns[0].checkbox(
                label="Display Maximum",
                key="max_button",
            )
            max_interp_button = columns[1].checkbox(
                label="Use Interp.",
                key="max_interp_button",
                help="Use cubic interpolation to improve the calculation of the maximum point.",
            )
            if max_button:
                output = list(zip(*[signal.get_max(max_interp_button) for signal in signals]))
                x_max = SignalData(z_dims, output[0], signalname="x_max")
                y_max = SignalData(z_dims, output[1], signalname="y_max")

            # ------------------------------------------------ MIN POINT -----------------------------------------------

            x_min, y_min = None, None

            columns = st.sidebar.columns(2)
            min_button = columns[0].checkbox(
                label="Display Minimum",
                key="min_button",
            )
            min_interp_button = columns[1].checkbox(
                label="Use Interp.",
                key="min_interp_button",
                help="Use cubic interpolation to improve the calculation of the minimum point.",
            )
            if min_button:
                output = list(zip(*[signal.get_min(min_interp_button) for signal in signals]))
                x_min = SignalData(z_dims, output[0], signalname="x_min")
                y_min = SignalData(z_dims, output[1], signalname="y_min")

            # -------------------------------------------------- FWHM --------------------------------------------------

            fwhm, fwhm_xleft, fwhm_yleft, fwhm_xright, fwhm_yright = None, None, None, None, None

            columns = st.sidebar.columns(2)
            fwhm_button = columns[0].checkbox(
                label="Display FWHM",
                key="fwhm_button",
            )
            fwhm_button_interp = columns[1].checkbox(
                label="Use Interp.",
                key="fwhm_button_interp",
                help="Use linear interpolation to improve the calculation of the FWHM.",
            )
            if fwhm_button:
                output = list(zip(*[sig.get_fwhm(fwhm_button_interp) for sig in signals]))
                fwhm = SignalData(z_dims, output[0], signalname="FWHM")
                fwhm_xleft = SignalData(z_dims, output[1], signalname="FWHM (x-left)")
                fwhm_yleft = SignalData(z_dims, output[2], signalname="FWHM (y-left)")
                fwhm_xright = SignalData(z_dims, output[3], signalname="FWHM (x-right)")
                fwhm_yright = SignalData(z_dims, output[4], signalname="FWHM (y-right)")

            # -------------------------------------------------- FWHM --------------------------------------------------

            area = None

            columns = st.sidebar.columns(2)
            area_button = columns[0].checkbox(
                label="Display Area",
                key="area_button",
            )

            if area_button:
                output = [signal.get_area() for signal in signals]
                area = SignalData(z_dims, output, signalname="FWHM")

            # -------------------------------------------- FITTED PARAMETERS -------------------------------------------

            fit_extract_signals = {}
            fit_errors = {}
            if fit_params:
                for i, key in enumerate(parameters):
                    y = Dimension(np.array([fit_param[i] for fit_param in fit_params]), quantity=key)
                    fit_extract_signals[key] = SignalData(z_dims, y, signalname=key)
                    fit_errors[key] = np.array([param_error[i] for param_error in param_errors])

            # ----------------------------------------------------------------------------------------------------------
            # ------------------------------------------------- DISPLAY ------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------

            extracted_data = [x_max, x_min, fwhm, area] + list(fit_extract_signals.keys())
            n_extracted = len([e for e in extracted_data if e is not None])

            if len(signals) == 1 and n_extracted:
                columns = st.columns([3, 1])
                plot_spot = columns[0].container()
                info_spot = columns[1].container()
                info_spot.markdown("### About Your Data")
                info_spot.number_input(
                    label="Precision",
                    min_value=0,
                    key="precision_input",
                )

                # Fit results
                if fit_params:
                    info_spot.markdown("#### Fitting Results")
                    fit_params = fit_params[0]
                    r_squared = r_squared[0]
                    param_errors = param_errors[0]
                    rel_error = rel_error[0]

                    # Add the R2 to the list of parameters and errors
                    fit_params = list(fit_params) + [r_squared]
                    param_errors = list(param_errors) + [float("nan")]
                    parameters += ["R<sup>2</sup>"]

                    # Convert the parameters and errors to strings
                    fit_params_str = [number_to_str(f, sss.precision_input, "f", True) for f in fit_params]
                    rel_error_str = [f"{f * 100:.2f}" for f in rel_error]

                    # Generate the dataframe and display it
                    df = pd.DataFrame(
                        data=[fit_params_str, rel_error_str],
                        columns=parameters,
                        index=["Value", "Relative Error (%)"],
                    )
                    df = df.transpose()
                    df.columns.filename = "Parameter"
                    info_spot.html("Equation: " + equation)
                    info_spot.html(generate_html_table(df))

                # Max point
                if x_max:
                    info_spot.markdown("##### Maximum Point")
                    if not np.isnan(x_max.y[0].data):
                        info_spot.html(x_max.y[0].get_value_label_html(sss.precision_input, "f", True))
                        info_spot.html(y_max.y[0].get_value_label_html(sss.precision_input, "f", True))
                    else:
                        info_spot.markdown("Could not calculate the maximum point.")

                # Min point
                if x_min:
                    info_spot.markdown("##### Minimum Point")
                    if not np.isnan(x_min.y[0].data):
                        info_spot.html(x_min.y[0].get_value_label_html(sss.precision_input, "f", True))
                        info_spot.html(y_min.y[0].get_value_label_html(sss.precision_input, "f", True))
                    else:
                        info_spot.markdown("Could not calculate the minimum point.")

                # FWHM
                if fwhm:
                    info_spot.markdown("##### FWHM")
                    if not np.isnan(fwhm.y[0].data):
                        info_spot.html(fwhm.y[0].get_value_label_html(sss.precision_input, "f", True))
                    else:
                        info_spot.markdown("Could not calculate the FWHM.")

                # Area
                if area:
                    info_spot.markdown("##### Area")
                    if not np.isnan(area.y[0].data):
                        info_spot.html(area.y[0].get_value_label_html(sss.precision_input, "f", True))
                    else:
                        info_spot.markdown("Could not calculate the Area.")

            else:
                plot_spot = st.container()

            # ------------------------------------------- EXTRACTED DOWNLOAD -------------------------------------------

            if any([x_max, x_min, fwhm]):
                header_name = [z_dims[0].get_label_raw()]
                data = [[z_dim.data for z_dim in z_dims]]

                if x_max:
                    header_name += [x_max.y.get_label_raw(), y_max.y[0].get_label_raw()]
                    data += [x_max.y.data, y_max.y.data]
                if x_min:
                    header_name += [x_min.y.get_label_raw(), y_min.y[0].get_label_raw()]
                    data += [x_min.y.data, y_min.y.data]
                if fwhm:
                    header_name += [fwhm.y.get_label_raw()]
                    data += [fwhm.y.data]
                if area:
                    header_name += [area.y.get_label_raw()]
                    data += [area.y.data]

                # Generate the export data and download button
                export_data = matrix_to_string(data, header_name)
                st.sidebar.download_button(
                    label="Download Extracted Parameters",
                    data=export_data,
                    file_name="extracted_parameters.csv",
                    use_container_width=True,
                    key="extracted_download_button",
                )

            # Determine how many figures to create
            if len(signals) > 1:
                nb_figures = 1 + n_extracted
            else:
                nb_figures = 1
            figures = [make_figure() for i in range(nb_figures)]

            # ------------------------------------------------ MAIN DATA -----------------------------------------------

            # Plot the raw signals if they exist
            if sss.display_raw or not s_signals:
                plot_signals(
                    signals=raw_signals,
                    figure=figures[0],
                    legendgroup="Raw Data" if (s_signals or fit_signals) and len(signals) > 1 else None,
                    legendgrouptitle_text="Raw Data" if (s_signals or fit_signals) and len(signals) > 1 else None,
                    filename=multifile_cond,
                )

            # Plot the smoothed signals if they exist
            if s_signals:
                plot_signals(
                    signals=s_signals,
                    figure=figures[0],
                    legendgroup="Smoothed Data" if sss.display_raw and len(signals) > 1 else None,
                    legendgrouptitle_text="Smoothed Data" if sss.display_raw and len(signals) > 1 else None,
                    filename=multifile_cond,
                )

            # Plot the fit signals if they exist
            if fit_signals:
                plot_signals(
                    signals=fit_signals,
                    figure=figures[0],
                    legendgroup="Fitted Data" if len(signals) > 1 else None,
                    legendgrouptitle_text="Fitted Data" if len(signals) > 1 else None,
                    filename=multifile_cond,
                    line=dict(dash="dash"),
                )

            # Display the maximum point(s)
            if x_max:
                for i, (x, y) in enumerate(zip(x_max.y.data, y_max.y.data)):
                    if not np.isnan(x):
                        scatter_plot(
                            figure=figures[0],
                            x_data=x,
                            y_data=y,
                            legendgroup="Max. Point",
                            label="Max. Point",
                            marker=dict(color="#CD5C5C"),
                            showlegend=i == 0,
                        )

            # Display the minimum point(s)
            if x_min:
                for i, (x, y) in enumerate(zip(x_min.y.data, y_min.y.data)):
                    if not np.isnan(x):
                        scatter_plot(
                            figure=figures[0],
                            x_data=x,
                            y_data=y,
                            legendgroup="Min. Point",
                            label="Min. Point",
                            marker=dict(color="#87CEEB"),
                            showlegend=i == 0,
                        )

            # Display the FWHM(s)
            if fwhm:
                for i, (f, xleft, yleft, xright, yright) in enumerate(
                    zip(fwhm.y.data, fwhm_xleft.y.data, fwhm_yleft.y.data, fwhm_xright.y.data, fwhm_yright.y.data)
                ):
                    if not np.isnan(f):
                        scatter_plot(
                            figure=figures[0],
                            x_data=[xleft, xright],
                            y_data=[yleft, yright],
                            label="FWHM",
                            legendgroup="FWHM",
                            marker=dict(color="#66CDAA"),
                            showlegend=i == 0,
                        )

            # --------------------------------------------- EXTRACTED DATA ---------------------------------------------

            datasets = [raw_signals + (s_signals if s_signals else []) + (fit_signals if fit_signals else [])]

            if len(signals) > 1:
                figure_index = 1

                # Max point
                if x_max and y_max:
                    y_max.plot(figure=figures[figure_index], plot_method="Scatter")
                    x_max.plot(figure=figures[figure_index], secondary_y=True, plot_method="Scatter")
                    figure_index += 1
                    datasets.append([x_max, y_max])

                # Min point
                if x_min and y_min:
                    y_min.plot(figure=figures[figure_index], plot_method="Scatter")
                    x_min.plot(figure=figures[figure_index], secondary_y=True, plot_method="Scatter")
                    figure_index += 1
                    datasets.append([x_min, y_min])

                # FWHM
                if fwhm:
                    fwhm.plot(figure=figures[figure_index], plot_method="Scatter")
                    figure_index += 1
                    datasets.append([fwhm])

                # Area
                if area:
                    area.plot(figure=figures[figure_index], plot_method="Scatter")
                    figure_index += 1
                    datasets.append([area])

                # Fit parameters
                for key in fit_extract_signals:
                    fit_extract_signals[key].plot(
                        figure=figures[figure_index],
                        error_y=dict(
                            type="data",
                            array=fit_errors[key],
                            visible=True,
                        ),
                        plot_method="Scatter",
                    )
                    figure_index += 1
                    datasets.append([fit_extract_signals[key]])

            # -------------------------------------------- STREAMLIT FIGURES -------------------------------------------

            if len(figures) == 1:
                with plot_spot:
                    display_data(figures[0], datasets[0], 1, multifile_cond)
            else:
                columns = plot_spot.columns(2)
                for i, figure in enumerate(figures):
                    figure.update_layout(height=400)
                    with columns[i % 2]:
                        display_data(figure, datasets[i], i, multifile_cond)

        # Signal dictionary case
        else:
            print("Displaying a dictionary")
            plot_spot = st.container()
            figures = []
            for i, key in enumerate(signals):
                figure = make_figure()
                figure.update_layout(title=key)
                plot_signals(
                    signals=signals[key],
                    figure=figure,
                    filename=multifile_cond,
                )
                figures.append(figure)

            if len(figures) == 1:
                with plot_spot:
                    display_data(figures[0], list(signals.values())[0], 1, multifile_cond)
            else:
                columns = plot_spot.columns(2)
                for i, figure in enumerate(figures):
                    figure.update_layout(height=400)
                    with columns[i % 2]:
                        key = list(signals.keys())[i]
                        display_data(figures[i], signals[key], i, multifile_cond)

# ----------------------------------------------------- INFORMATION ----------------------------------------------------

with st.expander("About", expanded=not file):
    text = f"""
    *Raft* is a user-friendly web app designed to give you a quick look at your data without needing to write a single line 
    of code or spend time reformatting files. It supports a wide range of data file formats and takes care of the behind-
    the-scenes work of reading and processing the data, so you can jump straight to visualising and exploring your results.
    
    The idea came from a simple frustration: too often, valuable time is lost just trying to open and make sense of a data 
    file. This app removes that barrier. Whether you are checking raw instrument output, exploring simulation results, or 
    comparing experimental datasets, it lets you plot and inspect the data in just a few clicks. It is especially handy in 
    fast-paced research environments where quick decisions depend on a fast understanding of the data.

    App created and maintained by [{__author__}](https://emmanuelpean.me/).  
    [Version {__version__}]({__github__}) (last updated: {__date__})."""
    st.info(text)

# --------------------------------------------------- DATA PROCESSING --------------------------------------------------

with st.expander("Data Processing"):
    text = """*Raft* also offers basic data processing capabilities, allowing you to quickly and easily process and 
    extract meaningful information from your data. Data are processed sequentially using a linear pipeline. This means 
    each method in the list operates on the output of the previous one. In particular, normalisation and fitting are 
    applied to the smoothed signal. The available options include:"""
    st.markdown(text)

    st.markdown(f"""##### {AVERAGING_LABEL}""")
    text = """If multiple signals are displayed, they can be averaged in groups of X signals. For example, if 5 signals 
    are displayed and averaged every 2 signals, this yields 2 averaged signals: the first is the average of the 1st and 
    2nd raw signals, the second is the average of the 3rd and 4th. The 5th signal is discarded, as it does not complete 
    a full group."""
    st.markdown(text)

    st.markdown(f"""##### {BACKGROUND_LABEL}""")
    text = """If a __lower__ and __upper range__ of x-values is provided, the corresponding y-values are averaged and 
    subtracted from the entire y-values to remove the background."""
    st.markdown(text)

    st.markdown(f"""##### {RANGE_LABEL}""")
    text = """If a __lower__ or __upper range__ of x-values is provided, the data are limited to this range."""
    st.markdown(text)

    st.markdown(f"""##### {INTERP_LABEL}""")
    text = """Data can be interpolated using different methods:
    * __Fixed Step__ interpolation – Data are interpolated using a specified step size.
    * __Point Count__ interpolation – Data are interpolated to fit a specified number of points."""
    st.markdown(dedent(text))

    st.markdown(f"""##### {DERIVE_LABEL}""")
    text = """The n order derivative can be calculated by setting the __Derivative Order__."""
    st.markdown(dedent(text))

    st.markdown(f"""##### {SMOOTHING_LABEL}""")
    text = """Applies a Savitzky–Golay smoothing filter using the specified __Filter Length__ and __Polynomial Order__ 
    (__subplot a__)."""
    st.markdown(dedent(text))

    st.markdown(f"##### {NORM_LABEL}")
    text = """Data can be normalised using different methods:
    * __Max. normalisation__ – the y-values are normalised with respect to their maximum value.
    * __Feature scaling__ – the y-values are normalised based on specified minimum and maximum values."""
    st.markdown(dedent(text))

    st.markdown(f"##### {FITTING_LABEL}")
    models = "".join(["\n* " + key for key in MODELS.keys()])
    text = f"""Data can be fitted using the following __models__: {models}\n
    Initial guess values are automatically estimated from the data but can be manually adjusted."""
    st.markdown(dedent(text))

    st.markdown(f"##### {EXTRACTION_LABEL}")
    text = """The following information can be extracted from the processed data:
    * __Maximum point__: The x and y coordinates corresponding to the peak y-value in the data (__subplot c__).
    * __Minimum point__: The x and y coordinates corresponding to the lowest y-value in the data (__subplot c__).
    
    For both maximum and minimum points, the precision of the estimated positions can be enhanced using cubic __interpolation__ 
    around the peak maximum, which refines the estimate beyond the raw data resolution.
    * __Full Width at Half Maximum (FWHM)__: The width of the peak measured between the two x-values where the y-value equals 
    half of the maximum y-value. This provides a useful metric of the peak sharpness or spread (__subplot b__).
    
    Similar to the max/min points, the precision of the FWHM measurement can be further refined using linear __interpolation__. 
    
    * __Area__: The area under the curve. Note that a negative curve yields a negative area."""
    st.markdown(dedent(text))
    st.html(render_image(DATA_PROCESSING_PATH, 1000))

# -------------------------------------------------- DATA FILE FORMATS -------------------------------------------------

with st.expander("Data File Formats"):
    st.markdown("""These are example data files for the different supported file types.""")

    n_columns = 5

    # List the software
    softwares = set([filetype for filetype in FILE_TYPE_DICT.values() if "(" in filetype])
    softwares = sorted(set([filetype[: filetype.index("(") - 1] for filetype in softwares]))

    # For each software, display them
    for i, software in enumerate(softwares):
        if i % n_columns == 0:
            columns = st.columns(n_columns)
        filetypes = [filetype for filetype in FILE_TYPE_DICT.values() if software in filetype]
        paths = [path for path in FILE_TYPE_DICT if FILE_TYPE_DICT[path] in filetypes]
        paths, filetypes = zip(*sorted(zip(paths, filetypes), key=lambda v: os.path.basename(v[0])))
        filetypes = make_unique(filetypes)

        with columns[i % n_columns].popover(software, use_container_width=True):

            for path, filetype in zip(paths, filetypes):
                content = read_file(path, "rb", None)
                st.download_button(
                    label=f"Download {filetype}",
                    data=content,
                    file_name=os.path.basename(path),
                    use_container_width=True,
                )

# ------------------------------------------------------ CHANGELOG -----------------------------------------------------

with st.expander("Changelog"):
    st.markdown(read_file(os.path.join(PROJECT_PATH, "CHANGELOG.md")), unsafe_allow_html=True)

# ----------------------------------------------------- DISCLAIMER -----------------------------------------------------

with st.expander("License & Disclaimer"):
    st.markdown(read_file(os.path.join(PROJECT_PATH, "LICENSE.txt")))

# Stop the profiler
if profiler:
    profiler.stop()
