"""Graphical user interface of Raft"""

import os

import numpy as np
import pandas as pd
import streamlit as st

from data_files import FILETYPES, read_data_file
from fitting import MODELS, get_model_parameters
from plot import plot_signals, scatter_plot, ps, go
from resources import LOGO_PATH, CSS_STYLE_PATH, ICON_PATH, DATA_PROCESSING_PATH, LOGO_TEXT_PATH
from signaldata import Dimension, SignalData
from utils import (
    render_image,
    matrix_to_string,
    read_file,
    number_to_str,
    generate_html_table,
    normalise_list,
    tab_bar,
    dedent,
    make_unique,
)

__version__ = "2.0.0"
__name__ = "Raft"
__date__ = "March 2025"
__author__ = "Emmanuel V. Péan"
__github__ = "https://github.com/Emmanuelpean/raft"


# ------------------------------------------------------ CONSTANTS -----------------------------------------------------

# Project path
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Section labels
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
st.set_page_config(__name__, page_icon=ICON_PATH, layout="wide")
st.logo(LOGO_TEXT_PATH, icon_image=LOGO_PATH)

# Load the custom CSS
css_content = read_file(CSS_STYLE_PATH)
st.html(f"<style>{css_content}</style>")

# ---------------------------------------------------- SESSION STATE ---------------------------------------------------

# Create a shortened name for the session state
sss = st.session_state

# Dictionary storing the default values of session state values (including widgets and interaction status)
settings_defaults = {
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
}


# Add the settings to the session state using their default value
def set_default_settings(reset: bool = False) -> None:
    """Store the settings in the session state using their default value
    :param reset: if True, reset the stored value"""

    for key in settings_defaults:
        if key not in sss or reset:
            sss[key] = settings_defaults[key]


set_default_settings()


def refresh_session_state_settings() -> None:
    """Refresh the session state settings. To be called when new widgets are displayed"""

    for key in settings_defaults:
        sss[key] = sss[key]


def set_session_state_value_function(
    key: str,
    value: any,
) -> callable:
    """Return a function that changes the value of a key in the session state
    :param key: session state key
    :param value: new value"""

    def function() -> None:
        """Set the session state key value"""

        sss[key] = value

    return function


# Add data and the file_uploader key to the session state
if "data" not in sss:
    sss.data = None
if "file_uploader" not in sss:
    sss.file_uploader = "file_uploader_key0"
if "reset_settings" not in sss:
    sss.reset_settings = True


def reset_data() -> None:
    """Reset the settings and remove the stored data"""

    reset_interacted()
    sss.data = None  # remove the stored data
    if sss.reset_settings:
        set_default_settings(True)  # reset the default settings


def reset_interacted() -> None:
    """Reset the interacted session state keys"""

    for key in settings_defaults:
        if "interacted" in key:
            sss[key] = False


def display_data(
    fig: go.Figure,
    dataset: list[SignalData],
    key: int,
) -> None:
    """Display a figure and the associated data in different tabs
    :param fig: figure object
    :param dataset: data to display in the dataframe
    :param key: figure key int"""

    tabs = st.tabs(["Graph", "Data"])

    # Show the figure
    tabs[0].plotly_chart(fig, use_container_width=True, key=f"figure_{key}")

    # Get the data amd columns
    x_data = dataset[0].x.data
    ys_data = [d.y.data for d in dataset]
    df_columns = make_unique([dataset[0].x.get_label_html()] + [d.get_name(True) for d in dataset])

    # Dataframe
    dataframe = pd.DataFrame({name: array for name, array in zip(df_columns, [x_data] + ys_data)})
    tabs[1].dataframe(dataframe.style.format("{:5g}"), use_container_width=True, hide_index=True)


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
    key="filetype_select",
)

# Detect message
detect_placeholder = st.sidebar.empty()

# -------------------------------------------------------- LOGO --------------------------------------------------------

refresh_session_state_settings()  # refresh the session state

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
        print("Reading data files")
        signals, filetype = read_data_file(file, filetype_select)

        # If signals have been loaded, normalise them
        if signals:
            signals = normalise_list(signals)

        # Store them in the session state
        sss.data = [signals, filetype]

    # Otherwise load the data from the session state
    else:
        signals, filetype = sss.data

    if filetype_select == FILETYPES[0]:
        detect_placeholder.markdown(f"Detected: {filetype}")

    # If the signal could not be loaded, display a warning message
    if filetype is None:
        st.markdown("###")  # some space on top for the st.logo
        text = "Unable to read the file"
        if tab == FILE_UPLOADER_MODES[0]:
            st.warning(text + ".")
        else:
            st.warning(text + "s.")

    # Show the options for selecting a specific signal
    else:

        # ----------------------------------------------- DATA SELECTION -----------------------------------------------

        # Select the data type to plot if signal is a dictionary
        if isinstance(signals, dict):
            selection = st.sidebar.selectbox(
                label="Data Type",
                options=["All"] + list(signals.keys()),
                on_change=reset_interacted,
                key="type_select",
            )
            if selection in signals:
                signals = signals[selection]

        # Select the signal in a list if multiple signals
        if isinstance(signals, list) and len(signals) > 1:
            names = [signal.get_name(False) for signal in signals]
            signal_dict = dict(zip(names, signals))
            col_selection = st.sidebar.selectbox(
                label="Data Selection",
                options=["All"] + sorted(signal_dict.keys()),
                on_change=reset_interacted,
                key="data_select",
            )
            if col_selection in signal_dict:
                signals = [signal_dict[col_selection]]

        # ----------------------------------------------- SIGNAL SORTING -----------------------------------------------

        # Default list of z dimensions
        z_dims = [Dimension(i + 1) for i in range(len(signals))]

        if isinstance(signals, list) and len(signals) > 1:

            # List of all z_dict keys
            z_keys = list(set([key for signal in signals for key in signal.z_dict]))

            # Select the key used to sort the data
            help_str = dedent(
                """Select the quantity by which to sort the data files. For example:
                * Filename – sorts the files alphabetically by their filenames.
                * Timestamp – sorts the files according to the timestamps extracted from their contents.
                * Emission Wavelength - sort the files according to the emission wavelength extracted from their contents
                * ..."""
            )
            sorting_key = st.sidebar.selectbox(
                label="Data Sorting",
                options=["Filename"] + z_keys,
                help=help_str,
            )

            if sorting_key == "Filename":
                signals = sorted(signals, key=lambda signal: signal.name)  # TODO decide which name to use
            else:
                try:
                    signals = sorted(signals, key=lambda v: v.z_dict[sorting_key].data)
                    z_dims = [signal.z_dict[sorting_key] for signal in signals]
                except Exception as e:
                    print("An error occurred while trying to sort the signals")
                    print(e)

        # ----------------------------------------------------------------------------------------------------------
        # ----------------------------------------------- DATA PROCESSING ----------------------------------------------
        # ----------------------------------------------------------------------------------------------------------

        if isinstance(signals, list):

            st.sidebar.markdown("## Data Processing")

            # ----------------------------------------------- BACKGROUND -----------------------------------------------

            try:
                xrange = [float(sss.bckg_range_input1), float(sss.bckg_range_input2)]
                signals = [signal.remove_background(xrange) for signal in signals]
                expander_label = f"__✔ {BACKGROUND_LABEL} {xrange[0]} - {xrange[1]}__"
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

            try:
                xrange = [float(sss.range_input1), float(sss.range_input2)]
                signals = [signal.reduce_range(xrange) for signal in signals]
                expander_label = f"__✔ {RANGE_LABEL} {xrange[0]} - {xrange[1]}__"
            except:
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
                    dx_str = number_to_str(dx, 3, True)
                    expander_label = f"__✔ {INTERP_LABEL} (step = {dx_str} {signals[0].x.get_unit_label_html()})__"
            except:
                expander_label = INTERP_LABEL

            with st.sidebar.expander(expander_label, sss["interp_interacted"]):

                help_str = dedent(
                    """Interpolate the data using different methods:
                    * __Fixed Step__ interpolation – Data are interpolated using a specified step size.
                    * __Point Count__ interpolation – Data are interpolated to fit a specified number of points."""
                )

                st.segmented_control(
                    label="Interpolation Type",
                    options=INTERP_OPTIONS,
                    key="interp_type",
                    on_change=set_session_state_value_function("interp_interacted", True),
                    help=help_str,
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

                help_str = dedent(
                    """Data can be normalised using different methods:
                    * __Max. normalisation__ – the y-values are normalised with respect to their maximum value.
                    * __Feature scaling__ – the y-values are normalised based on specified minimum and maximum values."""
                )

                st.segmented_control(
                    label="Normalisation Type",
                    options=NORM_OPTIONS,
                    key="norm_type",
                    on_change=set_session_state_value_function("norm_interacted", True),
                    help=help_str,
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
            try:
                if sss.fitting_model in MODELS:
                    # Get the fit function, equation and guess function, and the function parameters
                    fit_function, equation, guess_function = MODELS[sss.fitting_model]
                    parameters = get_model_parameters(fit_function)

                    # Reset the guess value default dict is set to None
                    if sss.guess_values is None:
                        try:
                            index = int(len(signals) / 2)
                            guess_values = guess_function(signals[index].x.data, signals[index].y.data)
                        except:
                            guess_values = np.ones(len(parameters))
                        sss.guess_values = dict(zip(parameters, guess_values))

                    fit_signals, fit_params, param_errors, r_squared = [], [], [], []
                    for signal in signals:
                        output = signal.fit(fit_function, sss.guess_values)
                        fit_signals.append(output[0])
                        fit_params.append(output[1])
                        param_errors.append(output[2])
                        r_squared.append(output[3])
                    rel_error = np.array(param_errors) / np.array(fit_params)

                    expander_label = f"__✔ {FITTING_LABEL} ({sss.fitting_model})__"
            except Exception as e:
                print("Fit failed")
                print(e)
                expander_label = FITTING_LABEL
                fit_signals, fit_params, param_errors, r_squared = [], [], [], []
                equation, parameters = "", []
                rel_error = []

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

                    # Guess parameter and value
                    columns = st.columns(2)

                    parameter = columns[0].selectbox(
                        label="Parameter",
                        options=parameters,
                        key="parameter_model_key",
                        on_change=set_session_state_value_function("fitting_interacted", True),
                    )

                    guess_value_key = sss.fitting_model + parameter + "guess_value"

                    if guess_value_key not in sss:
                        sss[guess_value_key] = number_to_str(sss.guess_values[parameter], 4, False)

                    def store_guess_value() -> None:
                        """Store the guess value as a float in the guess_values dictionary"""

                        set_session_state_value_function("fitting_interacted", True)
                        try:
                            sss.guess_values[parameter] = float(sss[guess_value_key])
                        except:
                            pass

                    columns[1].text_input(
                        label="Guess Value",
                        key=guess_value_key,
                        on_change=store_guess_value,
                    )

            # Set the fit signals as the new default
            if fit_signals:
                signals = fit_signals

            # Data reset toggle
            st.sidebar.toggle(
                label="Reset Data Processing Settings Upon Data Change.",
                help="When toggled, this option will reset the data processing settings every time the data is changed.",
                key="reset_settings",
            )

            # ----------------------------------------------- DATA EXPORT ----------------------------------------------

            if len(signals) == 1:

                # Add raw data
                header = [raw_signals[0].x.get_label_raw(), raw_signals[0].y.get_label_raw()]
                data = [raw_signals[0].x.data, raw_signals[0].y.data]

                # Add smoothed data if exist
                if s_signals:
                    header += [raw_signals[0].y.get_label_raw() + " (smoothed)"]
                    data += [s_signals[0].y.data]

                # Add fitted data if exist
                if fit_signals:
                    header += [raw_signals[0].y.get_label_raw() + " (fit)"]
                    data += [fit_signals[0].y.data]

                # Generate the export data and download button
                export_data = matrix_to_string(data, header)
                st.sidebar.download_button(
                    label="Download Processed Data",
                    data=export_data,
                    file_name="data.csv",
                    use_container_width=True,
                    key="download_button",
                )

            # ----------------------------------------------------------------------------------------------------------
            # --------------------------------------------- DATA EXTRACTION --------------------------------------------
            # ----------------------------------------------------------------------------------------------------------

            st.sidebar.markdown(f"## {EXTRACTION_LABEL}")

            # ------------------------------------------------ MAX POINT -----------------------------------------------

            x_max, y_max, i_max = None, None, None
            xmax_signal, ymax_signal = None, None

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
                try:
                    x_max, y_max, i_max = zip(*[signal.get_max(max_interp_button) for signal in signals])
                    xmax_signal, ymax_signal = SignalData(z_dims, x_max, "x_max"), SignalData(z_dims, y_max, "y_max")
                except:
                    pass

            # ------------------------------------------------ MIN POINT -----------------------------------------------

            x_min, y_min, i_min = None, None, None
            xmin_signal, ymin_signal = None, None

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
                try:
                    x_min, y_min, i_min = zip(*[signal.get_min(min_interp_button) for signal in signals])
                    xmin_signal, ymin_signal = SignalData(z_dims, x_min, "x_min"), SignalData(z_dims, y_min, "y_min")
                except:
                    pass

            # -------------------------------------------------- FWHM --------------------------------------------------

            fwhm, x_left, y_left, x_right, y_right = None, None, None, None, None
            fwhm_signal = None

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
                try:
                    fwhm, x_left, y_left, x_right, y_right = zip(*[sig.get_fwhm(fwhm_button_interp) for sig in signals])
                    fwhm_signal = SignalData(z_dims, fwhm, "FWHM")
                except:
                    pass

            # -------------------------------------------- FITTED PARAMETERS -------------------------------------------

            fit_extract_signals = {}
            if fit_params:
                for i, key in enumerate(parameters):
                    y = Dimension(np.array([fit_param[i] for fit_param in fit_params]), quantity=key)
                    fit_extract_signals[key] = SignalData(z_dims, y)

            # ----------------------------------------------------------------------------------------------------------
            # ------------------------------------------------- DISPLAY ------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------

            extracted_data = [x_max, x_min, fwhm] + list(fit_extract_signals.keys())
            n_extracted = len([e for e in extracted_data if e is not None])

            if len(signals) == 1 and n_extracted:
                columns = st.columns([3.5, 1])
                plot_spot = columns[0].container()
                info_spot = columns[1].container()
                info_spot.markdown("### About Your Data")

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
                    fit_params_str = [number_to_str(f, 4, True) for f in fit_params]
                    rel_error_str = [f"{f * 100:.2f}" for f in rel_error]

                    # Generate the dataframe and display it
                    df = pd.DataFrame(
                        data=[fit_params_str, rel_error_str],
                        columns=parameters,
                        index=["Value", "Relative Error (%)"],
                    )
                    df = df.transpose()
                    df.columns.name = "Parameter"
                    info_spot.html("Equation: " + equation)
                    info_spot.html(generate_html_table(df))

                # Max point
                if x_max and y_max:
                    info_spot.markdown("##### Maximum Point")
                    info_spot.html(x_max[0].get_value_label_html())
                    info_spot.html(y_max[0].get_value_label_html())

                # Min point
                if x_min and y_min:
                    info_spot.markdown("##### Minimum Point")
                    info_spot.html(x_min[0].get_value_label_html())
                    info_spot.html(y_min[0].get_value_label_html())

                # FWHM
                if fwhm:
                    info_spot.markdown("##### FWHM")
                    if not np.isnan(fwhm[0].data):
                        info_spot.html(fwhm[0].get_value_label_html())
                    else:
                        info_spot.markdown("Could not calculate the FWHM.")
            else:
                plot_spot = st.container()

            # Determine how many figures to create
            if len(signals) > 1:
                nb_figures = 1 + n_extracted
            else:
                nb_figures = 1
            figures = [ps.make_subplots(1, 1, specs=[[{"secondary_y": True}]]) for i in range(nb_figures)]

            # ------------------------------------------------ MAIN DATA -----------------------------------------------

            # Plot the raw signals if they exist
            if sss.display_raw or not s_signals:
                plot_signals(
                    signals=raw_signals,
                    figure=figures[0],
                    legendgroup="Raw Data",
                    legendgrouptitle_text="Raw Data",
                )

            # Plot the smoothed signals if they exist
            if s_signals:
                plot_signals(
                    signals=s_signals,
                    figure=figures[0],
                    legendgroup="Smoothed Data",
                    legendgrouptitle_text="Smoothed Data",
                )

            # Plot the fit signals if they exist
            if fit_signals:
                plot_signals(
                    signals=fit_signals,
                    figure=figures[0],
                    legendgroup="Fitted Data",
                    legendgrouptitle_text="Fitted Data",
                )

            # Display the maximum point(s)
            if x_max and y_max:
                for x, y in zip(x_max, y_max):
                    scatter_plot(
                        figure=figures[0],
                        x_data=x.data,
                        y_data=y.data,
                        legendgroup="Max. Point",
                        label="Max. Point",
                        marker=dict(color="#CD5C5C"),
                        showlegend=x == x_max[0],
                    )

            # Display the minimum point(s)
            if x_min and y_min:
                for x, y in zip(x_min, y_min):
                    scatter_plot(
                        figure=figures[0],
                        x_data=x.data,
                        y_data=y.data,
                        legendgroup="Min. Point",
                        label="Min. Point",
                        marker=dict(color="#87CEEB"),
                        showlegend=x == x_min[0],
                    )

            # Display the FWHM(s)
            if fwhm:
                for f, x1, y1, x2, y2 in zip(fwhm, x_left, y_left, x_right, y_right):
                    if not np.isnan(f.data):
                        scatter_plot(
                            figure=figures[0],
                            x_data=[x1.data, x2.data],
                            y_data=[y1.data, y2.data],
                            label="FWHM",
                            legendgroup="FWHM",
                            marker=dict(color="#66CDAA"),
                            showlegend=f == fwhm[0],
                        )

            # --------------------------------------------- EXTRACTED DATA ---------------------------------------------

            datasets = [raw_signals + (s_signals if s_signals else []) + (fit_signals if fit_signals else [])]
            if len(signals) > 1:
                figure_index = 1
                # Max point
                if xmax_signal and ymax_signal:
                    ymax_signal.plot(figure=figures[figure_index])
                    xmax_signal.plot(figure=figures[figure_index], secondary_y=True)
                    figure_index += 1
                    datasets.append([xmax_signal, ymax_signal])
                    print(datasets)

                # Min point
                if xmin_signal and ymin_signal:
                    ymin_signal.plot(figure=figures[figure_index])
                    xmin_signal.plot(figure=figures[figure_index], secondary_y=True)
                    figure_index += 1
                    datasets.append([xmin_signal, ymin_signal])

                # FWHM
                if fwhm_signal:
                    fwhm_signal.plot(figure=figures[figure_index])
                    figure_index += 1
                    datasets.append([fwhm_signal])

                # Fit parameters
                for key in fit_extract_signals:
                    fit_extract_signals[key].plot(figure=figures[figure_index])
                    figure_index += 1
                    datasets.append([fit_extract_signals[key]])

            # -------------------------------------------- STREAMLIT FIGURES -------------------------------------------

            if len(figures) == 1:
                with plot_spot:
                    display_data(figures[0], datasets[0], 1)
            else:
                columns = plot_spot.columns(2)
                for i, figure in enumerate(figures):
                    figure.update_layout(height=400)
                    print(datasets)
                    with columns[i % 2]:
                        display_data(figure, datasets[i], i)

        # Signal dictionary case
        else:
            print("Displaying a dictionary")
            plot_spot = st.container()
            figures = []
            for i, key in enumerate(signals):
                figure = ps.make_subplots(1, 1, specs=[[{"secondary_y": True}]])
                figure.update_layout(title=key)
                plot_signals(signals[key], figure)
                figures.append(figure)

            if len(figures) == 1:
                with plot_spot:
                    display_data(figures[0], list(signals.values())[0], 1)
            else:
                columns = plot_spot.columns(2)
                for i, figure in enumerate(figures):
                    figure.update_layout(height=400)
                    with columns[i % 2]:
                        key = list(signals.keys())[i]
                        display_data(figures[i], signals[key], i)


# ----------------------------------------------------- INFORMATION ----------------------------------------------------


with st.expander("About", expanded=not file):
    text = f"""*Raft* is a free tool to plot the content a various data files. Just drag and drop your file and get 
    the relevant information from it!  
    App created and maintained by [{__author__}](https://emmanuelpean.streamlit.app/).  
    [Version {__version__}]({__github__}) (last updated: {__date__})."""
    st.info(text)


with st.expander("Data Processing"):
    text = """*Raft* also offers basic data processing capabilities, allowing you to quickly and easily process and 
    extract meaningful information from your data. Data are processed sequentially using a linear pipeline. This means 
    each method in the list operates on the output of the previous one. In particular, normalisation and fitting are 
    applied to the smoothed signal. The available options include:"""
    st.markdown(text)

    st.markdown(f"""##### {BACKGROUND_LABEL}""")
    text = """If a __lower__ and __upper range__ of x-values is provided, the corresponding y-values are averaged and 
    subtracted from the entire y-values to remove the background."""
    st.markdown(text)

    st.markdown(f"""##### {RANGE_LABEL}""")
    text = """If a __lower__ and __upper range__ of x-values is provided, the data are limited to this range."""
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
    half of the maximum y-value. This provides a useful metric of the peak’signal sharpness or spread (__subplot b__).
    
    Similar to the max/min points, the precision of the FWHM measurement can be further refined using linear __interpolation__. """
    st.markdown(dedent(text))
    st.html(render_image(DATA_PROCESSING_PATH, 1000))

# ------------------------------------------------------ CHANGELOG -----------------------------------------------------

with st.expander("Changelog"):
    st.markdown(read_file(os.path.join(PROJECT_PATH, "CHANGELOG.md")))

# ----------------------------------------------------- DISCLAIMER -----------------------------------------------------

with st.expander("License & Disclaimer"):
    st.markdown(read_file(os.path.join(PROJECT_PATH, "LICENSE.txt")))
