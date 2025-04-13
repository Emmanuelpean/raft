"""Graphical user interface of Raft"""

import os

import numpy as np
import pandas as pd
import streamlit as st

from data_files import FUNCTIONS, detect_file_type
from fitting import MODELS, get_model_parameters
from plot import plot, scatter_plot
from resources import LOGO_PATH, LOGO_TEXT_PATH, CSS_STYLE_PATH, ICON_PATH, DATA_PROCESSING_PATH
from signaldata import SignalData, Dimension
from utils import render_image, matrix_to_string, read_file, number_to_str, generate_html_table

__version__ = "2.0.0"
__name__ = "Raft"
__date__ = "March 2025"
__author__ = "Emmanuel V. Péan"
__github__ = "https://github.com/Emmanuelpean/raft"


project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------------------------------------------------------- SETUP -------------------------------------------------------

st.set_page_config(__name__, page_icon=ICON_PATH, layout="wide")
st.sidebar.html(render_image(LOGO_PATH, 35))  # sidebar logo


BACKGROUND_LABEL = "Background Subtraction"
RANGE_LABEL = "Data Range"
INTERP_LABEL = "Interpolation"
DERIVE_LABEL = "Derivative"
SMOOTHING_LABEL = "Smoothing"
NORM_LABEL = "Normalisation"
FITTING_LABEL = "Fitting"
EXTRACTION_LABEL = "Data Extraction"


# Change the default style
@st.cache_resource
def set_style() -> None:
    """Set the default style"""
    with open(CSS_STYLE_PATH) as ofile:
        st.html(f"<style>{ofile.read()}</style>")


set_style()

# ---------------------------------------------------- SESSION STATE ---------------------------------------------------


ss = st.session_state


ss.settings_defaults = {
    "bckg_interacted": False,
    "bckg_range_input1": "",
    "bckg_range_input2": "",
    "range_interacted": False,
    "range_input1": "",
    "range_input2": "",
    "smoothing_interacted": False,
    "sg_fw": 0,
    "sg_po": 0,
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


def set_default_setting(
    key: str,
    reset: bool = False,
) -> None:
    """Store the specific setting in the session state using their default value
    :param key: setting key
    :param reset: if True, reset the stored value"""

    if key not in ss or reset:
        ss[key] = ss.settings_defaults[key]


# Add the settings to the session state using their default value
def set_default_settings(reset: bool = False) -> None:
    """Store the settings in the session state using their default value
    :param reset: if True, reset the stored value"""

    for key in ss.settings_defaults:
        set_default_setting(key, reset)


def refresh_session_state() -> None:
    """Refresh the session state. To be called when new widgets are displayed"""

    for key in ss.settings_defaults:
        ss[key] = ss[key]


def set_interact_value(
    key: str,
) -> callable:
    """Return a function that sets an interaction session state value to True and refresh the session state
    :param key: session state key"""

    def function() -> None:
        """Set the session state key value"""
        ss[key] = True
        refresh_session_state()

    return function


set_default_settings()


# ----------------------------------------------------- DATA INPUT -----------------------------------------------------


# File uploader
file = st.sidebar.file_uploader(
    label="File uploader",
    label_visibility="hidden",
    on_change=lambda: set_default_settings(True),
)

# File type
filetypes = ["Detect"] + sorted(FUNCTIONS.keys())
filetype_help = "Select the file type. If 'Detect' is selected, the file type will be automatically detected"
filetype = st.sidebar.selectbox(
    label="Data file type",
    options=filetypes,
    help=filetype_help,
    key="filetype_select",
)
filetype_message = st.sidebar.empty()

signal = None

# If no file is provided or no signal is stored
if not file:
    st.html(render_image(LOGO_TEXT_PATH, 30))  # main logo

else:

    # -------------------------------------------------- DATA LOADING --------------------------------------------------

    # Attempt to load the data by testing every file types
    if filetype == filetypes[0]:

        signal, extension = detect_file_type(file)
        filetype_message.markdown(f"Detected: {extension}")

    # Attempt to load the data using the provided function
    else:
        try:
            signal = FUNCTIONS[filetype][0](file)
        except Exception as e:
            print(e)
            pass

    # If the signal could be loaded, display a warning message
    if signal is None:
        st.warning("Unable to read that file")

    else:

        # ----------------------------------------------- DATA SELECTION -----------------------------------------------

        # Select the data type to plot if signal is a dictionary
        if isinstance(signal, dict):
            selection = st.sidebar.selectbox(
                label="Data type to plot",
                options=["All"] + list(signal.keys()),
                key="type_select",
            )
            if selection in signal:
                signal = signal[selection]

        # Select the signal in a list if multiple signals
        if isinstance(signal, (list, tuple)) and len(signal) > 1:
            names = [s.get_name(False) for s in signal]
            signal_dict = dict(zip(names, signal))
            col_selection = st.sidebar.selectbox(
                label="Data to plot",
                options=["All"] + sorted(signal_dict.keys()),
                key="data_select",
            )
            if col_selection in signal_dict:
                signal = signal_dict[col_selection]

        # ----------------------------------------------- DATA PROCESSING ----------------------------------------------

        # If only 1 signal is selected
        if isinstance(signal, SignalData):

            refresh_session_state()

            # ----------------------------------------------- BACKGROUND -----------------------------------------------

            try:
                xrange = [float(ss.bckg_range_input1), float(ss.bckg_range_input2)]
                signal = signal.remove_background(xrange)
                expander_label = f"__✔ {BACKGROUND_LABEL} {xrange[0]} - {xrange[1]}__"
            except:
                expander_label = BACKGROUND_LABEL

            with st.sidebar.expander(expander_label, ss["bckg_interacted"]):

                columns = st.columns(2)

                label1 = Dimension(0, "Lower Range", signal.x.unit).get_label_raw()
                help_str = f"Lower range of {signal.x.quantity} used to calculate the average background signal."
                columns[0].text_input(
                    label=label1,
                    key="bckg_range_input1",
                    help=help_str,
                    on_change=set_interact_value("bckg_interacted"),
                )

                label2 = Dimension(0, "Upper Range", signal.x.unit).get_label_raw()
                help_str = f"Upper range of {signal.x.quantity} used to calculate the average background signal."
                columns[1].text_input(
                    label=label2,
                    key="bckg_range_input2",
                    help=help_str,
                    on_change=set_interact_value("bckg_interacted"),
                )

            # ----------------------------------------------- DATA RANGE -----------------------------------------------

            try:
                xrange = [float(ss.range_input1), float(ss.range_input2)]
                signal = signal.reduce_range(xrange)
                expander_label = f"__✔ {RANGE_LABEL} {xrange[0]} - {xrange[1]}__"
            except:
                expander_label = RANGE_LABEL

            with st.sidebar.expander(expander_label, ss["range_interacted"]):

                range_cols = st.columns(2)

                label1 = Dimension(0, "Lower Range", signal.x.unit).get_label_raw()
                help_str = f"Lower Range of {signal.x.quantity} to display."
                range_cols[0].text_input(
                    label=label1,
                    key="range_input1",
                    help=help_str,
                    on_change=set_interact_value("range_interacted"),
                )

                label2 = Dimension(0, "Upper Range", signal.x.unit).get_label_raw()
                help_str = f"Upper Range of {signal.x.quantity} to display."
                range_cols[1].text_input(
                    label=label2,
                    key="range_input2",
                    help=help_str,
                    on_change=set_interact_value("range_interacted"),
                )

            # ---------------------------------------------- INTERPOLATION ---------------------------------------------

            expander_label = INTERP_LABEL
            dx = None
            try:
                if ss.interp_type == "Fixed Step":
                    dx = float(ss.interp_dx)
                    signal = signal.interpolate(dx=dx)
                elif ss.interp_type == "Point Count":
                    n_count = int(ss.interp_n)
                    signal = signal.interpolate(dx=n_count)
                    dx = np.mean(np.diff(signal.x.data))
                if dx:
                    dx_str = number_to_str(dx, 3)
                    expander_label = f"__✔ {INTERP_LABEL} (step = {dx_str} {signal.x.get_unit_label_html()})__"
            except:
                pass

            with st.sidebar.expander(expander_label, ss["interp_interacted"]):

                help_str = """Interpolate the data using different methods:
* __Fixed Step__ interpolation – Data are interpolated using a specified step size.
* __Point Count__ interpolation – Data are interpolated to fit a specified number of points."""

                st.radio(
                    label="Interpolation Type",
                    options=["None", "Fixed Step", "Point Count"],
                    key="interp_type",
                    horizontal=True,
                    on_change=set_interact_value("interp_interacted"),
                    help=help_str,
                )

                if ss["interp_type"] != "None":
                    st.text_input(
                        label="f",
                        key={"Fixed Step": "interp_dx", "Point Count": "interp_n"}[ss.interp_type],
                        label_visibility="collapsed",
                        on_change=set_interact_value("interp_interacted"),
                    )

            # ----------------------------------------------- DERIVATION -----------------------------------------------

            expander_label = DERIVE_LABEL
            try:
                if ss.derive_order > 0:
                    signal = signal.derive(n=ss.derive_order)
                    expander_label = f"__✔ {DERIVE_LABEL} ({ss.derive_order} order)__"
            except:
                pass

            with st.sidebar.expander(expander_label, ss["derive_interacted"]):

                st.number_input(
                    label="Derivative Order",
                    min_value=0,
                    key="derive_order",
                    help="Calculate the n-th order derivative.",
                    on_change=set_interact_value("derive_interacted"),
                )

            # ------------------------------------------------ SMOOTHING -----------------------------------------------

            signal_s = None
            expander_label = SMOOTHING_LABEL
            try:
                if ss.sg_fw > 0 and ss.sg_po > 0:
                    signal_s = signal.smooth(ss.sg_fw, ss.sg_po)
                    expander_label = f"__✔ {SMOOTHING_LABEL} ({ss.sg_fw}, {ss.sg_po})__"
            except:
                pass

            with st.sidebar.expander(expander_label, ss["smoothing_interacted"]):

                smoothing_cols = st.columns(2)

                smoothing_cols[0].number_input(
                    label="Filter Length",
                    min_value=0,
                    help="Length of the Savitzky-Golay filter window",
                    key="sg_fw",
                    on_change=set_interact_value("smoothing_interacted"),
                )

                smoothing_cols[1].number_input(
                    label="Polynomial Order",
                    min_value=0,
                    help="Order of the polynomial used to fit the samples",
                    key="sg_po",
                    on_change=set_interact_value("smoothing_interacted"),
                )

            # ---------------------------------------------- NORMALISATION ---------------------------------------------

            expander_label = NORM_LABEL
            try:
                if ss["norm_type"] == "Max. Normalisation":
                    if signal_s:
                        signal = signal.normalise(other=signal_s.y.data)
                        signal_s = signal_s.normalise()
                    else:
                        signal = signal.normalise()
                    expander_label = f"__✔ {NORM_LABEL} ({ss['norm_type']})__"
                elif ss["norm_type"] == "Feature Scaling":
                    kwargs = dict(a=float(ss["norm_a"]), b=float(ss["norm_b"]))
                    if signal_s:
                        signal = signal.feature_scale(other=signal_s.y.data, **kwargs)
                        signal_s = signal_s.feature_scale(**kwargs)
                    else:
                        signal = signal.feature_scale(**kwargs)
                    expander_label = f"__✔ {NORM_LABEL} ({ss['norm_type']} {kwargs['a']} - {kwargs['b']})__"
            except:
                pass

            with st.sidebar.expander(expander_label, ss["norm_interacted"]):

                help_str = """Data can be normalised using different methods:
* __Max. normalisation__ – the y-values are normalised with respect to their maximum value.
* __Feature scaling__ – the y-values are normalised based on specified minimum and maximum values."""

                st.radio(
                    label="Normalisation Type",
                    options=["None", "Max. Normalisation", "Feature Scaling"],
                    key="norm_type",
                    horizontal=True,
                    on_change=set_interact_value("norm_interacted"),
                    help=help_str,
                )

                if ss["norm_type"] == "Feature Scaling":

                    columns = st.columns(2)

                    columns[0].text_input(
                        label="Maximum Value",
                        key="norm_a",
                        on_change=set_interact_value("norm_interacted"),
                    )

                    columns[1].text_input(
                        label="Minimum Value",
                        key="norm_b",
                        on_change=set_interact_value("norm_interacted"),
                    )

            # Plot the signal and store it as the raw signal
            figure = plot(signal)
            raw_signal = signal

            # If the smoothed signal exist, plot it and replace the studied signal with it
            if signal_s:
                signal_s.plot(figure)
                signal = signal_s

            # ------------------------------------------------- FITTING ------------------------------------------------

            fit_signal, fit_params, param_errors, r_squared, equation, parameters = None, None, None, None, "", []
            expander_label = FITTING_LABEL
            try:

                # Get the fit function, equation and guess function, and the function parameters
                fit_function, equation, guess_function = MODELS[ss.fitting_model]
                parameters = get_model_parameters(fit_function)

                # Reset the guess value default dict is set to None
                if ss.guess_values is None:
                    try:
                        guess_values = guess_function(signal.x.data, signal.y.data)
                    except:
                        guess_values = np.ones(len(parameters))
                    ss.guess_values = dict(zip(parameters, guess_values))

                fit_signal, fit_params, param_errors, r_squared = signal.fit(fit_function, ss.guess_values)
                expander_label = f"__✔ {FITTING_LABEL} ({ss.fitting_model})__"
            except:
                pass

            with st.sidebar.expander(expander_label, ss["fitting_interacted"]):

                def on_change() -> None:
                    """Set fitting_interacted to True and reset guess_values"""

                    set_interact_value("fitting_interacted")
                    ss.guess_values = None

                st.selectbox(
                    label="Model",
                    options=["None"] + list(MODELS.keys()),
                    on_change=set_interact_value("fitting_interacted"),
                    key="fitting_model",
                    help="Use the selected model to fit the data.",
                )

                if ss.fitting_model in MODELS:

                    # Display the equation
                    st.html("Equation: " + equation)

                    # Guess parameter and value
                    columns = st.columns(2)

                    parameter = columns[0].selectbox(
                        label="Parameter",
                        options=parameters,
                        key="parameter_model_key",
                        on_change=set_interact_value("fitting_interacted"),
                    )

                    guess_value_key = ss.fitting_model + parameter + "guess_value"

                    if guess_value_key not in ss:
                        ss[guess_value_key] = number_to_str(ss.guess_values[parameter], 4, False)

                    def store_guess_value() -> None:
                        """Store the guess value as a float in the guess_values dictionary"""

                        set_interact_value("fitting_interacted")
                        try:
                            ss.guess_values[parameter] = float(ss[guess_value_key])
                        except:
                            pass

                    columns[1].text_input(
                        label="Guess Value",
                        key=guess_value_key,
                        on_change=store_guess_value,
                    )

            # Plot the fit if it exists and replace the studied signal with it
            if fit_signal is not None:
                fit_signal.plot(figure)
                signal = fit_signal

            # ----------------------------------------------- DATA EXPORT ----------------------------------------------

            header = [raw_signal.x.get_label_raw(), raw_signal.y.get_label_raw()]
            data = [raw_signal.x.data, raw_signal.y.data]
            if signal_s:
                header += [raw_signal.y.get_label_raw() + " (smoothed)"]
                data += [signal_s.y.data]
            if fit_signal:
                header += [raw_signal.y.get_label_raw() + " (fit)"]
                data += [fit_signal.y.data]

            export_data = matrix_to_string(data, header)
            st.sidebar.download_button(
                label="Download Data",
                data=export_data,
                file_name="data.csv",
                use_container_width=True,
                key="download_button",
            )

            # --------------------------------------------- DATA EXTRACTION --------------------------------------------

            st.sidebar.markdown(f"#### {EXTRACTION_LABEL}")

            # Max point
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

            # Min point
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

            # FWHM
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

            buttons = (fit_params, max_button, min_button, fwhm_button)
            n_buttons = len([e for e in buttons if e is not None and e is not False])

            if n_buttons:

                # Plot the signal
                main_columns = st.columns([3.5, 1])
                plot_spot = main_columns[0].container()
                info_spot = main_columns[1].container()

                column_index = 0

                info_spot.markdown("#### About your data")

                # Fit
                if fit_params is not None:
                    info_spot.markdown("##### Fitting Results")

                    # Add the R2 to the list of parameters and errors
                    fit_params = list(fit_params) + [r_squared]
                    param_errors = list(param_errors) + [0]
                    parameters += ["R<sup>2</sup>"]

                    # Convert the parameters and errors to strings
                    fit_params_str = [number_to_str(f, 4, True) for f in fit_params]
                    rel_error = np.array(param_errors) / np.array(fit_params)
                    rel_error_str = [f"{f * 100:.2f}" for f in rel_error]

                    # Generate the dataframe and display it
                    df = pd.DataFrame(
                        [fit_params_str, rel_error_str],
                        columns=parameters,
                        index=["Value", "Relative Error (%)"],
                    )
                    df = df.transpose()
                    df.columns.name = "Parameter"
                    info_spot.html("Equation: " + equation)
                    info_spot.html(generate_html_table(df))

                # Max point
                if max_button:
                    info_spot.markdown("##### Maximum")
                    x_max, y_max, i_max = signal.get_max(max_interp_button)
                    info_spot.html(x_max.get_value_label_html())
                    info_spot.html(y_max.get_value_label_html())

                    # Display the maximum point
                    scatter_plot(
                        figure=figure,
                        x_data=x_max.data,
                        y_data=y_max.data,
                        label="Max. Point",
                    )

                # Min point
                if min_button:
                    info_spot.markdown("##### Minimum")
                    x_min, y_min, i_min = signal.get_min(min_interp_button)
                    info_spot.html(x_min.get_value_label_html())
                    info_spot.html(y_min.get_value_label_html())

                    # Display the minimum point
                    scatter_plot(
                        figure=figure,
                        x_data=x_min.data,
                        y_data=y_min.data,
                        label="Min. Point",
                    )

                # FWHM
                if fwhm_button:
                    info_spot.markdown("##### FWHM")
                    fwhm, x_left, y_left, x_right, y_right = signal.get_fwhm(fwhm_button_interp)

                    # Display the FWHM
                    if not np.isnan(fwhm.data):
                        scatter_plot(
                            figure=figure,
                            x_data=[x_left.data, x_right.data],
                            y_data=[y_left.data, y_right.data],
                            label="FWHM",
                        )
                        info_spot.html(fwhm.get_value_label_html())
                    else:
                        info_spot.markdown("Could not calculate the FWHM.")

            else:

                plot_spot = st.container()

        # If multiple signals are selected
        else:
            plot_spot = st.container()
            figure = plot(signal)

        plot_spot.plotly_chart(figure, use_container_width=True)


# ----------------------------------------------------- INFORMATION ----------------------------------------------------


with st.expander("About", expanded=file is None):
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
    st.markdown(text)

    st.markdown(f"""##### {DERIVE_LABEL}""")
    text = """The n order derivative can be calculated by setting the __Derivative Order__."""
    st.markdown(text)

    st.markdown(f"""##### {SMOOTHING_LABEL}""")
    text = """Applies a Savitzky–Golay smoothing filter using the specified __Filter Length__ and __Polynomial Order__ 
    (__subplot a__)."""
    st.markdown(text)

    st.markdown(f"##### {NORM_LABEL}")
    text = """Data can be normalised using different methods:
* __Max. normalisation__ – the y-values are normalised with respect to their maximum value.
* __Feature scaling__ – the y-values are normalised based on specified minimum and maximum values."""
    st.markdown(text)

    st.markdown(f"##### {FITTING_LABEL}")
    models = "".join(["\n* " + key for key in MODELS.keys()])
    text = f"""Data can be fitted using the following __models__: {models}\n
Initial guess values are automatically estimated from the data but can be manually adjusted."""
    st.markdown(text)

    st.markdown(f"##### {EXTRACTION_LABEL}")
    text = """The following information can be extracted from the processed data:
* __Maximum point__: The x and y coordinates corresponding to the peak y-value in the data (__subplot c__).
* __Minimum point__: The x and y coordinates corresponding to the lowest y-value in the data (__subplot c__).

For both maximum and minimum points, the precision of the estimated positions can be enhanced using cubic __interpolation__ 
around the peak maximum, which refines the estimate beyond the raw data resolution.
* __Full Width at Half Maximum (FWHM)__: The width of the peak measured between the two x-values where the y-value equals 
half of the maximum y-value. This provides a useful metric of the peak’s sharpness or spread (__subplot b__).

Similar to the max/min points, the precision of the FWHM measurement can be further refined using linear __interpolation__. """
    st.markdown(text)
    st.html(render_image(DATA_PROCESSING_PATH, 50))

# ------------------------------------------------------ CHANGELOG -----------------------------------------------------

with st.expander("Changelog"):
    st.markdown(read_file(os.path.join(project_path, "CHANGELOG.md")))

# ----------------------------------------------------- DISCLAIMER -----------------------------------------------------

with st.expander("License & Disclaimer"):
    st.markdown(read_file(os.path.join(project_path, "LICENSE.txt")))
