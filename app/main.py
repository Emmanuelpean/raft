"""Graphical user interface of Raft"""

import os

import numpy as np
import pandas as pd
import streamlit as st

from data_files import FUNCTIONS, detect_file_type
from fitting import MODELS, get_model_parameters
from plot import plot, scatter_plot
from resources import LOGO_FILENAME, LOGO_TEXT_FILENAME, CSS_STYLE_PATH, ICON_FILENAME
from signaldata import SignalData, Dimension
from utils import render_image, matrix_to_string, read_txt_file, number_to_str, generate_html_table

__version__ = "2.0.0"
__date__ = "March 2025"

dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------------------------------------------------------- SETUP -------------------------------------------------------

st.set_page_config("Raft", page_icon=ICON_FILENAME, layout="wide")
st.sidebar.html(render_image(LOGO_FILENAME, 35))  # sidebar logo


# Change the default style
@st.cache_resource
def set_style() -> None:
    """Set the default style"""
    with open(CSS_STYLE_PATH) as ofile:
        st.html(f"<style>{ofile.read()}</style>")


set_style()

# ---------------------------------------------------- SESSION STATE ---------------------------------------------------

# Session state
ss = st.session_state


def reset_stored_guess_values(reset: bool = False) -> None:
    """Reset the stored guess values
    :param reset: if True, reset the stored value"""

    if "guess_values" not in ss or reset:
        ss.guess_values = None


reset_stored_guess_values()


ss.settings_defaults = {
    "background_label": "",
    "bckg_range_input1": "",
    "bckg_range_input2": "",
    "range_label": "",
    "range_input1": "",
    "range_input2": "",
    "smoothing_label": "",
    "sg_fw": 0,
    "sg_po": 0,
    "fitting_label": "",
    "fitting_model": "None",
    "norm_label": "",
    "norm_type": "None",
    "norm_a": "1",
    "norm_b": "0",
}


# Add the settings to the session state using their default value
def set_settings(reset: bool = False) -> None:
    """Store the settings in the session state using their default value
    :param reset: if True, reset the stored value"""

    for key in ss.settings_defaults:
        if key not in ss or reset:
            ss[key] = ss.settings_defaults[key]
    reset_stored_guess_values(reset)


set_settings()

# ----------------------------------------------------- DATA INPUT -----------------------------------------------------

# File uploader
file = st.sidebar.file_uploader(label="File uploader", label_visibility="hidden", on_change=lambda: set_settings(True))

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
    st.html(render_image(LOGO_TEXT_FILENAME, 30))  # main logo
    st.info(
        f"""RAFT is a free tool to plot the content a various data files. Just drag and drop your file and get 
        the relevant information from it!  
        App created and maintained by [Emmanuel V. Péan](https://emmanuelpean.streamlit.app/).  
        [Version {__version__}](https://github.com/Emmanuelpean/raft) (last updated: {__date__})."""
    )

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
        except:
            pass

    # If the signal could be loaded, display a warning message
    if signal is None:
        st.warning("Unable to read that file")

    # else display the data
    else:

        # Plot the signal
        main_columns = st.columns([4, 1])
        plot_spot = main_columns[0].empty()
        info_spot = main_columns[1].container()

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

        # If only 1 signal is selected
        if isinstance(signal, SignalData):

            # ----------------------------------------------- DATA EXPORT ----------------------------------------------

            header = [signal.x.get_label_raw(), signal.y.get_label_raw()]
            export_data = matrix_to_string([signal.x.data, signal.y.data], header)
            st.sidebar.download_button(
                "Download data",
                export_data,
                "data.csv",
                use_container_width=True,
                key="download_button",
            )

            # ----------------------------------------------- BACKGROUND -----------------------------------------------

            def get_expander_status(ss_label: str, label: str) -> bool:
                """Get the expander status based on the session state label
                :param ss_label: session state label key
                :param label: current label"""

                status = ss[ss_label] != label  # True (opened) if label has changed
                if ss[ss_label] == ss.settings_defaults[ss_label]:
                    status = False
                ss[ss_label] = label
                return status

            EXPANDER_LABEL = "Background Removal"
            try:
                xrange = [float(ss.bckg_range_input1), float(ss.bckg_range_input2)]
                signal = signal.remove_background(xrange)
                expander_label = f"__✔ {EXPANDER_LABEL} {xrange[0]} - {xrange[1]}__"
            except:
                expander_label = EXPANDER_LABEL
            expander_status = get_expander_status("background_label", expander_label)

            with st.sidebar.expander(expander_label, expanded=expander_status):

                columns = st.columns(2)

                label1 = Dimension(0, "Lower Range", signal.x.unit).get_label_raw()
                help_str = f"Lower range of {signal.x.quantity} used to calculate the average background signal."
                columns[0].text_input(
                    label=label1,
                    key="bckg_range_input1",
                    help=help_str,
                )

                label2 = Dimension(0, "Upper Range", signal.x.unit).get_label_raw()
                help_str = f"Upper range of {signal.x.quantity} used to calculate the average background signal."
                columns[1].text_input(
                    label=label2,
                    key="bckg_range_input2",
                    help=help_str,
                )

            # ----------------------------------------------- DATA RANGE -----------------------------------------------

            RANGE_LABEL = "Data Range"
            try:
                xrange = [float(ss.range_input1), float(ss.range_input2)]
                signal = signal.reduce_range(xrange)
                expander_label = f"__✔ {RANGE_LABEL} {xrange[0]} - {xrange[1]}__"
            except:
                expander_label = RANGE_LABEL
            expander_status = get_expander_status("range_label", expander_label)

            with st.sidebar.expander(expander_label, expanded=expander_status):

                range_cols = st.columns(2)

                label1 = Dimension(0, "Lower Range", signal.x.unit).get_label_raw()
                help_str = f"Lower Range of {signal.x.quantity} to display."
                range_cols[0].text_input(
                    label=label1,
                    key="range_input1",
                    help=help_str,
                )

                label2 = Dimension(0, "Upper Range", signal.x.unit).get_label_raw()
                help_str = f"Upper Range of {signal.x.quantity} to display."
                range_cols[1].text_input(
                    label=label2,
                    key="range_input2",
                )

            # ---------------------------------------------- NORMALISATION ---------------------------------------------

            NORM_LABEL = "Normalisation"
            expander_label = NORM_LABEL
            if ss["norm_type"] != "None":
                try:
                    if ss["norm_type"] == "Max Normalisation":
                        signal = signal.normalise()
                    elif ss["norm_type"] == "Feature Scaling":
                        signal = signal.feature_scale(float(ss["norm_a"]), float(ss["norm_b"]))
                        print(signal.x)
                        print(signal.y)

                    expander_label = f"__✔ {NORM_LABEL} ({ss['norm_type']})__"
                except:
                    pass

            expander_status = get_expander_status("norm_label", expander_label)

            with st.sidebar.expander(expander_label, expanded=expander_status):

                st.radio(
                    label="f",
                    options=["None", "Max Normalisation", "Feature Scaling"],
                    key="norm_type",
                    horizontal=True,
                    label_visibility="collapsed",
                )

                # Refresh the session state
                ss["norm_a"] = ss["norm_a"]
                ss["norm_b"] = ss["norm_b"]

                if ss["norm_type"] == "Feature Scaling":
                    columns = st.columns(2)
                    columns[0].text_input(
                        label="Maximum Value",
                        key="norm_a",
                    )
                    columns[1].text_input(
                        label="Minimum Value",
                        key="norm_b",
                    )

            # Plot the signal
            figure = plot(signal)

            # ------------------------------------------------ SMOOTHING -----------------------------------------------

            SMOOTHING_LABEL = "Smoothing"
            if ss.sg_fw > 0 and ss.sg_po > 0:
                signals_s = signal.smooth(ss.sg_fw, ss.sg_po)
                signals_s.plot(figure)
                signal = signals_s
                expander_label = f"__✔ {SMOOTHING_LABEL} ({ss.sg_fw}, {ss.sg_po})__"
            else:
                expander_label = SMOOTHING_LABEL
            expander_status = get_expander_status("smoothing_label", expander_label)

            with st.sidebar.expander(expander_label, expanded=expander_status):

                smoothing_cols = st.columns(2)
                smoothing_cols[0].number_input(
                    label="Filter Length",
                    min_value=0,
                    help="Length of the Savitzky-Golay filter window",
                    key="sg_fw",
                )

                smoothing_cols[1].number_input(
                    label="Polynomial Order",
                    min_value=0,
                    help="Order of the polynomial used to fit the samples",
                    key="sg_po",
                )

            # ------------------------------------------------- FITTING ------------------------------------------------

            FITTING_LABEL = "Fitting"
            fit_signal, fit_params, param_errors, r_squared = None, None, None, None
            if ss.fitting_model in MODELS:

                # Get the fit function, equation and guess function, and the function parameters
                fit_function, equation, guess_function = MODELS[ss.fitting_model]
                parameters = get_model_parameters(fit_function)

                # Determine the default guess values
                guess_values = guess_function(signal.x.data, signal.y.data)
                guess_values = dict(zip(parameters, guess_values))

                # Reset the guess value default dict is set to None
                if ss.guess_values is None:
                    ss.guess_values = guess_values

                fit_function = MODELS[ss.fitting_model][0]
                try:
                    fit_signal, fit_params, param_errors, r_squared = signal.fit(fit_function, ss.guess_values)
                    expander_label = f"__✔ {FITTING_LABEL} ({ss.fitting_model})__"
                except:
                    pass
            else:
                expander_label = FITTING_LABEL
            expander_status = get_expander_status("fitting_label", expander_label)

            with st.sidebar.expander(expander_label, expanded=expander_status):

                st.selectbox(
                    label="Model",
                    options=["None"] + list(MODELS.keys()),
                    on_change=lambda: reset_stored_guess_values(True),
                    key="fitting_model",
                    help="Use the selected model to fit the data.",
                )

                if ss.fitting_model in MODELS:

                    # Display the equation
                    # noinspection PyUnboundLocalVariable
                    st.html("Equation: " + equation)

                    # Guess parameter and value
                    columns = st.columns(2)

                    # noinspection PyUnboundLocalVariable
                    parameter = columns[0].selectbox(
                        label="Parameter",
                        options=parameters,
                    )

                    ss_key = ss.fitting_model + parameter + "guess_value"
                    if ss_key not in ss:
                        ss[ss_key] = number_to_str(ss.guess_values[parameter], 2)

                    def store_guess_value() -> None:
                        """Store the guess value as a float in the guess_values dictionary"""

                        try:
                            ss.guess_values[parameter] = float(ss[ss_key])
                        except:
                            pass

                    columns[1].text_input(
                        label="Guess Value",
                        key=ss_key,
                        on_change=lambda: store_guess_value(),
                    )

            if fit_signal is not None:
                fit_signal.plot(figure)
                signal = fit_signal

            # --------------------------------------------- DATA EXTRACTION --------------------------------------------

            st.sidebar.markdown(f"#### Data Extraction")

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
                column_index = 0

                info_spot.markdown("### About your data")

                # Fit
                if fit_params is not None:
                    info_spot.markdown("##### Fitting Results")

                    # Add the R2 to the list of parameters and errors
                    fit_params = list(fit_params) + [r_squared]
                    param_errors = list(param_errors) + [0]
                    parameters += ["R<sup>2</sup>"]

                    # Convert the parameters and errors to strings
                    fit_params_str = [number_to_str(f, 2) for f in fit_params]
                    rel_error = np.array(param_errors) / np.array(fit_params)
                    rel_error_str = [number_to_str(f * 100) for f in rel_error]

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
                        figure,
                        x_max.data,
                        y_max.data,
                        "Max. Point",
                    )

                # Min point
                if min_button:
                    info_spot.markdown("##### Minimum")
                    x_min, y_min, i_min = signal.get_min(min_interp_button)
                    info_spot.html(x_min.get_value_label_html())
                    info_spot.html(y_min.get_value_label_html())

                    # Display the minimum point
                    scatter_plot(
                        figure,
                        x_min.data,
                        y_min.data,
                        "Min. Point",
                    )

                # FWHM
                if fwhm_button:
                    info_spot.markdown("##### FWHM")
                    fwhm, x_left, y_left, x_right, y_right = signal.get_fwhm(fwhm_button_interp)

                    # Display the FWHM
                    if not np.isnan(fwhm.data):
                        scatter_plot(
                            figure,
                            [x_left.data, x_right.data],
                            [y_left.data, y_right.data],
                            "FWHM",
                        )
                        info_spot.html(fwhm.get_value_label_html())
                    else:
                        info_spot.markdown("Could not calculate the FWHM.")

        # If multiple signals are selected
        else:
            figure = plot(signal)

        plot_spot.plotly_chart(figure, use_container_width=True)

# ------------------------------------------------------ CHANGELOG -----------------------------------------------------

with st.expander("Changelog"):
    st.markdown(read_txt_file(os.path.join(dirname, "CHANGELOG.md")))

# ----------------------------------------------------- DISCLAIMER -----------------------------------------------------

with st.expander("License & Disclaimer"):
    st.markdown(read_txt_file(os.path.join(dirname, "LICENSE.txt")))
