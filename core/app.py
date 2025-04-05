"""Graphical user interface of Raft"""

import os

import plotly.graph_objects as go
import streamlit as st

from data_files import functions, detect_file_type
from plot import plot
from resource import LOGO_FILENAME, LOGO_TEXT_FILENAME, CSS_STYLE_PATH
from signal_data import SignalData, Dimension
from utils import render_image, matrix_to_string, read_txt_file

dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------------------------------------------------------- SETUP -------------------------------------------------------

st.set_page_config("Raft", layout="wide")
st.sidebar.markdown(render_image(LOGO_FILENAME, 30), unsafe_allow_html=True)  # sidebar logo


# Change the default style
@st.cache_resource
def set_style() -> None:
    """Set the default style"""
    with open(CSS_STYLE_PATH) as ofile:
        st.markdown(f"<style>{ofile.read()}</style>", unsafe_allow_html=True)


set_style()

# ----------------------------------------------------- DATA INPUT -----------------------------------------------------

# File uploader
file = st.sidebar.file_uploader("File uploader", label_visibility="hidden")

# File type
filetypes = ["Detect"] + sorted(functions.keys())
filetype_help = "Select the file type. If 'Detect' is selected, the file type will be automatically detected"
filetype = st.sidebar.selectbox(
    label="Data file type",
    options=filetypes,
    help=filetype_help,
)
filetype_message = st.sidebar.empty()

signal = None

# If no file is provided or no signal is stored
if not file:
    st.markdown(render_image(LOGO_TEXT_FILENAME, 25), unsafe_allow_html=True)  # main logo
    st.info(
        "RAFT is a free tool to plot the content a various data files. Just drag and drop your file and get the relevant information from it!"
    )

else:

    # -------------------------------------------------- DATA LOADING --------------------------------------------------
    print("loading file")

    # Attempt to load the data by testing every file types
    if filetype == "Detect":

        signal, extension = detect_file_type(file)  # TODO bug here
        filetype_message.markdown("Detected: %s" % extension)

    # Attempt to load the data using the provided function
    else:
        try:
            signal = functions[filetype][0](file)
        except:
            pass

    # If the signal could be loaded, display an error message, else store it in the session state
    if signal is None:
        st.warning("Unable to read that file")

    # If signals are stored in the session state
    else:

        plot_spot = st.empty()

        # ----------------------------------------------- DATA SELECTION -----------------------------------------------

        # Select the data type to plot if signal is a dictionary
        if isinstance(signal, dict):
            selection = st.sidebar.selectbox("Data type to plot", ["All"] + list(signal.keys()))
            if selection in signal:
                signal = signal[selection]

        # Select the signal in a list if multiple signals
        if isinstance(signal, (list, tuple)) and len(signal) > 1:
            names = [s.get_name(False) for s in signal]
            signal_dict = dict(zip(names, signal))
            col_selection = st.sidebar.selectbox("Data to plot", ["All"] + sorted(signal_dict.keys()))
            if col_selection in signal_dict:
                signal = signal_dict[col_selection]

        # If only 1 signal is selected
        if isinstance(signal, SignalData):

            # ----------------------------------------------- DATA EXPORT ----------------------------------------------

            header = [signal.x.get_label_raw(), signal.y.get_label_raw()]
            export_data = matrix_to_string([signal.x.data, signal.y.data], header)
            st.sidebar.download_button("Download data", export_data, "data.csv", use_container_width=True)

            # ----------------------------------------------- DATA RANGE -----------------------------------------------

            min_label = Dimension(0, "min. " + signal.x.quantity, signal.x.unit).get_label_html()
            max_label = Dimension(0, "max. " + signal.x.quantity, signal.x.unit).get_label_html()
            range_cols = st.sidebar.columns(2)
            range_button1 = range_cols[0].text_input(label=min_label, value=signal.x.data[0])
            range_button2 = range_cols[1].text_input(label=max_label, value=signal.x.data[-1])
            try:
                signal = signal.reduce_range([float(range_button1), float(range_button2)])
            except ValueError:
                pass

            # Plot the signal
            figure = plot(signal)

            # ------------------------------------------------ SMOOTHING -----------------------------------------------

            smoothing_cols = st.sidebar.columns(2)
            smoothing_fw = smoothing_cols[0].number_input(
                label="Smoothing\n\n (Filter Length)",
                value=0,
                help="Length of the Savitzky-Golay filter window",
            )
            smoothing_po = smoothing_cols[1].number_input(
                label="Smoothing\n\n (Polynomial Order)",
                value=0,
                help="Order of the polynomial used to fit the samples",
            )

            if smoothing_fw > 0 and smoothing_po > 0:
                signals_s = signal.smooth(smoothing_fw, smoothing_po)
                signals_s.plot(figure)
                signal = signals_s

            # --------------------------------------------- DATA EXTRACTION --------------------------------------------

            max_button = st.sidebar.checkbox(label="Display maximum")
            min_button = st.sidebar.checkbox(label="Display minimum")
            fwhm_cols = st.sidebar.columns(2)
            fwhm_button = fwhm_cols[0].checkbox(label="Display FWHM")
            fwhm_button_interp = fwhm_cols[1].checkbox(
                "use interpolation",
                key="fwhm_button_interp_",
                help="""Use interpolation to improve the calculation of the FWHM""",
            )
            buttons = (max_button, min_button, fwhm_button)

            if sum(buttons) > 0:
                st.markdown("### About your data")
                columns = st.columns(sum(buttons))

                # Max point
                if max_button:
                    col = columns[0]
                    col.markdown("##### Maximum")
                    x_max, y_max, i_max = signal.get_max()
                    col.markdown(x_max.get_value_label_html())
                    col.markdown(y_max.get_value_label_html())

                    # Display the maximum point
                    trace = go.Scatter(
                        x=[x_max.data],
                        y=[y_max.data],
                        mode="markers",
                        name="Max. point",
                    )
                    figure.add_trace(trace)

                # Min point
                if min_button:
                    col = columns[sum(buttons[:1])]
                    col.markdown("##### Minimum")
                    x_min, y_min, i_min = signal.get_min()
                    col.markdown(x_min.get_value_label_html())
                    col.markdown(y_min.get_value_label_html())

                    # Display the minimum point
                    trace = go.Scatter(
                        x=[x_min.data],
                        y=[y_min.data],
                        mode="markers",
                        name="Min. point",
                    )
                    figure.add_trace(trace)

                # FWHM
                if fwhm_button:
                    col = columns[sum(buttons[:2])]
                    col.markdown("##### FWHM")
                    fwhm, x_left, y_left, x_right, y_right = signal.get_fwhm(fwhm_button_interp)
                    col.markdown(fwhm.get_value_label_html())

                    # Display the FWHM
                    trace = go.Scatter(
                        x=[x_left.data, x_right.data],
                        y=[y_left.data, y_right.data],
                        name="FWHM",
                    )
                    figure.add_trace(trace)

                figure.update_traces(marker_size=15)

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
