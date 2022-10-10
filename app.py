import streamlit as st
import core.data_files as pdf
import core.plot as pp
from core import utils
from core import resources
from core.signal import SignalData
import streamlit.components.v1 as components
import os
import plotly.graph_objects as go

# -------------------------------------------------------- SETUP -------------------------------------------------------

st.set_page_config('Raft', layout='wide')
st.markdown("""<style> #MainMenu {visibility: hidden;} footer {visibility: hidden;} </style>""", unsafe_allow_html=True)  # hide main menu and footer
st.markdown("""%s""" % utils.render_image(resources.logo_text_filename, 25), unsafe_allow_html=True)  # main logo
st.sidebar.markdown("""%s""" % utils.render_image(resources.logo_filename, 30), unsafe_allow_html=True)  # sidebar logo
if 'filetype' not in st.session_state:
    st.session_state.filetype = 'Detect'

# ----------------------------------------------------- DATA INPUT -----------------------------------------------------

# File uploader
filename = st.sidebar.file_uploader('a', on_change=lambda: setattr(st.session_state, 'filetype', 'Detect'), label_visibility='hidden')

# File type
filetypes = ['Detect'] + sorted(pdf.functions.keys())
filetype_sb = st.sidebar.empty()
filetype = filetype_sb.selectbox('Data file', filetypes, index=filetypes.index(st.session_state.filetype),
                                 help="Select the file type. If 'Detect' is selected, the file type will be automatically detected")

signal = None

if not filename:
    st.info('Input a file')

else:

    # -------------------------------------------------- DATA LOADING --------------------------------------------------

    # Attempt to load the data by testing every file types
    if filetype == 'Detect':

        # Only search through the filetypes matching the extension
        extension = os.path.splitext(filename.name)[1]
        functions = {key: value for key, value in pdf.functions.items() if
                     extension.lower() == '.' + pdf.extensions[key].lower() or pdf.extensions[key] == ''}

        for key in functions:
            # noinspection PyBroadException
            try:
                signal = pdf.functions[key](filename)
                filetype = filetype_sb.selectbox('Data file', filetypes, index=filetypes.index(key))
                st.session_state.filetype = key
                break
            except:
                pass

    # Attempt to load the data using the provided function
    else:
        # noinspection PyBroadException
        try:
            signal = pdf.functions[filetype](filename)
        except:
            pass

    if signal:

        plot_spot = st.empty()

        # ----------------------------------------------- DATA SELECTION -----------------------------------------------

        # Select the data type to plot if signal is a dictionary
        if isinstance(signal, dict):
            selection = st.sidebar.selectbox('Data to plot', ['All'] + list(signal.keys()))
            if selection != 'All':
                signal = signal[selection]

        # Convert the list of signals to a signal if only 1 signal in the list
        if isinstance(signal, (list, tuple)) and len(signal) == 1:
            signal = signal[0]

        # Select the signal in a list
        if isinstance(signal, (list, tuple)):
            col_selection = st.sidebar.selectbox('Column to plot', ['All'] + list(range(1, len(signal) + 1)))
            if col_selection != 'All':
                signal = signal[col_selection - 1]

        # If only 1 signal is selected
        if isinstance(signal, SignalData):

            # ----------------------------------------------- DATA EXPORT ----------------------------------------------

            header = [signal.x.get_label(), signal.y.get_label()]
            export_data = utils.matrix_to_string([signal.x.data, signal.y.data], header)
            st.download_button('Download data', export_data, 'pears_fit_data.csv')

            # ----------------------------------------------- DATA RANGE -----------------------------------------------

            st.sidebar.markdown('Data range (%s)' % signal.x.unit)
            range_cols = st.sidebar.columns(2)
            range_button1 = range_cols[0].number_input('', min_value=signal.x.data[0], max_value=signal.x.data[-2], value=signal.x.data[0], label_visibility='collapsed')
            range_button2 = range_cols[1].number_input('', min_value=signal.x.data[1], max_value=signal.x.data[-1], value=signal.x.data[-1], label_visibility='collapsed')
            signal = signal.reduce_range([range_button1, range_button2])

            figure = pp.plot(signal)  # plot the signal

            # ------------------------------------------------ SMOOTHING -----------------------------------------------

            st.sidebar.markdown('Smoothing')
            smoothing_cols = st.sidebar.columns(2)
            smoothing_entry1 = smoothing_cols[0].number_input('Filter length', value=0, help="Length of the Savitzky-Golay filter window")
            smoothing_entry2 = smoothing_cols[1].number_input('Polynomial order', value=0, help="Order of the polynomial used to fit the samples")
            if smoothing_entry1 > 0 and smoothing_entry2 > 0:
                signals_s = signal.smooth(smoothing_entry1, smoothing_entry2)
                signals_s.name += ' - smoothed'
                signals_s.plot(figure)
                signal = signals_s

            # --------------------------------------------- DATA EXTRACTION --------------------------------------------

            max_button = st.sidebar.checkbox('Display maximum', key='max_button_')
            min_button = st.sidebar.checkbox('Display minimum', key='min_button_')
            fwhm_cols = st.sidebar.columns(2)
            fwhm_button = fwhm_cols[0].checkbox('Display FWHM', key='fwhm_button_')
            fwhm_button_interp = fwhm_cols[1].checkbox('use interpolation', key='fwhm_button_interp_',
                                                       help="""Use interpolation to improve the calculation of the FWHM""")
            buttons = (max_button, min_button, fwhm_button)

            if sum(buttons) > 0:
                st.markdown('### About your data')
                columns = st.columns(sum(buttons))

                # Max point
                if max_button:
                    col = columns[0]
                    col.markdown('##### Maximum')
                    x_max, y_max, i_max = signal.get_max()
                    col.markdown(x_max.get_value_label_html())
                    col.markdown(y_max.get_value_label_html())

                    # Display the maximum point
                    trace = go.Scatter(x=x_max.data, y=y_max.data, mode='markers', name='Max. point')
                    figure.add_trace(trace)

                # Min point
                if min_button:
                    col = columns[sum(buttons[:1])]
                    col.markdown('##### Minimum')
                    x_min, y_min, i_min = signal.get_min()
                    col.markdown(x_min.get_value_label_html())
                    col.markdown(y_min.get_value_label_html())

                    # Display the minimum point
                    trace = go.Scatter(x=x_min.data, y=y_min.data, mode='markers', name='Min. point')
                    figure.add_trace(trace)

                # FWHM
                if fwhm_button:
                    col = columns[sum(buttons[:2])]
                    col.markdown('##### FWHM')
                    fwhm, x_left, y_left, x_right, y_right = signal.get_fwhm(fwhm_button_interp)
                    col.markdown(fwhm.get_value_label_html())

                    # Display the FWHM
                    trace = go.Scatter(x=[x_left.data[0], x_right.data[0]], y=[y_left.data[0], y_right.data[0]], name='FWHM')
                    figure.add_trace(trace)

                figure.update_traces(marker_size=15)

        # If multiple signals are selected
        else:
            figure = pp.plot(signal)

        plot_spot.plotly_chart(figure, use_container_width=True)

    else:
        st.warning("Unable to read that file. You can submit your file to emmanuel.pean@swansea.ac.uk and it will be added to the list")


with st.expander('Changelog'):
    st.markdown("""
    #### Upcoming features
    * Multiple files upload
    #### October 2022 - V 0.2
    * Added the following options when a single data set is displayed:
        * Download button do download the data displayed
        * Option to change the data range displayed
        * Option to smooth the data using the Savitzky-Golay filter
        * Option to determine the maximum and minimum point, and FWHM""")


# ------------------------------------------------------ ANALYTICS -----------------------------------------------------

components.html("""<script async defer data-website-id="820a9cf5-cbf2-430f-aff1-c6b647194d74" 
src="https://pears-tracking.herokuapp.com/umami.js"></script>""")

# ------------------------------------------------------- FOOTER -------------------------------------------------------

footer = """<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>App created and maintained by <a href="mailto:emmanuel.pean@swansea.ac.uk" target="_blank">Emmanuel V. Péan</a>
(<a href="https://twitter.com/emmanuel_pean" target="_blank">Twitter</a>) - Version 0.2
</div>"""
st.markdown(footer, unsafe_allow_html=True)


# TODO optimise code to only plot and load data if file uploader changed
