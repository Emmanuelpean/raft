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

# ----------------------------------------------------- DATA INPUT -----------------------------------------------------

# File uploader
file = st.sidebar.file_uploader('File uploader', label_visibility='hidden')
print('File is %s' % file)

# File type
filetypes = ['Detect'] + sorted(pdf.functions.keys())
filetype_help = "Select the file type. If 'Detect' is selected, the file type will be automatically detected"
filetype = st.sidebar.selectbox('Data file type', filetypes, help=filetype_help)
filetype_message = st.sidebar.empty()
print(filetype)

signal = None

# If no file is provided or no signal is stored
if not file:
    st.info('RAFT is a free tool to plot the content a various data files. Just drag and drop your file and get the relevant information from it!')

else:

    # -------------------------------------------------- DATA LOADING --------------------------------------------------
    print('loading file')

    # Attempt to load the data by testing every file types
    if filetype == 'Detect':

        # Only search through the filetypes matching the extension
        extension = os.path.splitext(file.name)[1]
        functions = {key: value[0] for key, value in pdf.functions.items() if
                     extension.lower() == '.' + pdf.functions[key][1].lower() or pdf.functions[key][1] == ''}

        for filetype in functions:
            # noinspection PyBroadException
            try:
                signal = pdf.functions[filetype][0](file)
                filetype_message.markdown('Detected: %s' % filetype)
                break
            except:
                pass

    # Attempt to load the data using the provided function
    else:
        # noinspection PyBroadException
        try:
            signal = pdf.functions[filetype][0](file)
        except:
            pass

    # If the signal could be loaded, display an error message, else store it in the session state
    if signal is None:
        st.warning("Unable to read that file. You can submit your file to emmanuel.pean@swansea.ac.uk and it will be added to the list")

    # If signals are stored in the session state
    else:

        plot_spot = st.empty()

        # ----------------------------------------------- DATA SELECTION -----------------------------------------------

        # Select the data type to plot if signal is a dictionary
        if isinstance(signal, dict):
            selection = st.sidebar.selectbox('Data type to plot', ['All'] + list(signal.keys()))
            if selection != 'All':
                signal = signal[selection]

        # Convert the list of signals to a signal if only 1 signal in the list
        if isinstance(signal, (list, tuple)) and len(signal) == 1:
            signal = signal[0]

        # Select the signal in a list
        if isinstance(signal, (list, tuple)):
            names = [s.get_name(False) for s in signal]
            signal_dict = dict(zip(names, signal))
            col_selection = st.sidebar.selectbox('Data to plot', ['All'] + sorted(signal_dict.keys()))
            if col_selection != 'All':
                signal = signal_dict[col_selection]

        # If only 1 signal is selected
        if isinstance(signal, SignalData):

            # ----------------------------------------------- DATA EXPORT ----------------------------------------------

            header = [signal.x.get_label(), signal.y.get_label()]
            export_data = utils.matrix_to_string([signal.x.data, signal.y.data], header)
            st.download_button('Download data', export_data, 'data.csv')

            # ----------------------------------------------- DATA RANGE -----------------------------------------------

            label = 'Data range'
            if signal.x.unit:
                label += ' (% s)' % signal.x.unit
            st.sidebar.markdown(label)
            range_cols = st.sidebar.columns(2)
            range_button1 = range_cols[0].number_input('a', min_value=signal.x.data[0], max_value=signal.x.data[-2],
                                                       value=signal.x.data[0], label_visibility='collapsed')
            range_button2 = range_cols[1].number_input('a', min_value=signal.x.data[1], max_value=signal.x.data[-1],
                                                       value=signal.x.data[-1], label_visibility='collapsed')
            signal = signal.reduce_range([range_button1, range_button2])

            figure = pp.plot(signal)  # plot the signal

            # ------------------------------------------------ SMOOTHING -----------------------------------------------

            st.sidebar.markdown('Smoothing')
            smoothing_cols = st.sidebar.columns(2)
            smoothing_entry1 = smoothing_cols[0].number_input('Filter length', value=0, help="Length of the Savitzky-Golay filter window")
            smoothing_entry2 = smoothing_cols[1].number_input('Polynomial order', value=0, help="Order of the polynomial used to fit the samples")
            if smoothing_entry1 > 0 and smoothing_entry2 > 0:
                signals_s = signal.smooth(smoothing_entry1, smoothing_entry2)
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


with st.expander('Changelog'):
    st.markdown("""
    #### February 2023 - Minor release
    * Bug fixes
    * Now able to read incomplete Fluoracle files
    #### November 2022 - Minor release
    * Data labelling has been significantly improved.
    * Bug fix and code optimisation.
    * Added Zem3 file support.
    #### October 2022 - V 0.2
    * Added the following options when a single data set is displayed:
        * Download button do download the data displayed.
        * Option to change the data range displayed.
        * Option to smooth the data using the Savitzky-Golay filter.
        * Option to determine the maximum and minimum point, and FWHM.""")


# ------------------------------------------------------ ANALYTICS -----------------------------------------------------

components.html("""<script async defer src="https://analytics.umami.is/script.js" data-website-id="18c966b7-05b9-4473-ab5c-8d493d03f892"></script>""")

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
<p>App created and maintained by <a href="https://emmanuelpean.streamlitapp.com" target="_blank">Emmanuel V. Péan</a> - Version 0.2.3
</div>"""
st.markdown(footer, unsafe_allow_html=True)

print('\n')

# TODO check tests of data_files
# TODO: LATER: add stricter check for file formats
# TODO: generated new labels if already exist in memory
