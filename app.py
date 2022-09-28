import streamlit as st
import core.data_files as pdf
import core.plot as pp
from core import utils
from core import resources
import streamlit.components.v1 as components
import os

# -------------------------------------------------------- SETUP -------------------------------------------------------

st.set_page_config('Raft', layout='wide')
st.markdown("""%s""" % utils.render_image(resources.logo_text_filename, 25), unsafe_allow_html=True)  # main logo
st.sidebar.markdown("""%s""" % utils.render_image(resources.logo_filename, 30), unsafe_allow_html=True)  # sidebar logo
if 'filetype' not in st.session_state:
    st.session_state.filetype = 'Detect'

# -------------------------------------------------------- MAIN --------------------------------------------------------

# File uploader
filename = st.sidebar.file_uploader('a', on_change=lambda: setattr(st.session_state, 'filetype', 'Detect'), label_visibility='hidden')

# File type
filetypes = ['Detect'] + sorted(pdf.functions.keys())
filetype_sb = st.sidebar.empty()
filetype = filetype_sb.selectbox('Data file', filetypes, index=filetypes.index(st.session_state.filetype))

signals = None

if not filename:
    st.info('Input a file')

# Load the data and plot them
else:

    # Attempt to load the data by testing every file types
    if filetype == 'Detect':

        # Only search through the filetypes matching the extension
        extension = os.path.splitext(filename.name)[1]
        functions = {key: value for key, value in pdf.functions.items() if
                     extension.lower() == '.' + pdf.extensions[key].lower() or pdf.extensions[key] == ''}

        for key in functions:
            # noinspection PyBroadException
            try:
                signals = pdf.functions[key](filename)
                filetype = filetype_sb.selectbox('Data file', filetypes, index=filetypes.index(key))
                st.session_state.filetype = key
                break
            except:
                pass

    # Attempt to load the data using the provided function
    else:
        # noinspection PyBroadException
        try:
            signals = pdf.functions[filetype](filename)
        except:
            pass

    # Plot the signals
    if signals:
        if isinstance(signals, dict):
            selection = st.sidebar.selectbox('Data to plot', ['All'] + list(signals.keys()))
            if selection == 'All':
                figure = pp.plot(signals)
            else:
                figure = pp.plot(signals[selection])
        elif isinstance(signals, (list, tuple)) and len(signals) > 1:
            selection = st.sidebar.selectbox('Column to plot', ['All'] + list(range(1, len(signals) + 1)))
            if selection == 'All':
                figure = pp.plot(signals)
            else:
                figure = pp.plot(signals[int(selection) - 1])
        else:
            figure = pp.plot(signals)
        st.plotly_chart(figure, use_container_width=True)

    else:
        st.warning("Unable to read that file. You can submit your file to emmanuel.pean@swansea.ac.uk and it will be added to the list")

# ------------------------------------------------------ ANALYTICS -----------------------------------------------------

components.html("""<script async defer data-website-id="62a61960-56c2-493b-90e0-20e6796ecfa4" 
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
(<a href="https://twitter.com/emmanuel_pean" target="_blank">Twitter</a>) - Version 0.1  
</div>"""
st.markdown(footer, unsafe_allow_html=True)
