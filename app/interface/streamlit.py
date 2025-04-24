"""Module containing custom streamlit components"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_files.signal_data import SignalData
from utils.miscellaneous import make_unique


pd.set_option("styler.render.max_elements", 40000000)


def tab_bar(values: list, **kwargs) -> str:
    """Custom tabs
    :param list values: list of tab names
    :param kwargs: keyword arguments passed to st.radio"""

    label = "tab_bar_label"
    active_tab = st.radio(label, values, label_visibility="collapsed", **kwargs)
    child = values.index(active_tab) + 1
    primary_color = st.get_option("theme.primaryColor")
    st.html(
        f"""  
            <style type="text/css">
            div[aria-label="{label}"] {{
                border-bottom: 2px solid rgba(49, 51, 63, 0.1);
                margin-bottom: -2rem !important;
            }}
            div[aria-label="{label}"] > label > div:first-of-type {{
               display: none
            }}
            div[aria-label="{label}"] {{
                flex-direction: unset
            }}
            div[aria-label="{label}"] label {{
                padding-bottom: 0.5em;
                border-radius: 0;
                position: relative;
                top: 4px;
            }}
            div[aria-label="{label}"] label .st-fc {{
                padding-left: 0;
            }}
            div[aria-label="{label}"] label:hover p {{
                color: {primary_color};
            }}
            div[aria-label="{label}"] label:nth-child({child}) {{    
                border-bottom: 2px solid {primary_color};
            }}
            div[aria-label="{label}"] label:nth-child({child}) p {{    
                color: {primary_color};
                padding-right: 0;
            }}
            </style>
        """
    )

    return active_tab


def display_data(
    fig: go.Figure,
    dataset: list[SignalData],
    key: int,
    filename: bool,
) -> None:
    """Display a figure and the associated data in different tabs
    :param fig: figure object
    :param dataset: data to display in the dataframe
    :param key: figure key int
    :param filename: argument passed to get_name of the signals"""

    tabs = st.tabs(["Graph", "Data"])

    # Show the figure
    tabs[0].plotly_chart(fig, use_container_width=True, key=f"figure_{key}")

    # Get the data amd columns
    x_data = dataset[0].x.data
    ys_data = [d.y.data for d in dataset]
    df_columns = make_unique([dataset[0].x.get_label_html()] + [d.get_name(filename) for d in dataset])

    # Dataframe
    dataframe = pd.DataFrame({name: array for name, array in zip(df_columns, [x_data] + ys_data)})
    tabs[1].dataframe(dataframe.style.format("{:5g}"), use_container_width=True, hide_index=True)
