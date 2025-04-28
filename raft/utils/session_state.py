"""This module contains general functions to handle the streamlit session state"""

import streamlit as st


def refresh_session_state_widgets(widgets: list | tuple | dict) -> None:
    """Refresh the session state settings. To be called when new widgets are displayed
    :param widgets: list-like of widgets"""

    for key in widgets:
        st.session_state[key] = st.session_state[key]


def set_session_state_value_function(
    key: str,
    value: any,
) -> callable:
    """Return a function that changes the value of a key in the session state
    :param key: session state key
    :param value: new value"""

    def function() -> None:
        """Set the session state key value"""

        st.session_state[key] = value

    return function


def set_default_widgets(
    defaults: dict,
    reset: bool = False,
) -> None:
    """Store the settings in the session state using their default value
    :param defaults: dictionary containing the default value of the widgets
    :param reset: if True, reset the stored value"""

    for key in defaults:
        if key not in st.session_state or reset:
            st.session_state[key] = defaults[key]
