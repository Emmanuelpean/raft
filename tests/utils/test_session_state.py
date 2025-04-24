"""Test module for the functions in the `utils/session_state.py` module.

This module contains unit tests for the functions implemented in the `utils/session_state.py` module. The purpose of these tests is to
ensure the correct functionality of each function in different scenarios and to validate that the expected outputs are
returned.

Tests should cover various edge cases, valid inputs, and any other conditions that are necessary to confirm the
robustness of the functions."""

import streamlit as st

from utils.session_state import refresh_session_state_widgets, set_session_state_value_function, set_default_widgets


class TestRefreshSessionStateWidgets:

    def test_refresh_with_list(self) -> None:

        st.session_state.a = 1
        st.session_state.b = 2
        refresh_session_state_widgets(["a", "b"])
        assert st.session_state["a"] == 1
        assert st.session_state["b"] == 2

    def test_refresh_with_dict_keys(self) -> None:

        st.session_state.a = 10
        st.session_state.b = 20
        refresh_session_state_widgets({"a": None, "b": None})
        assert st.session_state["a"] == 10
        assert st.session_state["b"] == 20


class TestSetSessionStateValueFunction:

    def test_set_value_function(self) -> None:

        st.session_state.a = 1
        set_session_state_value_function("a", 42)()
        assert st.session_state.a == 42


class TestSetDefaultWidgets:

    def test_set_defaults_when_missing(self) -> None:

        defaults = {"foo": 1, "bar": 2}
        set_default_widgets(defaults)
        assert st.session_state["foo"] == 1
        assert st.session_state["bar"] == 2

    def test_no_override_without_reset(self) -> None:

        st.session_state.foo = 99
        defaults = {"foo": 1}
        set_default_widgets(defaults, reset=False)
        assert st.session_state.foo == 99  # should not change

    def test_override_with_reset(self) -> None:

        st.session_state.foo = 99
        defaults = {"foo": 1}
        set_default_widgets(defaults, reset=True)
        assert st.session_state.foo == 1
