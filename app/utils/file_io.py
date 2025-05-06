"""Module containing functions for reading and writing files"""

import base64
import mimetypes

import streamlit as st


@st.cache_resource
def read_file(path: str, mode: str = "r", encoding: str | None = "utf-8",) -> str:
    """Read the content of a file and store it as a resource.
    :param path: file path
    :param mode: open mode
    :param encoding: file encoding"""

    with open(path, mode=mode, encoding=encoding) as ofile:
        return ofile.read()


@st.cache_resource
def render_image(file_path: str, width: int = 100,) -> str:
    """Render an image file as base64 embedded HTML
    :param str file_path: path to the image file
    :param int width: image width in pixels
    :return: HTML string for rendering the image"""

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for file: {file_path}")

    with open(file_path, "rb") as ofile:
        encoded = base64.b64encode(ofile.read()).decode()

    return f'<center><img src="data:{mime_type};base64,{encoded}" width="{width}px"/></center>'
