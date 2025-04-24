"""Test module for the functions in the `utils/file_io.py` module.

This module contains unit tests for the functions implemented in the `utils/file_io.py` module. The purpose of these tests is to
ensure the correct functionality of each function in different scenarios and to validate that the expected outputs are
returned.

Tests should cover various edge cases, valid inputs, and any other conditions that are necessary to confirm the
robustness of the functions."""

import base64
import os
import tempfile

import pytest
import streamlit as st

from utils.file_io import read_file, render_image


class TestReadFile:
    """Test class for the read_txt_file function."""

    # File path for temporary test file
    TEMP_FILE = "_temp.txt"

    def teardown_method(self) -> None:
        """Teardown method that runs after each test."""

        # Clean up test file after each test
        if os.path.exists(self.TEMP_FILE):
            os.remove(self.TEMP_FILE)

        # Clear the cache
        st.cache_resource.clear()

    def test_read_existing_file(self) -> None:
        """Test reading from an existing file with valid content."""

        # Create file with some content
        with open(self.TEMP_FILE, "w") as f:
            f.write("Hello, World!")

        # Read the content using our function
        content = read_file(self.TEMP_FILE)

        # Assert the content matches what we wrote
        assert content == "Hello, World!"

    def test_read_multiline_file(self) -> None:
        """Test reading from a file with multiple lines."""

        # Create a file with multiline content
        with open(self.TEMP_FILE, "w") as f:
            f.write("Line 1\nLine 2\nLine 3")

        # Read the content using our function
        content = read_file(self.TEMP_FILE)

        # Assert the content matches what we wrote
        assert content == "Line 1\nLine 2\nLine 3"
        # Additional check for line count
        assert len(content.splitlines()) == 3

    def test_nonexistent_file(self) -> None:
        """Test that trying to read a nonexistent file raises an error."""

        # Check that the function raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            read_file(self.TEMP_FILE)


class TestRenderImage:

    def test_valid_image_file(self) -> None:

        image_data = b"\x89PNG\r\n\x1a\n" + b"fakeimagecontent"
        expected_mime = "image/png"

        # Create a temporary image file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name

        html_output = render_image(tmp_path, width=200)

        encoded = base64.b64encode(image_data).decode()
        expected_html = f'<center><img src="data:{expected_mime};base64,{encoded}" width="200px"/></center>'
        assert html_output == expected_html

    def test_unknown_mime_type_raises(self) -> None:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".unknown") as tmp:
            tmp.write(b"Some binary content")
            tmp_path = tmp.name

        with pytest.raises(ValueError, match=r"Could not determine MIME type"):
            render_image(tmp_path)

    def test_default_width(self) -> None:

        image_data = b"GIF87a" + b"fakegifdata"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name

        html_output = render_image(tmp_path)
        assert 'width="100px"' in html_output
        assert "data:image/gif;base64," in html_output
