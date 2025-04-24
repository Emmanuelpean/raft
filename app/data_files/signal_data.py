"""This module contains classes and functions for handling, analyzing, and plotting 2D signal data. It provides tools for:
- Defining and managing physical quantities and their units with the `Dimension` class.
- Storing signal data as `SignalData` objects, which hold associated x and y dimensions (data, quantity, and unit).
- Plotting signal data using Plotly.
- Calculating signal properties such as extrema, Full Width at Half Maximum (FWHM), and half-intensity points.
- Smoothing and reducing the range of signal data.
- Interpolating signal data for more accurate results.

Key Components:
---------------
1. **Dimension Class**: Represents a physical dimension (e.g., time, intensity, etc.) associated with data values,
quantities, and units. It includes methods to generate formatted labels for displaying data.
2. **SignalData Class**: Represents a 2D signal with associated x and y `Dimension` objects. This class includes
methods for data manipulation (e.g., smoothing, range reduction), calculating extrema, and plotting.
3. **get_label Function**: Converts a string to a formatted html label suitable for use with plotly."""

from __future__ import annotations

import datetime as dt
import re

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as ps
import scipy.signal as ss

from config import constants
from data_processing.fitting import fit_data
from utils.miscellaneous import merge_dicts
from utils.string import number_to_str
from data_processing.processing import (
    normalise,
    feature_scale,
    interpolate_point,
    interpolate_data,
    get_derivative,
)


def get_label(
    string: str,
    dictionary: dict[str, str],
    capitalize: bool = True,
) -> str:
    """Create a label from a string to display it with matplotlib
    :param string: string
    :param dictionary: dictionary containing conversion for specific strings
    :param capitalize: if True, try to capitalize the label
    Note: The string should not contain the symbol '?'"""

    if "?" in string:
        raise AssertionError("? should not be in the string")

    label = " " + string + " "  # initialise the label

    # Replace the strings that have a dictionary entry only if they start and end with a white space or _ or ^
    for item in sorted(dictionary.keys(), key=lambda a: -len(a)):
        condition_start = any(x + item in label for x in [" ", "_", "^"])
        condition_end = any(item + x in label for x in [" ", "_", "^"])
        if condition_start and condition_end:
            label = label.replace(item, "?" + dictionary[item])

    # Superscripts
    label = re.sub(r"\^([0-9a-zA-Z+-]+)", r"<sup>\1</sup>", label)

    # Subscripts
    label = re.sub(r"_([0-9a-zA-Z+-]+)", r"<sub>\1</sub>", label)

    # Remove whitespaces
    label = label.strip()

    # Capitalize
    if label[0] != "?" and capitalize and len(label.split("_")[0]) > 1:
        label = label[0].capitalize() + label[1:]

    label = label.replace("?", "")
    label = label.replace("|", " ")

    return label


class Dimension(object):
    """A Dimension corresponds to a 0D or 1D data associated with a physical quantity and unit"""

    def __init__(
        self,
        data: any,
        quantity: str = "",
        unit: str = "",
    ) -> None:
        """Object initialisation
        :param data: value or np.ndarray of values
        :param quantity: quantity associated with the data
        :param unit: unit associated with the data"""

        self.data = data
        self.quantity = quantity
        self.unit = unit

    def __call__(self, *args: object, **kwargs: object) -> Dimension:
        """Return a Dimension object with the same parameter as this one, but overridden with args and kwargs.
        :param args: arguments passed to the constructor in order
        :param kwargs: keyword arguments passed to the constructor"""

        args = dict(zip(("data", "quantity", "unit"), args))
        input_kwargs = dict(data=self.data, quantity=self.quantity, unit=self.unit)
        return Dimension(**merge_dicts(args, kwargs, input_kwargs))

    def get_quantity_label_html(self) -> str:
        """Get the quantity label"""

        if self.quantity:
            # If the label is only one alpha character
            if sum([c.isalpha() for c in self.quantity]) == 1:
                capitalise = False

            else:
                capitalise = True

            return get_label(self.quantity, constants.quantities_label, capitalise)

        else:
            return ""

    def get_unit_label_html(self) -> str:
        """Get the unit label"""

        if self.unit:
            return get_label(self.unit, constants.units_label, False)
        else:
            return ""

    def get_label_html(self) -> str:
        """Return the quantity and unit as a string"""

        if self.unit:
            return self.get_quantity_label_html() + " (" + self.get_unit_label_html() + ")"
        else:
            return self.get_quantity_label_html()

    def get_value_label_html(self) -> str:
        """Return the value label for display"""

        label = f"{self.get_quantity_label_html()}: {number_to_str(self.data, 4, True)}"
        if self.unit:
            return f"{label} {self.get_unit_label_html()}"
        else:
            return label

    def get_label_raw(self) -> str:
        """Get a simple label of the dimension"""

        if self.unit:
            return f"{self.quantity} ({self.unit})"
        else:
            return self.quantity

    def __repr__(self) -> str:
        """__repr__ method"""

        # Data
        string = f"Dimension({self.data}"

        # Quantity, unit
        if self.quantity:
            string += f", {self.quantity}"
        if self.unit:
            string += f", {self.unit}"

        return string + ")"


class SignalData(object):
    """A SignalData object corresponds to 2 Dimension objects (X and Y), a filename, a signalname, and a
    dictionary for storing additional information"""

    def __init__(
        self,
        x: Dimension | list[Dimension],
        y: Dimension | list[Dimension],
        filename: str = "",
        signalname: str | list[str] = "",
        z_dict: dict | None = None,
    ) -> None:
        """Initialise the object by storing the input arguments
        :param x: x dimension
        :param y: y dimension
        :param filename: file filename
        :param signalname: signal filename
        :param z_dict: additional optional information about the signal"""

        # If the input is a list of Dimensions, convert them to a single Dimension
        if isinstance(x, (list, tuple)):
            x = Dimension(np.array([d.data for d in x]), x[0].quantity, x[0].unit)
        if isinstance(y, (list, tuple)):
            y = Dimension(np.array([d.data for d in y]), y[0].quantity, y[0].unit)

        self.x = x
        self.y = y
        if not self.x.quantity:
            self.x.quantity = "X-quantity"
            if not self.x.unit:
                self.x.unit = "X-unit"
        if not self.y.quantity:
            self.y.quantity = "Y-quantity"
            if not self.y.unit:
                self.y.unit = "Y-unit"

        # Names
        self.filename = filename
        if isinstance(signalname, str):
            self.signalname = [signalname]
        else:
            self.signalname = signalname

        # Z dict
        if z_dict is None:
            self.z_dict = {}
        else:
            self.z_dict = z_dict

    def get_name(self, filename: bool) -> str:
        """Get the filename of the signal for display purpose
        :param filename: if True, use the filename"""

        signalname = [e for e in self.signalname if e]

        # If no signal name, just return the filename
        if not signalname:
            return self.filename

        # If the signal has a filename
        else:
            # If filename arg, return the long name
            if filename and self.filename:
                return self.filename + ": " + " - ".join(signalname)

            # Else, just return the signalname
            else:
                return " - ".join(signalname)

    def plot(
        self,
        figure: go.Figure | None = None,
        filename: bool = False,
        secondary_y: bool = False,
        **kwargs,
    ) -> go.Figure:
        """Plot the signal data in a plotly figure
        :param figure: plotly figure object
        :param filename: argument passed to get_name
        :param secondary_y: if True, use the secondary y-axis
        :param kwargs: keyword arguments passed to Scatter"""

        # Generate a new figure if not provided
        if figure is None:
            figure = ps.make_subplots(1, 1, specs=[[{"secondary_y": secondary_y}]])

        font = dict(size=16, color="black")
        hovertemplate = "X: %{x}<br>Y: %{y}<br>" + f"File {self.filename}<br>"
        signalname = [f for f in self.signalname if f]
        if signalname:
            hovertemplate += f"Curve: {' - '.join(signalname)}<br>"
        hovertemplate += "<extra></extra>"

        # Generate the trace and add it to the figure
        trace = go.Scattergl(
            x=self.x.data,
            y=self.y.data,
            name=self.get_name(filename),
            legendgrouptitle_font=dict(size=17),
            hovertemplate=hovertemplate,
            **kwargs,
        )
        figure.add_trace(trace, secondary_y=secondary_y)

        # Set the x-axis settings
        figure.update_xaxes(
            automargin="left+top+bottom+right",
            title_text=self.x.get_label_html(),
            title_font=font,
            tickfont=font,
            showgrid=True,
            gridcolor="lightgray",
        )

        # If not datetimes are displayed, update the tickformat of the xaxes
        if not isinstance(self.x.data[0], dt.datetime):
            figure.update_xaxes(tickformat=",")

        # Set the y-axis settings
        yaxis_kwargs = dict(
            automargin="left+top+bottom+right",
            title_text=self.y.get_label_html(),
            tickformat=",",
            title_font=font,
            tickfont=font,
            gridcolor="lightgray",
            exponentformat="power",
        )

        if secondary_y:
            figure.update_yaxes(
                showgrid=False,
                zeroline=False,
                color="black",
                secondary_y=secondary_y,
                **yaxis_kwargs,
            )
        else:
            figure.update_yaxes(
                **yaxis_kwargs,
            )

        # Update the figure layout
        figure.update_layout(
            {"uirevision": "foo"},
            margin=dict(l=0, r=0, t=40, b=0, pad=0),
            legend=dict(font=font),
            height=800,
        )

        return figure

    # ---------------------------------------------- SIGNAL TRANSFORMATION ---------------------------------------------

    def remove_background(self, xrange: list[float | int] | tuple[float | int]) -> SignalData:
        """Remove the background signal
        :param xrange: range of values where the average background signal is calculated"""

        index1 = np.abs(self.x.data - xrange[0]).argmin()
        index2 = np.abs(self.x.data - xrange[1]).argmin()
        mean_val = np.mean(self.y.data[index1:index2])
        if np.isnan(mean_val):
            raise AssertionError()
        return SignalData(
            self.x(data=self.x.data),
            self.y(data=self.y.data - mean_val),
            self.filename,
            self.signalname,
            self.z_dict,
        )

    def smooth(self, *args) -> SignalData:
        """Smooth the y data
        :param args: arguments passed to the scipy.signal savgol_filter function"""

        y = ss.savgol_filter(self.y.data, *args)
        return SignalData(
            self.x,
            self.y(data=y),
            self.filename,
            self.signalname + ["Smoothed"],
            self.z_dict,
        )

    def reduce_range(self, xrange: list[float | int] | tuple[float | int]) -> SignalData:
        """Reduce the x-axis range
        :param xrange: data range"""

        index1 = np.abs(self.x.data - xrange[0]).argmin()
        index2 = np.abs(self.x.data - xrange[1]).argmin()
        return SignalData(
            self.x(data=self.x.data[index1:index2]),
            self.y(data=self.y.data[index1:index2]),
            self.filename,
            self.signalname,
            self.z_dict,
        )

    def normalise(self, *args, **kwargs) -> SignalData:
        """Normalise the signal"""

        y = self.y(normalise(self.y.data, *args, **kwargs))
        return SignalData(
            self.x,
            y,
            self.filename,
            self.signalname,
            self.z_dict,
        )

    def feature_scale(self, *args, **kwargs) -> SignalData:
        """Feature scale the signal"""

        y = self.y(feature_scale(self.y.data, *args, **kwargs))
        return SignalData(
            self.x,
            y,
            self.filename,
            self.signalname,
            self.z_dict,
        )

    def interpolate(self, *args, **kwargs) -> SignalData:
        """Interpolate the signal"""

        x_data, y_data = interpolate_data(self.x.data, self.y.data, *args, **kwargs)
        return SignalData(
            self.x(x_data),
            self.y(y_data),
            self.filename,
            self.signalname,
            self.z_dict,
        )

    def derive(self, *args, **kwargs) -> SignalData:
        """Calculate the signal derivative"""

        x_data, y_data = get_derivative(self.x.data, self.y.data, *args, **kwargs)
        return SignalData(
            self.x(x_data),
            self.y(y_data),
            self.filename,
            self.signalname,
            self.z_dict,
        )

    def fit(self, *args, **kwargs) -> tuple[SignalData, np.ndarray, np.ndarray, float]:
        """Fit the data
        :param args: arguments passed to fit_data
        :param kwargs: keyword arguments passed to fit_data"""

        params, param_errors, y_fit, r_squared = fit_data(self.x.data, self.y.data, *args, **kwargs)
        inf_indexes = np.isinf(y_fit)
        x_data, y_data = self.x.data[~inf_indexes], y_fit[~inf_indexes]
        fit_signal = SignalData(
            self.x(x_data),
            self.y(y_data),
            self.filename,
            self.signalname + ["Fit"],
            self.z_dict,
        )
        return fit_signal, params, param_errors, r_squared

    # ------------------------------------------------- DATA EXTRACTION ------------------------------------------------

    def _get_point(
        self,
        point: str,
        interpolation: bool = False,
    ) -> tuple[Dimension, Dimension, Dimension]:
        """Get the extrema point of a signal
        :param point: 'max', 'min'
        :param interpolation: if True, use interpolation to increase the accuracy of the measurement"""

        if point == "min":
            method, quantity = "argmin", constants.min_qt
        else:
            method, quantity = "argmax", constants.max_qt
        x_data, y_data = self.x.data, self.y.data
        index = getattr(y_data, method)()

        if interpolation:
            try:
                x_data, y_data = interpolate_point(self.x.data, self.y.data, index, kind="cubic")
                index = getattr(y_data, method)()
            except:
                pass

        # Point dimensions
        x_e = self.x(data=x_data[index], quantity=quantity + " " + self.x.quantity)
        y_e = self.y(data=y_data[index], quantity=quantity + " " + self.y.quantity)

        return x_e, y_e, Dimension(index)

    def get_max(self, *args, **kwargs) -> tuple[Dimension, Dimension, Dimension]:
        """Get the maximum intensity signal"""

        return self._get_point("max", *args, **kwargs)

    def get_min(self, *args, **kwargs) -> tuple[Dimension, Dimension, Dimension]:
        """Get the minimum intensity signal"""

        return self._get_point("min", *args, **kwargs)

    def get_extrema(self) -> tuple[Dimension, Dimension, Dimension]:
        """Get the extremum intensity signal"""

        M = self.get_max()
        m = self.get_min()
        if abs(M[1].data) < abs(m[1].data):  # accessing data avoid pycharm from highlighting error
            x_ext = m[0](quantity=m[0].quantity.replace(constants.min_qt, "extremum"))
            y_ext = m[1](quantity=m[1].quantity.replace(constants.min_qt, "extremum"))
            i_ext = m[2]
        else:
            x_ext = M[0](quantity=M[0].quantity.replace(constants.max_qt, "extremum"))
            y_ext = M[1](quantity=M[1].quantity.replace(constants.max_qt, "extremum"))
            i_ext = M[2]

        return x_ext, y_ext, i_ext

    def get_fwhm(self, interpolation: bool = False) -> tuple[Dimension, Dimension, Dimension, Dimension, Dimension]:
        """Get the signal FWHM. Calculate the maximum point if not already stored"""

        x_ext, y_ext, i_ext = self.get_extrema()
        m = i_ext.data

        # Get the half-intensity on the right of the extremum
        if m != len(self.y.data) - 1:
            x_right, y_right = self.get_halfint_point(self.x.data[m:], self.y.data[m:], interpolation)
        else:
            x_right, y_right = None, None

        # Get the half-intensity on the left of the extremum
        if m != 0:
            x_left, y_left = self.get_halfint_point(self.x.data[: m + 1], self.y.data[: m + 1], interpolation)
        else:
            x_left, y_left = None, None

        if x_left is not None and x_right is not None:
            fwhm = x_right - x_left
        else:
            fwhm = float("nan")

        return (
            self.x(data=np.abs(fwhm), quantity=constants.fwhm_qt),
            self.x(data=x_left),
            self.y(data=y_left),
            self.x(data=x_right),
            self.y(data=y_right),
        )

    def get_halfint_point(
        self,
        x: np.ndarray,
        y: np.ndarray,
        interpolation: bool = False,
        halfint: float | None = None,
    ) -> tuple[float, float]:
        """Calculate the half intensity point
        :param x: x data
        :param y: y data
        :param interpolation: if True, use interpolation to improve the calculation
        :param halfint: optional half intensity. If not provided, calculate it through the x and y data"""

        # Calculate the half intensity
        if not halfint:
            y_min = np.min(y)
            y_max = np.max(y)

            # Determine if the peak is upside down
            if np.abs(y_min) > np.abs(y_max):
                halfint = (y_min - y_max) / 2.0 + y_max
            else:
                halfint = (y_max - y_min) / 2.0 + y_min

        # Find the closest points to the half intensity
        index = np.abs(y - halfint).argsort()[0]

        # Interpolation
        if interpolation:
            try:
                x_int, y_int = interpolate_point(x, y, index)
                return self.get_halfint_point(x_int, y_int, False, halfint)
            except:  # pragma: no cover
                print("Interpolation failed")
                pass

        return x[index], y[index]
