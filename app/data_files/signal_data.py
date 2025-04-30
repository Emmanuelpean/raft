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

import numpy as np
import plotly.graph_objects as go
import scipy.signal as ss

from config import constants
from data_processing.fitting import fit_data
from data_processing.processing import (
    normalise,
    feature_scale,
    interpolate_point,
    interpolate_data,
    get_derivative,
    finite_argm,
    get_area,
)
from interface.plot import make_figure
from utils.miscellaneous import merge_dicts
from utils.strings import number_to_str, get_label_html


class Dimension(object):
    """A Dimension corresponds to a 0D or 1D data associated with a physical quantity and unit"""

    def __init__(
        self,
        data: any,
        quantity: str = "",
        unit: str = "",
    ) -> None:
        """Initialise the object by storing the parameters as attributes
        :param data: value or np.ndarray of values
        :param quantity: quantity associated with the data. An empty string by default.
        :param unit: unit associated with the data. An empty string by default."""

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

    def __getitem__(self, item) -> Dimension:
        """getitem special method"""

        return self(self.data[item])

    # ----------------------------------------------------- LABELS -----------------------------------------------------

    def get_quantity_label_html(self) -> str:
        """Get the quantity label"""

        # Capitalise the quantity if it's more than 1 character
        capitalise = sum([c.isalpha() for c in self.quantity]) > 1

        # Return the label
        return get_label_html(self.quantity, constants.QUANTITIES_LABEL, capitalise)

    def get_unit_label_html(self) -> str:
        """Get the unit label"""

        return get_label_html(self.unit, constants.UNITS_LABEL, False)

    def get_label_html(self) -> str:
        """Return the quantity and unit as a string"""

        if self.unit:
            return f"{self.get_quantity_label_html()} ({self.get_unit_label_html()})"
        else:
            return self.get_quantity_label_html()

    def get_label_raw(self) -> str:
        """Get a simple label of the dimension"""

        if self.unit:
            return f"{self.quantity} ({self.unit})"
        else:
            return self.quantity

    def get_value_label_html(self, *args, **kwargs) -> str:
        """Return the value label for display"""

        if not isinstance(self.data, (int, float, np.integer)):
            raise AssertionError("Data should be an integer or float")

        label = f"{self.get_quantity_label_html()}: {number_to_str(self.data, *args, **kwargs)}"
        if self.unit:
            return f"{label} {self.get_unit_label_html()}"
        else:
            return label

    def __repr__(self) -> str:
        """__repr__ method"""

        # Data
        string = f"Dimension({self.data}"

        # Quantity
        if self.quantity:
            string += f", {self.quantity}"

        # Unit
        if self.unit:
            string += f", {self.unit}"

        return string + ")"


class SignalData(object):
    """A SignalData object corresponds to 2 Dimension objects (X and Y), a filename, a signalname, and a
    dictionary for storing additional information"""

    def __init__(
        self,
        x: Dimension | list[Dimension] | tuple[Dimension],
        y: Dimension | list[Dimension] | tuple[Dimension],
        filename: str = "",
        signalname: str | list[str] = "",
        z_dict: dict | None = None,
    ) -> None:
        """Initialise the object by processing and storing the input arguments
        :param x: x dimension
        :param y: y dimension
        :param filename: file filename
        :param signalname: signal filename
        :param z_dict: additional optional information about the signal"""

        # For each dimension, if the input is a list of Dimensions, convert them to a single Dimension, else store the dimension
        if isinstance(x, (list, tuple)):
            self.x = Dimension(np.array([dim.data for dim in x]), x[0].quantity, x[0].unit)
        else:
            self.x = x

        if isinstance(y, (list, tuple)):
            self.y = Dimension(np.array([dim.data for dim in y]), y[0].quantity, y[0].unit)
        else:
            self.y = y

        # Add missing quantity and units only if both are missing
        if not self.x.quantity and not self.x.unit:
            self.x.quantity, self.x.unit = "X-quantity", "X-unit"
        if not self.y.quantity and not self.y.unit:
            self.y.quantity, self.y.unit = "Y-quantity", "Y-unit"

        # File name and signal name
        self.filename = filename
        if isinstance(signalname, str):
            self.signalnames = [signalname]
        else:
            self.signalnames = signalname

        # Z dictionary
        if z_dict is None:
            self.z_dict = {}
        else:
            self.z_dict = z_dict

    def __call__(self, *args: object, **kwargs: object) -> SignalData:
        """Return a SignalData object with the same parameter as this one, but overridden with args and kwargs.
        :param args: arguments passed to the constructor in order
        :param kwargs: keyword arguments passed to the constructor"""

        args = dict(zip(("x", "y", "filename", "signalname", "z_dict"), args))
        input_kwargs = dict(x=self.x, y=self.y, filename=self.filename, signalname=self.signalnames, z_dict=self.z_dict)
        return SignalData(**merge_dicts(args, kwargs, input_kwargs))

    @property
    def signalname(self) -> str:
        """Return the formatted signal name"""

        return " - ".join([name for name in self.signalnames if name])

    def get_name(self, filename: bool) -> str:
        """Get the filename of the signal for display purpose
        :param filename: if True, use the filename
        :return: filename if the signalname does not exist
                 filename + signalname if filename is True and exist, and signalname exist
                 signalname if signalname exist and filename is False or does not exist"""

        # If no signal name, just return the filename
        if not self.signalname:
            return self.filename

        # If the signal has a filename
        else:
            # If filename arg, return the long name
            if filename and self.filename:
                return f"{self.filename}: {self.signalname}"

            # Else, just return the signalname
            else:
                return self.signalname

    def plot(
        self,
        figure: go.Figure | None = None,
        filename: bool = False,
        secondary_y: bool = False,
        plot_method: str = "Scattergl",
        **kwargs,
    ) -> go.Figure:
        """Plot the signal data in a plotly figure and format it
        :param figure: plotly figure object
        :param filename: argument passed to get_name
        :param secondary_y: if True, use the secondary y-axis
        :param plot_method: "Scatter" or "Scattergl"
        :param kwargs: keyword arguments passed to Scatter"""

        # Generate a new figure if not provided
        if figure is None:
            figure = make_figure(secondary_y)

        # Hover template
        hovertemplate = "<extra></extra>X: %{x}<br>Y: %{y}<br>" + f"File {self.filename}<br>"
        if self.signalname:
            hovertemplate += f"Curve: {self.signalname}<br>"

        # Generate the trace and add it to the figure
        trace = getattr(go, plot_method)(
            x=self.x.data,
            y=self.y.data,
            name=self.get_name(filename),
            legendgrouptitle_font=dict(size=17),
            hovertemplate=hovertemplate,
            **kwargs,
        )
        figure.add_trace(trace, secondary_y=secondary_y)

        # If it's the first plot in the figure, update the axes
        primary_y_traces = [trace for trace in figure.data if getattr(trace, "yaxis", "y") == "y"]
        secondary_y_traces = [trace for trace in figure.data if getattr(trace, "yaxis", "y") == "y2"]

        if len(primary_y_traces) == 1 or len(secondary_y_traces) == 1:

            # Set the axes settings
            font = dict(size=16, color="black")
            axes_kwargs = dict(
                automargin="left+top+bottom+right",
                title_font=font,
                tickfont=font,
                showgrid=True,
                gridcolor="lightgray",
                exponentformat="power",
                tickformat=",",
                color="black",
            )

            # X-axis
            figure.update_xaxes(
                title_text=self.x.get_label_html(),
                **axes_kwargs,
            )

            # If datetimes are displayed, update the tickformat of the xaxes
            if isinstance(self.x.data[0], dt.datetime):
                figure.update_xaxes(tickformat=None)

            # Y-axis
            if secondary_y:
                figure.update_yaxes(
                    title_text=self.y.get_label_html(),
                    zeroline=False,
                    secondary_y=secondary_y,
                    **merge_dicts(dict(showgrid=False), axes_kwargs),
                )
            else:
                figure.update_yaxes(
                    title_text=self.y.get_label_html(),
                    **axes_kwargs,
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
            self.signalnames,
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
            self.signalnames + ["Smoothed"],
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
            self.signalnames,
            self.z_dict,
        )

    def normalise(self, *args, **kwargs) -> SignalData:
        """Normalise the signal"""

        y = self.y(normalise(self.y.data, *args, **kwargs))
        return SignalData(
            self.x,
            y,
            self.filename,
            self.signalnames,
            self.z_dict,
        )

    def feature_scale(self, *args, **kwargs) -> SignalData:
        """Feature scale the signal"""

        y = self.y(feature_scale(self.y.data, *args, **kwargs))
        return SignalData(
            self.x,
            y,
            self.filename,
            self.signalnames,
            self.z_dict,
        )

    def interpolate(self, *args, **kwargs) -> SignalData:
        """Interpolate the signal"""

        x_data, y_data = interpolate_data(self.x.data, self.y.data, *args, **kwargs)
        return SignalData(
            self.x(x_data),
            self.y(y_data),
            self.filename,
            self.signalnames,
            self.z_dict,
        )

    def derive(self, *args, **kwargs) -> SignalData:
        """Calculate the signal derivative"""

        x_data, y_data = get_derivative(self.x.data, self.y.data, *args, **kwargs)
        return SignalData(
            self.x(x_data),
            self.y(y_data),
            self.filename,
            self.signalnames,
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
            self.signalnames + ["Fit"],
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

        try:
            if point == "min":
                method, quantity = "argmin", constants.MIN_QT
            else:
                method, quantity = "argmax", constants.MAX_QT
            x_data, y_data = self.x.data, self.y.data
            index = finite_argm(method, y_data)

            if interpolation:
                x_data, y_data = interpolate_point(self.x.data, self.y.data, index, kind="cubic")
                index = getattr(y_data, method)()

            # Point dimensions
            x_e = self.x(data=x_data[index], quantity=quantity + " " + self.x.quantity)
            y_e = self.y(data=y_data[index], quantity=quantity + " " + self.y.quantity)

            return x_e, y_e, Dimension(index)
        except:
            return tuple((Dimension(data=float("nan")),) * 3)

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
            x_ext = m[0](quantity=m[0].quantity.replace(constants.MIN_QT, "extremum"))
            y_ext = m[1](quantity=m[1].quantity.replace(constants.MIN_QT, "extremum"))
            i_ext = m[2]
        else:
            x_ext = M[0](quantity=M[0].quantity.replace(constants.MAX_QT, "extremum"))
            y_ext = M[1](quantity=M[1].quantity.replace(constants.MAX_QT, "extremum"))
            i_ext = M[2]

        return x_ext, y_ext, i_ext

    def get_fwhm(self, interpolation: bool = False) -> tuple[Dimension, Dimension, Dimension, Dimension, Dimension]:
        """Get the signal FWHM. Calculate the maximum point if not already stored"""

        try:
            x_ext, y_ext, i_ext = self.get_extrema()
            m = i_ext.data

            # Get the half-intensity on the right of the extremum
            if m != len(self.y.data) - 1:
                x_right, y_right = self.get_halfint_point(self.x.data[m:], self.y.data[m:], interpolation)
            else:
                x_right, y_right = float("nan"), float("nan")

            # Get the half-intensity on the left of the extremum
            if m != 0:
                x_left, y_left = self.get_halfint_point(self.x.data[: m + 1], self.y.data[: m + 1], interpolation)
            else:
                x_left, y_left = float("nan"), float("nan")

            return (
                self.x(data=np.abs(x_right - x_left), quantity=constants.FWHM_QT),
                self.x(data=x_left),
                self.y(data=y_left),
                self.x(data=x_right),
                self.y(data=y_right),
            )

        except:
            return tuple((Dimension(float("nan")),) * 5)

    def get_halfint_point(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        interpolation: bool = False,
        halfint: float | None = None,
    ) -> tuple[float, float]:
        """Calculate the half intensity point
        :param x_data: x data
        :param y_data: y data
        :param interpolation: if True, use interpolation to improve the calculation
        :param halfint: optional half intensity. If not provided, calculate it through the x and y data"""

        try:
            # Calculate the half intensity
            if not halfint:
                y_min = np.min(y_data)
                y_max = np.max(y_data)

                # Determine if the peak is upside down
                if np.abs(y_min) > np.abs(y_max):
                    halfint = (y_min - y_max) / 2.0 + y_max
                else:
                    halfint = (y_max - y_min) / 2.0 + y_min

            # Find the closest points to the half intensity
            index = finite_argm("argmin", np.abs(y_data - halfint))

            # Interpolation
            if interpolation:
                x_int, y_int = interpolate_point(x_data, y_data, index)
                return self.get_halfint_point(x_int, y_int, False, halfint)

            return float(x_data[index]), float(y_data[index])

        except:
            return float("nan"), float("nan")

    def get_area(self) -> Dimension:
        """Get the area under the curve"""

        try:
            return Dimension(get_area(self.x.data, self.y.data), constants.AREA_QT)
        except Exception as e:
            print("Calculating the area failed")
            print(e)
            return Dimension(float("nan"), constants.AREA_QT)


def average_signals(
    signals: list[SignalData],
    dimensions: list[Dimension],
    n: int,
) -> tuple[list[SignalData], list[Dimension]]:
    """Average every n signals and the associated dimensions
    :param signals: list of SignalData objects
    :param dimensions: list of Dimension objects associated with the signals
    :param n: number of signals averaged"""

    # Extract the data
    signal_data = [signal.y.data for signal in signals]
    dimension_data = [dim.data for dim in dimensions]

    # Remove the unnecessary data
    n = min([n, len(signal_data)])
    index = len(signal_data) - (len(signal_data) % n)
    signal_data = signal_data[:index]
    dimension_data = dimension_data[:index]

    # Average
    s_avg_data = [np.mean(signal_data[i : i + n], axis=0) for i in range(0, len(signal_data), n)]
    d_avg_data = [dimensions[0](np.mean(dimension_data[i : i + n])) for i in range(0, len(dimension_data), n)]

    # Generate the averaged signals
    ys = [signals[0].y(data=avg) for avg in s_avg_data]
    signals_avg = [signals[0](y=y, signalname=f"Averaged {i + 1}") for i, y in enumerate(ys)]

    return signals_avg, d_avg_data


def get_z_dim(
    signals: list[SignalData],
    key: str,
    shift: bool = False,
) -> tuple[list[SignalData], list[Dimension]]:
    """Get the z dimension from the z_dict based on the choice of sorting key
    :param signals: list of SignalData
    :param key: z_dict key
    :param shift: if True and the key is a timestamp, shift the data to 0"""

    z_dims0 = [Dimension(i + 1, "Z-axis") for i in range(len(signals))]

    if len(signals) > 1:

        if key == "Filename":
            signals = sorted(signals, key=lambda s: s.get_name(True))
            return signals, z_dims0

        else:

            try:
                # Z dimensions list
                z_dims = [signal.z_dict[key] for signal in signals]

                # Try to convert them to float if string
                if isinstance(z_dims[0].data, str):
                    for z in z_dims:
                        z.data = float(z.data)

                # Sort the signals
                signals, z_dims = [list(f) for f in zip(*sorted(zip(signals, z_dims), key=lambda v: v[1].data))]

                # Time shift
                if shift and key == constants.TIMESTAMP_ID:
                    z_dims = [z((z.data - z_dims[0].data).total_seconds(), unit=constants.SECOND_UNIT) for z in z_dims]

                return signals, z_dims

            except Exception as e:
                print("An error occurred while trying to sort the signals")
                print(e)
                return signals, z_dims0

    else:
        return signals, z_dims0
