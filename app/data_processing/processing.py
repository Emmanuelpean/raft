"""Module containing functions to process and modify data"""

import numpy as np
import scipy.interpolate as sci
import scipy.integrate as scint


def normalise(
    ndarray: np.ndarray,
    other: None | np.ndarray = None,
) -> np.ndarray:
    """Normalise a numpy array
    :param ndarray: ndarray of floats or ints
    :param other: if provided, normalise ndarray with respect to a"""

    if other is None:
        other = ndarray
    return ndarray / np.nanmax(other)


def feature_scale(
    ndarray: np.ndarray,
    a: float = 1.0,
    b: float = 0.0,
    other: None | np.ndarray = None,
) -> np.ndarray:
    """Feature scale a numpy array ndarray
    :param ndarray: ndarray of floats or ints
    :param other: if provided, normalise ndarray with respect to a
    :param a: minimum value
    :param b: maximum value"""

    if other is None:
        other = ndarray
    return b + (ndarray - np.nanmin(other)) * (a - b) / (np.nanmax(other) - np.nanmin(other))


def interpolate_point(
    x: np.ndarray,
    y: np.ndarray,
    index: int,
    nb_point: int = 2,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate data around a given index
    :param x: ndarray corresponding to the x-axis
    :param y: ndarray corresponding to the y-axis
    :param index: index of the point
    :param nb_point: number of point to consider for the interpolation"""

    index1 = max([0, index - nb_point])
    index2 = min([len(x), index + nb_point + 1])
    x = x[index1:index2]
    y = y[index1:index2]

    # Interpolate the ys with the new x
    new_x = np.linspace(np.min(x), np.max(x), 1000)
    interp = sci.interp1d(x, y, **kwargs)
    return new_x, interp(new_x)


def interpolate_data(
    x: np.ndarray,
    y: np.ndarray,
    dx: int | float,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the y data of a signal
    :param x: x data
    :param y: y data
    :param dx: step (float) or array or number of points (int) of the new x data.
    :param kwargs: keyword arguments passed to the scipy interp1d function"""

    if dx <= 0:
        raise AssertionError("dx cannot be negative")

    if isinstance(dx, int):
        new_x = np.linspace(np.min(x), np.max(x), dx)
    else:
        new_x = np.arange(np.min(x), np.max(x), dx)
        if new_x[-1] > x[-1]:
            new_x = new_x[:-1]

    # Interpolate the ys with the new x
    interp = sci.interp1d(x, y, **kwargs)
    new_y = interp(new_x)

    return new_x, new_y


def get_derivative(
    x: np.ndarray,
    y: np.ndarray,
    n: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the nth derivative of a ndarray
    :param np.ndarray x: x values
    :param np.ndarray y: y values (float or SignalData)
    :param int n: order of the derivative"""

    for i in range(n):
        dx = np.diff(x)
        dy = np.diff(y)
        y = dy / dx
        x = (x[1:] + x[:-1]) / 2

    return x, y


def finite_argm(
    method: str,
    data: np.ndarray,
) -> int:
    """Get the argmin or argmax of an array ignoring inf and nan
    :param method: numpy method called
    :param data: numpy array
    :return: index or raise an error if all nan/inf"""

    data = data.copy()
    try:
        data[np.where(np.abs(data) == np.inf)] = float("nan")
    except ValueError:
        pass
    return int(getattr(np, "nan" + method)(data))


def get_area(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Get the area under the curve
    :param x: ndarray corresponding to the x-axis
    :param y: ndarray corresponding to the y-axis"""

    return scint.trapezoid(y, x)
