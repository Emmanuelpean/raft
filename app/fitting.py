"""This module provides a flexible interface for fitting (x, y) data using a variety of mathematical
models, including polynomial, exponential, logarithmic, trigonometric, and statistical functions."""

import inspect

import numpy as np
import scipy.optimize as sco
from scipy.special import wofz


def get_model_parameters(function: callable) -> list[str]:
    """Get a model parameters"""

    return list(inspect.signature(function).parameters.keys())[1:]


def fit_data(
    x_data: np.ndarray,
    y_data: np.ndarray,
    fit_function: callable,
    p0: None | dict[str, float | int] = None,
    bounds: None | dict[str, list[float | int]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | tuple[None, None, None, None]:
    """Fit (x, y) data using a user-defined function and initial parameter guesses.
    :param x_data: x data points to fit
    :param y_data: y data points to fit
    :param fit_function: function with signature f(x, *params) where params are the parameters to be fitted
    :param p0 : Initial guesses for the parameters. If None, the solver will try to determine them
    :param bounds : Lower and upper bounds on parameters"""

    keys = get_model_parameters(fit_function)
    p0 = [p0[key] for key in keys]

    # Perform the curve fit
    fit_kwargs = dict(f=fit_function, xdata=x_data, ydata=y_data, p0=p0)
    if bounds is not None:
        bounds = np.transpose([bounds[key] for key in keys])
        fit_kwargs.update(bounds=bounds)
    # noinspection PyTupleAssignmentBalance
    params, covariance = sco.curve_fit(**fit_kwargs)

    # Calculate the standard deviations of the parameters
    param_errors = np.sqrt(np.diag(covariance))

    # Calculate the fitted y values
    y_fit = fit_function(x_data, *params)

    # Calculate R-squared
    residuals = y_data - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return params, param_errors, y_fit, r_squared


def linear(x: np.ndarray, m: float, b: float) -> np.ndarray:
    """Linear function: f(x) = m*x + b"""

    return m * x + b


def quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Quadratic function: f(x) = a*x^2 + b*x + c"""

    return a * x**2 + b * x + c


def cubic(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Cubic function: f(x) = a*x^3 + b*x^2 + c*x + d"""

    return a * x**3 + b * x**2 + c * x + d


# Exponential and logarithmic functions
def exponential(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Exponential growth function: f(x) = a * exp(b * x) + c"""

    return a * np.exp(b * x) + c


def exponential_decay(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Exponential decay function: f(x) = a * exp(-b * x) + c"""

    return a * np.exp(-b * x) + c


def logarithmic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Logarithmic function: f(x) = a * log(b * x) + c"""

    return a * np.log(b * x) + c


def power_law(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Power law function: f(x) = a * x^b + c"""

    return a * x**b + c


# Trigonometric functions
def sine(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Sine function: f(x) = a * sin(b * x + c) + d"""

    return a * np.sin(b * x + c) + d


def cosine(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Cosine function: f(x) = a * cos(b * x + c) + d"""

    return a * np.cos(b * x + c) + d


def damped_sine(x: np.ndarray, a: float, b: float, c: float, d: float, e: float) -> np.ndarray:
    """Damped sine function: f(x) = a * exp(-b * x) * sin(c * x + d) + e"""

    return a * np.exp(-b * x) * np.sin(c * x + d) + e


# Statistical distributions
def gaussian(x: np.ndarray, a: float, mu: float, sigma: float, c: float) -> np.ndarray:
    """Gaussian function: f(x) = a * exp(-(x-mu)^2 / (2*sigma^2)) + c"""

    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + c


def lorentzian(x: np.ndarray, a: float, x0: float, gamma: float, c: float) -> np.ndarray:
    """Lorentzian function: f(x) = a * gamma^2 / ((x - x0)^2 + gamma^2) + c"""

    return a * gamma**2 / ((x - x0) ** 2 + gamma**2) + c


def voigt(x: np.ndarray, a: float, mu: float, sigma: float, gamma: float, c: float) -> np.ndarray:
    """Voigt function: Convolution of Gaussian and Lorentzian
    Implemented as a pseudo-Voigt approximation"""
    # This is an approximation of the Voigt profile

    z = (x - mu + 1j * gamma) / (sigma * np.sqrt(2))
    return a * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi)) + c


def guess_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Generate initial parameter guesses for linear function: f(x) = m*x + b"""

    m_guess, b_guess = np.polyfit(x, y, 1)  # Slope
    return float(m_guess), float(b_guess)


def guess_quadratic(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Generate initial parameter guesses for quadratic function: f(x) = a*x^2 + b*x + c"""

    coeffs = np.polyfit(x, y, 2)
    a_guess, b_guess, c_guess = coeffs
    return a_guess, b_guess, c_guess


def guess_cubic(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Generate initial parameter guesses for cubic function: f(x) = a*x^3 + b*x^2 + c*x + d"""

    coeffs = np.polyfit(x, y, 3)
    a_guess, b_guess, c_guess, d_guess = coeffs
    return a_guess, b_guess, c_guess, d_guess


def guess_exponential(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Generate initial parameter guesses for exponential growth function: f(x) = a * exp(b * x) + c"""

    # Assuming c is close to min(y)
    c_guess = min(y)

    # Subtract baseline and take log of positive values
    y_adjusted = y - c_guess
    # Protect against negative or zero values
    positive_indices = y_adjusted > 0
    if not np.any(positive_indices):
        # If no positive values, adjust our approach
        c_guess = min(y) - 0.1 * (max(y) - min(y))
        y_adjusted = y - c_guess
        positive_indices = y_adjusted > 0

    x_pos = x[positive_indices]
    y_pos = y_adjusted[positive_indices]

    if len(x_pos) < 2:
        # Not enough points for good estimate, use defaults
        return 1.0, 0.1, c_guess

    # Linear regression on log(y) vs x
    log_y = np.log(y_pos)
    slope, intercept = np.polyfit(x_pos, log_y, 1)

    a_guess = np.exp(intercept)
    b_guess = slope

    return a_guess, b_guess, c_guess


def guess_exponential_decay(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Generate initial parameter guesses for exponential decay function: f(x) = a * exp(-b * x) + c"""

    # Assuming c is close to min(y)
    c_guess = min(y)

    # Subtract baseline
    y_adjusted = y - c_guess

    # Protect against negative or zero values
    positive_indices = y_adjusted > 0
    if not np.any(positive_indices):
        # If no positive values, adjust our approach
        c_guess = min(y) - 0.1 * (max(y) - min(y))
        y_adjusted = y - c_guess
        positive_indices = y_adjusted > 0

    x_pos = x[positive_indices]
    y_pos = y_adjusted[positive_indices]

    if len(x_pos) < 2:
        # Not enough points for good estimate, use defaults
        return max(y) - min(y), 0.1, c_guess

    # Linear regression on log(y) vs x
    log_y = np.log(y_pos)
    slope, intercept = np.polyfit(x_pos, log_y, 1)

    a_guess = np.exp(intercept)
    b_guess = -slope  # Note the negative sign for decay

    return a_guess, b_guess, c_guess


def guess_logarithmic(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Generate initial parameter guesses for logarithmic function: f(x) = a * log(b * x) + c"""

    # Filter out non-positive x values
    positive_indices = x > 0
    x_pos = x[positive_indices]
    y_pos = y[positive_indices]

    if len(x_pos) < 2:
        return 1.0, 1.0, float(np.mean(y))

    # Fit a * log(x) + c to estimate a and c
    log_x = np.log(x_pos)
    a_guess, c_guess = np.polyfit(log_x, y_pos, 1)

    # Heuristic for b: scale based on median x
    b_guess = 1.0 / np.median(x_pos)

    return float(a_guess), float(b_guess), float(c_guess)


def guess_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Generate initial parameter guesses for power law function: f(x) = a * x^b + c"""

    # Estimate c as the minimum y value
    c_guess = min(y)

    # Filter out non-positive x values and adjust y
    positive_indices = x > 0
    x_pos = x[positive_indices]
    y_adjusted = y[positive_indices] - c_guess

    # Protect against non-positive y_adjusted values
    valid_indices = y_adjusted > 0
    x_valid = x_pos[valid_indices]
    y_valid = y_adjusted[valid_indices]

    if len(x_valid) < 2:
        # Not enough valid points, use default guesses
        return 1.0, 1.0, c_guess

    # Linear fit on log-log scale to estimate a and b
    log_x = np.log(x_valid)
    log_y = np.log(y_valid)

    coeffs = np.polyfit(log_x, log_y, 1)
    b_guess = coeffs[0]
    log_a = coeffs[1]
    a_guess = np.exp(log_a)

    return float(a_guess), float(b_guess), float(c_guess)


def guess_sine(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Generate initial parameter guesses for sine function: f(x) = a * sin(b * x + c) + d"""

    # d is the vertical offset (mean of y)
    d_guess = np.mean(y)

    # a is half the peak-to-peak amplitude
    a_guess = (np.max(y) - np.min(y)) / 2

    # Try to estimate frequency using FFT
    if len(x) > 4:
        # Need enough points for meaningful FFT
        y_centered = y - d_guess

        # If x is not uniformly spaced, interpolate onto uniform grid
        if len(set(np.diff(x))) > 1:  # Check if x spacing is non-uniform
            x_min, x_max = min(x), max(x)
            x_uniform = np.linspace(x_min, x_max, len(x))
            y_interpolated = np.interp(x_uniform, x, y_centered)

            # Perform FFT on interpolated data
            fft_values = np.fft.rfft(y_interpolated)
            fft_freqs = np.fft.rfftfreq(len(x_uniform), (x_max - x_min) / len(x_uniform))

            # Find dominant frequency (skip DC component at index 0)
            if len(fft_values) > 1:
                dominant_idx = np.argmax(np.abs(fft_values[1:])) + 1
                dominant_freq = fft_freqs[dominant_idx]
                b_guess = 2 * np.pi * dominant_freq
            else:
                b_guess = 2 * np.pi / (max(x) - min(x))
        else:
            # Uniform spacing
            fft_values = np.fft.rfft(y_centered)
            fft_freqs = np.fft.rfftfreq(len(x), x[1] - x[0])

            # Find dominant frequency (skip DC component)
            if len(fft_values) > 1:
                dominant_idx = np.argmax(np.abs(fft_values[1:])) + 1
                dominant_freq = fft_freqs[dominant_idx]
                b_guess = 2 * np.pi * dominant_freq
            else:
                b_guess = 2 * np.pi / (max(x) - min(x))
    else:
        # Not enough points for FFT, use simple estimate
        b_guess = 2 * np.pi / (max(x) - min(x))

    # Phase estimate (c) - try a few values and see which gives best match
    phase_candidates = np.linspace(0, 2 * np.pi, 8)
    best_score = float("inf")
    c_guess = 0

    for phase in phase_candidates:
        test_y = a_guess * np.sin(b_guess * x + phase) + d_guess
        score = np.sum((test_y - y) ** 2)
        if score < best_score:
            best_score = score
            c_guess = phase

    return float(a_guess), float(b_guess), float(c_guess), float(d_guess)


def guess_cosine(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Generate initial parameter guesses for cosine function: f(x) = a * cos(b * x + c) + d"""

    # Cosine is just a phase-shifted sine, so we can use the sine guesser
    a_guess, b_guess, c_sine_guess, d_guess = guess_sine(x, y)

    # Adjust phase for cosine (cos(x) = sin(x + pi/2))
    c_guess = c_sine_guess - np.pi / 2

    return a_guess, b_guess, c_guess, d_guess


def guess_damped_sine(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, float]:
    """Generate initial parameter guesses for damped sine function: f(x) = a * exp(-b * x) * sin(c * x + d) + e"""

    # e is the asymptotic value, approximately min(y) or mean of last few points
    if len(y) > 10:
        e_guess = np.mean(y[-5:])  # Mean of last 5 points
    else:
        e_guess = np.mean(y)  # Overall mean as fallback

    # Center the data
    y_centered = y - e_guess

    # Estimate envelope using peaks
    # First get absolute values
    y_abs = np.abs(y_centered)

    # Try to find local maxima as an envelope
    peak_indices = []
    for i in range(1, len(y_abs) - 1):
        if y_abs[i] > y_abs[i - 1] and y_abs[i] > y_abs[i + 1]:
            peak_indices.append(i)

    # If we found at least 2 peaks, estimate damping factor
    if len(peak_indices) >= 2:
        x_peaks = x[peak_indices]
        y_peaks = y_abs[peak_indices]

        # Ensure positive values for log
        positive_indices = y_peaks > 0
        x_peaks_pos = x_peaks[positive_indices]
        y_peaks_pos = y_peaks[positive_indices]

        if len(x_peaks_pos) >= 2:
            # Log transform to estimate exponential decay
            log_y_peaks = np.log(y_peaks_pos)
            slope, log_a = np.polyfit(x_peaks_pos, log_y_peaks, 1)

            a_guess = np.exp(log_a)  # Initial amplitude
            b_guess = -slope  # Damping factor (note: negative slope for decay)
        else:
            # Default guesses if not enough peaks
            a_guess = np.max(np.abs(y_centered))
            b_guess = 1.0 / (max(x) - min(x))
    else:
        # Default guesses if peak finding fails
        a_guess = np.max(np.abs(y_centered))
        b_guess = 1.0 / (max(x) - min(x))

    # Estimate frequency using the sine guesser but on the normalized data
    if b_guess > 0:
        y_normalized = y_centered / np.exp(-b_guess * x)
    else:
        y_normalized = y_centered

    # Use sine frequency estimation
    _, c_guess, d_guess, _ = guess_sine(x, y_normalized)

    # Make sure amplitude is positive
    if a_guess < 0:
        a_guess = -a_guess
        d_guess += np.pi  # Adjust phase

    return float(a_guess), float(b_guess), float(c_guess), float(d_guess), float(e_guess)


def guess_gaussian(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Generate initial parameter guesses for Gaussian function: f(x) = a * exp(-(x-mu)^2 / (2*sigma^2)) + c"""

    # c is the baseline, estimate as minimum y
    c_guess = min(y)

    # Adjust y by subtracting baseline
    y_adjusted = y - c_guess

    # a is the peak height above baseline
    a_guess = max(y_adjusted)

    # mu is the x position of the peak
    peak_idx = np.argmax(y_adjusted)
    mu_guess = x[peak_idx]

    # Estimate sigma by finding the half maximum points
    half_max = a_guess / 2

    # Find points closest to half maximum
    above_half_max = y_adjusted >= half_max

    if np.sum(above_half_max) > 1:
        # Find leftmost and rightmost points above half max
        indices = np.where(above_half_max)[0]
        leftmost = indices[0]
        rightmost = indices[-1]

        # If these are not the exact half max points, interpolate
        if leftmost > 0:  # Can interpolate left
            left_idx = leftmost - 1
            left_x = x[left_idx]
            right_x = x[leftmost]
            left_y = y_adjusted[left_idx]
            right_y = y_adjusted[leftmost]

            # Linear interpolation to find half max x position
            if right_y != left_y:  # Avoid division by zero
                x_left_half = left_x + (right_x - left_x) * (half_max - left_y) / (right_y - left_y)
            else:
                x_left_half = left_x
        else:
            x_left_half = x[leftmost]

        if rightmost < len(x) - 1:  # Can interpolate right
            left_idx = rightmost
            right_idx = rightmost + 1
            left_x = x[left_idx]
            right_x = x[right_idx]
            left_y = y_adjusted[left_idx]
            right_y = y_adjusted[right_idx]

            # Linear interpolation to find half max x position
            if right_y != left_y:  # Avoid division by zero
                x_right_half = left_x + (right_x - left_x) * (half_max - left_y) / (right_y - left_y)
            else:
                x_right_half = right_x
        else:
            x_right_half = x[rightmost]

        # FWHM = 2.355 * sigma for Gaussian
        fwhm = abs(x_right_half - x_left_half)
        sigma_guess = fwhm / 2.355
    else:
        # Fallback if we can't estimate FWHM
        sigma_guess = (max(x) - min(x)) / 5

    # Make sure sigma is positive and not too small
    sigma_guess = max(sigma_guess, (max(x) - min(x)) / 100)

    return float(a_guess), float(mu_guess), float(sigma_guess), float(c_guess)


def guess_lorentzian(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Generate initial parameter guesses for Lorentzian function: f(x) = a * gamma^2 / ((x - x0)^2 + gamma^2) + c"""

    # c is the baseline, estimate as minimum y
    c_guess = min(y)

    # Adjust y by subtracting baseline
    y_adjusted = y - c_guess

    # x0 is the x position of the peak
    peak_idx = np.argmax(y_adjusted)
    x0_guess = x[peak_idx]

    # a is the peak height above baseline (note: for Lorentzian, peak value is a + c)
    a_guess = max(y_adjusted)

    # Estimate gamma by finding the half maximum points
    half_max = a_guess / 2

    # Process similar to Gaussian FWHM estimation
    above_half_max = y_adjusted >= half_max

    if np.sum(above_half_max) > 1:
        indices = np.where(above_half_max)[0]
        leftmost = indices[0]
        rightmost = indices[-1]

        # Simple interpolation for half max points
        if leftmost > 0:
            left_x = x[leftmost - 1]
            right_x = x[leftmost]
            left_y = y_adjusted[leftmost - 1]
            right_y = y_adjusted[leftmost]

            if right_y != left_y:
                x_left_half = left_x + (right_x - left_x) * (half_max - left_y) / (right_y - left_y)
            else:
                x_left_half = left_x
        else:
            x_left_half = x[leftmost]

        if rightmost < len(x) - 1:
            left_x = x[rightmost]
            right_x = x[rightmost + 1]
            left_y = y_adjusted[rightmost]
            right_y = y_adjusted[rightmost + 1]

            if right_y != left_y:
                x_right_half = left_x + (right_x - left_x) * (half_max - left_y) / (right_y - left_y)
            else:
                x_right_half = right_x
        else:
            x_right_half = x[rightmost]

        # For Lorentzian, FWHM = 2*gamma
        fwhm = abs(x_right_half - x_left_half)
        gamma_guess = fwhm / 2
    else:
        # Fallback if we can't estimate FWHM
        gamma_guess = (max(x) - min(x)) / 10

    # Make sure gamma is positive and not too small
    gamma_guess = max(gamma_guess, (max(x) - min(x)) / 100)

    return float(a_guess), float(x0_guess), float(gamma_guess), float(c_guess)


def guess_voigt(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, float]:
    """Generate initial parameter guesses for Voigt function (Gaussian-Lorentzian convolution)"""

    # Start with Gaussian parameters
    a_gauss, mu_guess, sigma_guess, c_guess = guess_gaussian(x, y)

    # For Voigt, we need both Gaussian width (sigma) and Lorentzian width (gamma)
    # Start with a ratio that favors Gaussian slightly
    gamma_guess = sigma_guess * 0.5

    # Adjust amplitude for Voigt profile
    # Voigt amplitude needs adjustment compared to pure Gaussian
    a_guess = a_gauss * sigma_guess * np.sqrt(2 * np.pi)

    return a_guess, mu_guess, sigma_guess, gamma_guess, c_guess


MODELS = {
    "Linear": (linear, "y = m·x + b", guess_linear),
    "Quadratic": (quadratic, "y = a·x² + b·x + c", guess_quadratic),
    "Cubic": (cubic, "y = a·x³ + b·x² + c·x + d", guess_cubic),
    "Exponential Growth": (exponential, "y = a·e<sup>b·x</sup> + c", guess_exponential),
    "Exponential Decay": (exponential_decay, "y = a·e<sup>−b·x</sup> + c", guess_exponential_decay),
    "Logarithmic": (logarithmic, "y = a·ln(b·x) + c", guess_logarithmic),
    "Power Law": (power_law, "y = a·x<sup>b</sup> + c", guess_power_law),
    "Sine": (sine, "y = a·sin(b·x + c) + d", guess_sine),
    "Cosine": (cosine, "y = a·cos(b·x + c) + d", guess_cosine),
    "Damped Sine": (damped_sine, "y = a·e<sup>−b·x</sup>·sin(c·x + d) + e", guess_damped_sine),
    "Gaussian": (gaussian, "y = a·e<sup>−(x − μ)² / (2·σ²)</sup> + c", guess_gaussian),
    "Lorentzian": (lorentzian, "y = a·γ² / ((x − x₀)² + γ²) + c", guess_lorentzian),
    "Voigt": (voigt, "y = a·Re[wofz((x − μ + i·γ) / (σ·√2))] / (σ·√2π) + c", guess_voigt),
}
