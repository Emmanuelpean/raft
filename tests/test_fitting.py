"""Test module for the functions in the `fitting.py` module.

This module contains unit tests for the functions implemented in the `fitting.py` module. The purpose of these tests is to
ensure the correct functionality of each function in different scenarios and to validate that the expected outputs are
returned.

Tests should cover various edge cases, valid inputs, and any other conditions that are necessary to confirm the
robustness of the functions."""

import pytest

from app.fitting import *
from app.utils import are_close


def generate_expected_code(fit_output: list):  # pragma: no cover
    """Generate Python code that defines a dictionary called 'expected'
    using values from a list-based fit_output structure."""
    params = fit_output[0]
    param_errors = fit_output[1]
    r_squared = fit_output[3]

    code = (
        "expected = {\n"
        f"    'params': [{', '.join(f'{p:.8f}' for p in params)}],\n"
        f"    'param_errors': [{', '.join(f'{e:.8f}' for e in param_errors)}],\n"
        f"    'r_squared': {r_squared:.16f}\n"
        "}"
    )
    print(code)


class TestFitData:
    """Test class for fit_data function"""

    @staticmethod
    def generate_noisy_data(
        func: callable,
        params: dict[str, float],
        noise_level: float = 0.1,
        x0: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data with noise for testing"""

        np.random.seed(42)
        x_data = np.linspace(x0, 10, 50)
        y_true = func(x_data, **params)
        noise = np.random.normal(0, noise_level, len(x_data))
        y_noisy = y_true + noise
        return x_data, y_noisy

    @staticmethod
    def assert_fit_quality(
        fit_output: tuple,
        expected: dict,
    ) -> None:
        """Check if the fit is of acceptable quality"""

        assert are_close(fit_output[0], expected["params"])
        assert are_close(fit_output[1], expected["param_errors"])
        assert are_close(fit_output[3], expected["r_squared"])

    def test_linear_fit(self) -> None:
        """Test fit with linear function"""

        true_params = dict(m=2.5, b=3.0)  # m, b for y = m*x + b
        x_data, y_data = self.generate_noisy_data(linear, true_params, noise_level=0.5)
        fit_output = fit_data(x_data, y_data, linear, true_params)
        expected = {
            "params": [2.4710083, 3.03222155],
            "param_errors": [0.02225959, 0.12916986],
            "r_squared": 0.9961199353002853,
        }
        self.assert_fit_quality(fit_output, expected)

    def test_quadratic_fit(self) -> None:
        """Test fit with quadratic function"""

        true_params = dict(a=0.5, b=-2.0, c=5.0)  # a, b, c for y = a*x^2 + b*x + c
        x_data, y_data = self.generate_noisy_data(quadratic, true_params)
        fit_output = fit_data(x_data, y_data, quadratic, true_params)
        expected = {
            "params": [0.50214157, -2.02721402, 5.04140868],
            "param_errors": [0.00168019, 0.01737441, 0.03756755],
            "r_squared": 0.9999141697830860,
        }
        self.assert_fit_quality(fit_output, expected)

    def test_cubic_fit(self) -> None:
        """Test fit with quadratic function"""

        true_params = dict(a=0.5, b=-2.0, c=5.0, d=3.0)
        x_data, y_data = self.generate_noisy_data(cubic, true_params)
        fit_output = fit_data(x_data, y_data, cubic, true_params)
        expected = {
            "params": [0.49910268, -1.98439861, 4.91948859, 3.08356523],
            "param_errors": [0.00064418, 0.00980492, 0.04195150, 0.04795362],
            "r_squared": 0.9999992254644585,
        }
        self.assert_fit_quality(fit_output, expected)

    def test_exponential_fit(self) -> None:
        """Test fit with exponential function"""

        true_params = dict(a=2.0, b=0.3, c=1.0)  # a, b, c for y = a*exp(b*x) + c
        x_data, y_data = self.generate_noisy_data(exponential, true_params)
        fit_output = fit_data(x_data, y_data, exponential, true_params)
        expected = {
            "params": [1.97780830, 0.30098598, 1.02877415],
            "param_errors": [0.02339381, 0.00115386, 0.04800676],
            "r_squared": 0.9999267015059259,
        }
        self.assert_fit_quality(fit_output, expected)

    def test_exponential_decay_fit(self) -> None:
        """Test fit with exponential decay function"""

        true_params = dict(a=5.0, b=0.4, c=1.0)  # a, b, c for y = a*exp(-b*x) + c
        x_data, y_data = self.generate_noisy_data(exponential_decay, true_params)
        fit_output = fit_data(x_data, y_data, exponential_decay, true_params)
        expected = {
            "params": [5.10309909, 0.41001656, 0.97951783],
            "param_errors": [0.04980599, 0.01007091, 0.03054609],
            "r_squared": 0.9957287333194214,
        }
        self.assert_fit_quality(fit_output, expected)

    def test_logarithmic_fit(self) -> None:
        """Test fit with exponential decay function"""

        true_params = dict(a=5.0, b=0.4, c=1.0)
        x_data, y_data = self.generate_noisy_data(logarithmic, true_params, x0=3)
        fit_output = fit_data(x_data, y_data, logarithmic, true_params)
        expected = {
            "params": [4.94206020, 0.35260880, 1.65281069],
            "param_errors": [0.03834814, 51455.30413448, 721385.85497252],
            "r_squared": 0.9971793561422160,
        }
        self.assert_fit_quality(fit_output, expected)

    def test_power_law_fit(self) -> None:
        """Test fit with power_law function"""

        true_params = dict(a=5.0, b=0.4, c=1.0)
        x_data, y_data = self.generate_noisy_data(power_law, true_params)
        fit_output = fit_data(x_data, y_data, power_law, true_params)
        expected = {
            "params": [4.91893085, 0.40271900, 1.08348238],
            "param_errors": [0.08428910, 0.00510595, 0.08454555],
            "r_squared": 0.9990014370788753,
        }
        self.assert_fit_quality(fit_output, expected)

    def test_sine_fit(self) -> None:
        """Test fit with sine function"""

        true_params = dict(a=5.0, b=0.4, c=1.0, d=4.0)
        x_data, y_data = self.generate_noisy_data(sine, true_params)
        fit_output = fit_data(x_data, y_data, sine, true_params)
        expected = {
            "params": [5.02212548, 0.39557220, 1.02793337, 3.98198780],
            "param_errors": [0.01821233, 0.00257973, 0.01311088, 0.01929356],
            "r_squared": 0.9994882962181407,
        }
        self.assert_fit_quality(fit_output, expected)

    def test_cosine_fit(self) -> None:
        """Test fit with cosine function"""

        true_params = dict(a=5.0, b=0.4, c=1.0, d=4.0)
        x_data, y_data = self.generate_noisy_data(cosine, true_params)
        fit_output = fit_data(x_data, y_data, cosine, true_params)
        expected = {
            "params": [5.19732638, 0.39104084, 1.04640785, 4.16582480],
            "param_errors": [0.17390120, 0.00928806, 0.05053037, 0.18598331],
            "r_squared": 0.9985667355230208,
        }
        self.assert_fit_quality(fit_output, expected)

    def test_damped_sine_fit(self) -> None:
        """Test fit with damped_sine function"""

        true_params = dict(a=5.0, b=0.4, c=1.0, d=4.0, e=1.54)
        x_data, y_data = self.generate_noisy_data(damped_sine, true_params)
        fit_output = fit_data(x_data, y_data, damped_sine, true_params)
        expected = {
            "params": [4.86161789, 0.39911196, 0.99002641, 4.01660042, 1.50702147],
            "param_errors": [0.07294701, 0.00944758, 0.01027530, 0.02370077, 0.01469157],
            "r_squared": 0.9962838316988460,
        }
        self.assert_fit_quality(fit_output, expected)

    def test_gaussian_fit(self) -> None:
        """Test fit with gaussian function"""

        true_params = dict(a=3.0, mu=5.0, sigma=1.6, c=0.6)
        x_data, y_data = self.generate_noisy_data(gaussian, true_params)
        fit_output = fit_data(x_data, y_data, gaussian, true_params)
        expected = {
            "params": [2.95988308, 5.01227551, 1.56034143, 0.62159483],
            "param_errors": [0.03763067, 0.01889967, 0.02853320, 0.02939347],
            "r_squared": 0.9927040354949260,
        }
        self.assert_fit_quality(fit_output, expected)

    def test_lorentzian_fit(self) -> None:
        """Test fit with lorentzian function"""

        true_params = dict(a=3.0, x0=5.0, gamma=1.6, c=0.6)
        x_data, y_data = self.generate_noisy_data(lorentzian, true_params)
        fit_output = fit_data(x_data, y_data, lorentzian, true_params)
        expected = {
            "params": [2.96067289, 5.00927353, 1.56204167, 0.61483302],
            "param_errors": [0.04603849, 0.02048091, 0.05274112, 0.03901545],
            "r_squared": 0.9890805599836860,
        }
        self.assert_fit_quality(fit_output, expected)

    def test_voigt_fit(self) -> None:
        """Test fit with voigt function"""

        true_params = dict(a=5.0, mu=2.0, sigma=0.5, gamma=1.6, c=0.6)
        x_data, y_data = self.generate_noisy_data(voigt, true_params)
        fit_output = fit_data(x_data, y_data, voigt, true_params)
        expected = {
            "params": [5.65851920, 1.78835872, 0.00011284, 1.89123188, 0.55631642],
            "param_errors": [0.53803494, 0.07415275, 206.22004309, 0.17031415, 0.02868130],
            "r_squared": 0.9324065583778361,
        }
        self.assert_fit_quality(fit_output, expected)

    def test_bounded_fit(self) -> None:
        """Test fit with exponential decay function and bounds"""

        true_params = dict(a=5.0, b=0.4, c=1.0)
        bounds = dict(a=[0, 100], b=[0, 1], c=[0, 100000])
        x_data, y_data = self.generate_noisy_data(exponential_decay, true_params)
        fit_output = fit_data(x_data, y_data, exponential_decay, true_params, bounds)
        expected = {
            "params": [5.10309909, 0.41001656, 0.97951782],
            "param_errors": [0.04980600, 0.01007092, 0.03054605],
            "r_squared": 0.9957287333194214,
        }
        self.assert_fit_quality(fit_output, expected)

    def test_error_fit(self) -> None:
        """Test fit with exponential decay function and bounds"""

        true_params = dict(a=-5.0, b=0.4, c=1.0)
        bounds = dict(a=[0, 100], b=[0, 1], c=[0, 100000])
        x_data, y_data = self.generate_noisy_data(exponential_decay, true_params)
        with pytest.raises(ValueError):
            fit_data(x_data, y_data, exponential_decay, true_params, bounds)


def print_expected_from_tuple(values: tuple | list) -> None:  # pragma: no cover
    """Prints a list called 'expected' where each element corresponds to the elements in the input tuple."""

    expected = [v.item() if hasattr(v, "item") else v for v in values]
    print(f"expected = {expected}")


class TestParameterGuessFunctions:
    """Test class for parameter guess functions."""

    @staticmethod
    def add_noise(y) -> np.ndarray:
        """Add random noise to the data."""

        np.random.seed(42)
        return y + 0.05 * (np.random.rand(len(y)) - 0.5) * 2 * np.max(np.abs(y))

    def test_guess_linear(self) -> None:
        """Test linear function parameter guessing."""

        # Generate test data with known parameters
        x = np.linspace(-5, 5, 100)
        true_params = (2.5, -3.0)  # m, b
        y_true = linear(x, *true_params)
        y_noisy = self.add_noise(y_true)

        # Get parameter guesses
        guessed_params = guess_linear(x, y_noisy)
        expected = [2.50317630963796, -3.046219847763776]
        assert are_close(guessed_params, expected)

    def test_guess_quadratic(self) -> None:
        """Test quadratic function parameter guessing."""

        # Generate test data with known parameters
        x = np.linspace(-5, 5, 100)
        true_params = (1.2, -0.5, 2.0)  # a, b, c
        y_true = quadratic(x, *true_params)
        y_noisy = self.add_noise(y_true)

        # Get parameter guesses
        guessed_params = guess_quadratic(x, y_noisy)
        expected = [1.1977212133042152, -0.49293014951550784, 1.9164970879102299]
        assert are_close(guessed_params, expected)

    def test_guess_cubic(self) -> None:
        """Test cubic function parameter guessing."""

        # Generate test data with known parameters
        x = np.linspace(-5, 5, 100)
        true_params = (0.1, 0.5, -2.0, 1.0)  # a, b, c, d
        y_true = cubic(x, *true_params)
        y_noisy = self.add_noise(y_true)

        # Get parameter guesses
        guessed_params = guess_cubic(x, y_noisy)
        expected = [0.09754973503815946, 0.4989431713874625, -1.9592297498298472, 0.961274011784455]
        assert are_close(guessed_params, expected)

    def test_guess_exponential(self) -> None:
        """Test exponential function parameter guessing."""

        # Generate test data with known parameters
        x = np.linspace(-5, 5, 100)
        true_params = (2.0, 0.5, 1.0)  # a, b, c
        y_true = exponential(x, *true_params)
        y_noisy = self.add_noise(y_true)

        # Get parameter guesses
        guessed_params = guess_exponential(x, y_noisy)
        expected = [3.201427508097854, 0.36529310839583773, 0.0560041962813711]
        assert are_close(guessed_params, expected)

    def test_guess_exponential_decay(self) -> None:
        """Test exponential decay function parameter guessing."""

        # Use positive x data for exponential decay
        x = np.linspace(0, 5, 100)
        true_params = (5.0, 0.5, 1.0)  # a, b, c
        y_true = exponential_decay(x, *true_params)
        y_noisy = self.add_noise(y_true)

        # Get parameter guesses
        guessed_params = guess_exponential_decay(x, y_noisy)
        expected = [5.288981263447335, 0.5944592864805658, 1.1361727071534533]
        assert are_close(guessed_params, expected)

    def test_guess_logarithmic(self) -> None:
        """Test logarithmic function parameter guessing."""

        x = np.linspace(0.1, 5, 100)
        true_params = (2.0, 1.5, 1.0)  # a, b, c
        y_true = logarithmic(x, *true_params)
        y_noisy = self.add_noise(y_true)

        # Get parameter guesses
        guessed_params = guess_logarithmic(x, y_noisy)
        expected = [1.9998540437170602, 0.392156862745098, 1.796030742119528]
        assert are_close(guessed_params, expected)

    def test_guess_power_law(self) -> None:
        """Test power law function parameter guessing."""

        x = np.linspace(0.1, 5, 100)
        true_params = (2.0, 0.5, 1.0)  # a, b, c
        y_true = power_law(x, *true_params)
        y_noisy = self.add_noise(y_true)

        # Get parameter guesses
        guessed_params = guess_power_law(x, y_noisy)
        expected = [1.3311568784362127, 0.6969832913669517, 1.5638021793771437]
        assert are_close(guessed_params, expected)

    def test_guess_sine(self) -> None:
        """Test sine function parameter guessing."""

        x = np.linspace(0, 4 * np.pi, 100)
        true_params = (2.0, 1.0, 0.5, 1.0)  # a, b, c, d
        y_true = sine(x, *true_params)
        y_noisy = self.add_noise(y_true)

        # Get parameter guesses
        guessed_params = guess_sine(x, y_noisy)
        expected = [2.1030310552598452, 0.9999999999999999, 0.8975979010256552, 1.0006429310979499]
        assert are_close(guessed_params, expected)

    def test_guess_cosine(self) -> None:
        """Test cosine function parameter guessing."""

        x_data = np.linspace(0, 4 * np.pi, 100)
        true_params = (2.0, 1.0, 0.5, 1.0)  # a, b, c, d
        y_true = cosine(x_data, *true_params)
        y_noisy = self.add_noise(y_true)

        # Get parameter guesses
        guessed_params = guess_cosine(x_data, y_noisy)
        expected = [2.0853774361984017, 0.9999999999999999, 0.2243994752564138, 1.0086060525280351]
        assert are_close(guessed_params, expected)

    def test_guess_damped_sine(self) -> None:
        """Test damped sine function parameter guessing."""

        x_data = np.linspace(0, 6 * np.pi, 200)
        true_params = (5.0, 0.2, 1.0, 0.5, 1.0)  # a, b, c, d, e
        y_true = damped_sine(x_data, *true_params)
        y_noisy = self.add_noise(y_true)

        # Get parameter guesses
        guessed_params = guess_damped_sine(x_data, y_noisy)
        expected = [2.7201431883968903, 0.14060258175829796, 1.0, 0.8975979010256552, 1.152434392852364]
        assert are_close(guessed_params, expected)

    def test_guess_gaussian(self) -> None:
        """Test Gaussian function parameter guessing."""

        x = np.linspace(-5, 5, 100)
        true_params = (3.0, 0.5, 1.0, 0.5)  # a, mu, sigma, c
        y_true = gaussian(x, *true_params)
        y_noisy = self.add_noise(y_true)

        # Get parameter guesses
        guessed_params = guess_gaussian(x, y_noisy)
        expected = [3.310420389932966, 0.5555555555555554, 0.9910540130155195, 0.3324788544674132]
        assert are_close(guessed_params, expected)

    def test_guess_lorentzian(self) -> None:
        """Test Lorentzian function parameter guessing."""

        x = np.linspace(-5, 5, 100)
        true_params = (5.0, 0.5, 1.0, 0.5)  # a, x0, gamma, c
        y_true = lorentzian(x, *true_params)
        y_noisy = self.add_noise(y_true)

        # Get parameter guesses
        guessed_params = guess_lorentzian(x, y_noisy)
        expected = [5.258414082616232, 0.5555555555555554, 0.9534689383158621, 0.4577972093700907]
        assert are_close(guessed_params, expected)

    def test_guess_voigt(self) -> None:
        """Test Voigt function parameter guessing."""

        x = np.linspace(-5, 5, 100)
        true_params = (5.0, 0.5, 1.0, 0.5, 0.5)  # a, mu, sigma, gamma, c
        y_true = voigt(x, *true_params)
        y_noisy = self.add_noise(y_true)

        # Get parameter guesses
        guessed_params = guess_voigt(x, y_noisy)
        expected = [4.753454594318573, 0.5555555555555554, 1.2485700770519543, 0.6242850385259772, 0.4543895625514585]
        assert are_close(guessed_params, expected)
