
import numpy as np


def quadratic(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Quadratic function for fitting."""
    return a * x**2 + b * x


def gaussian(array, sigma):
    """Returns a 2D Gaussian of standard deviation sigma, centered on the center of the array."""
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if array.ndim != 2:
        raise ValueError("Input array must be 2D.")

    h, w = array.shape
    x = np.linspace(-w // 2, w // 2, w)
    y = np.linspace(-h // 2, h // 2, h)
    X, Y = np.meshgrid(x, y)

    return np.exp(-(X**2 + Y**2) / (2 * sigma**2))
