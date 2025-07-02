from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image


# Example for subclassing an ndarray found here:
# https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
class analogBWImage(np.ndarray):
    """An image class representing a grayscale image normalized to [0, 1].

    Can be constructed from a NumPy array or from a file path. RGB images are converted to grayscale.
    """

    def __new__(
        cls,
        img_array: Optional[np.ndarray] = None,
        img_path: Optional[Union[str, Path]] = None,
        root_dir: Optional[Union[str, Path]] = None,
    ):
        # Input validation
        if (img_array is None) == (img_path is None):
            raise ValueError("Provide exactly one of img_array or img_path.")

        if img_array is None and img_path is not None:
            root_dir = Path(root_dir or Path.cwd())
            img_path = Path(img_path)
            full_path = root_dir / img_path
            if not full_path.is_file():
                raise FileNotFoundError(f"Image file {full_path} does not exist.")
            img_array = np.array(Image.open(full_path))

        if not isinstance(img_array, np.ndarray):
            raise TypeError("img_array must be a numpy array.")

        if img_array.ndim not in (2, 3):
            raise ValueError("Image must be 2D or 3D (grayscale or RGB).")

        # Convert RGB to grayscale by taking the red channel
        if img_array.ndim == 3 and img_array.shape[2] == 3:
            img_array = img_array[:, :, 0]

        # Normalize to [0, 1]
        img_array = img_array.astype(np.float32)
        min_val, max_val = img_array.min(), img_array.max()
        img_array = (img_array - min_val) / (max_val - min_val)

        # Create ndarray instance
        obj = np.asarray(img_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        pass

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
