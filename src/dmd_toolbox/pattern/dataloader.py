from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
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



class DMDPadding:
    """
    Padding for DMD patterns.
    The padding is defined as a tuple of (top, bottom, left, right),
    and is meant to be fully illuminated by the incident beam.
    Careful about the Y axis: it is point down, so top is the first row and bottom is the last row.
    """

    def __init__(
        self, padding_dim: tuple = (0, 0, 0, 0), height: int = 1080, width: int = 1920
    ):
        """
        Initialize the DMD padding with specified dimensions and padding values.

        Parameters
        ----------
        padding_dim : tuple, optional
            Tuple of (top, bottom, left, right) padding values.
        height : int, optional
            Height of the DMD pattern.
        width : int, optional
            Width of the DMD pattern.
        """

        if not all(isinstance(x, int) for x in [*padding_dim]):
            raise TypeError("Padding values must be integers.")

        self.padding = padding_dim
        self.height = height
        self.width = width

        padding_image = np.zeros((height, width), dtype=np.bool_)
        padding_image[
            padding_dim[0] : height - padding_dim[1],
            padding_dim[2] : width - padding_dim[3],
        ] = True
        self.padding_image = padding_image

    def apply_padding(self, pattern: np.ndarray) -> np.ndarray:
        """Apply padding to a given pattern.
        Parameters
        ----------
        pattern : np.ndarray
            2D numpy array representing the pattern to be padded.

        Returns
        -------
        np.ndarray
            Padded pattern with the same dimensions as the DMD.
        """
        if not isinstance(pattern, np.ndarray):
            raise TypeError("Pattern must be a numpy array.")
        if pattern.ndim != 2:
            raise ValueError("Pattern must be a 2D array.")
        if (
            pattern.shape[0] != self.height - self.padding[0] - self.padding[1]
            or pattern.shape[1] != self.width - self.padding[2] - self.padding[3]
        ):
            raise ValueError("Pattern dimensions must match the DMD dimensions.")

        padded_pattern = np.zeros((self.height, self.width), dtype=pattern.dtype)
        top, bottom, left, right = self.padding
        padded_pattern[top : self.height - bottom, left : self.width - right] = pattern
        return padded_pattern

    def plot_padding(self):
        """Plot the padding image."""
        plt.imshow(self.padding_image, cmap="gray")
        plt.title("DMD Padding")
        plt.axis("off")
        plt.show()
