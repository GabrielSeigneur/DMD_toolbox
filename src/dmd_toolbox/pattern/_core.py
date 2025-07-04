import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import RectBivariateSpline

from dmd_toolbox.pattern.dataloader import DMDPadding

ERROR_DIFFUSION_MATRIX = [
    (0, -1, 7 / 16),
    (-1, 0, 5 / 16),
    (-1, 1, 3 / 16),
    (-1, -1, 1 / 16),
]

# workaround to allow to run in interactive window
ROOT = Path(os.getcwd())
if all(item in ROOT.parts for item in ["src", "dmd_toolbox"]):
    ROOT = ROOT.parent.parent





class analogDMDPattern:
    """Type for analog DMD patterns.
    This class is meant to handle pattern NumPy arrays that are in greyscale, with float values between 0 and 1."""

    def __init__(self, pattern_array: np.ndarray):
        """Initialize the analogDMDPattern with a numpy array representing the pattern.
        Parameters
        ----------
        pattern_array : np.ndarray
            Numpy array representing the DMD pattern. It should be a 2D array with values between 0 and 1.
        Raises
        ------

        ValueError
            If pattern_array is not a 2D array or if its values are not in the range [0, 1].
        Notes
        -----
        The pattern is expected to be a 2D numpy array with float values between 0 and 1,
        where 0 represents an OFF mirror and 1 represents an ON mirror.
        The pattern can be used as an input to the generation of binaryMask object,
        that would eventually be sent over to a DMD.
        """

        if pattern_array.ndim != 2:
            raise ValueError("Pattern must be a 2D array.")
        if not np.all((pattern_array >= 0) & (pattern_array <= 1)):
            raise ValueError("Pattern values must be between 0 and 1.")

        self.pattern = pattern_array

    def correct_distortion(self, alpha: float, beta: float, gamma: float):
        """Apply distortion correction to the pattern, and recenter the corrected pattern to the center of the image.
        The method uses 2D interpolation to correct the distortion based on the provided coefficients.
        EDIT: Commented code might work and might be more readable. I'll try to implement it later.

        Parameters
        ----------
        alpha, beta, gamma: float
            Distortion coefficients (see DMD Wiki on Notion page for more details).
        """

        # Interpolate the pattern with scipy.interpolate.RectBivariateSpline
        h, w = self.pattern.shape
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        X, Y = np.meshgrid(x, y)
        spline = RectBivariateSpline(y, x, self.pattern)

        # Create a grid of corrected coordinates
        h, w = self.pattern.shape
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        X_corr = (
            np.tan(beta)
            / (1 + np.tan(beta) * np.tan(alpha))
            * (X / np.sin(beta) - Y / np.cos(alpha))
        )
        Y_corr = (
            np.tan(alpha)
            / (1 + np.tan(beta) * np.tan(alpha))
            * (X / np.cos(beta) + Y / np.sin(alpha))
        )
        Y_corr = Y_corr / gamma  # Apply the gamma correction

        # Ensure the corrected coordinates are within the bounds of the original pattern
        X_corr = np.clip(X_corr, 0, 1)
        Y_corr = np.clip(Y_corr, 0, 1)

        # Evaluate the spline at the corrected coordinates
        corrected_pattern = spline.ev(Y_corr.flatten(), X_corr.flatten()).reshape(h, w)

        # Clip the values to ensure they are between 0 and 1
        corrected_pattern = np.clip(corrected_pattern, 0, 1)

        ## Recenter the corrected pattern to the center of the image
        # Compute center of mass of array
        center_x = np.sum(np.arange(w) * np.sum(corrected_pattern, axis=0)) / np.sum(
            corrected_pattern
        )
        center_y = np.sum(np.arange(h) * np.sum(corrected_pattern, axis=1)) / np.sum(
            corrected_pattern
        )
        roll_axis = (int(h / 2 - center_y), int(w / 2 - center_x))

        # Roll the array to recenter it
        corrected_pattern = np.roll(corrected_pattern, roll_axis, axis=(0, 1))

        self.pattern_corr = corrected_pattern
        print("Distortion correction applied successfully.")





class binaryMask:
    """Type for binary masks ready to send to the DMD. Calibrates, corrects for distortion and performs error-diffusion
    on the input analog pattern inset.
    Note that the analog pattern should be a 2D numpy array with values between 0 and 0.9.
    """

    def __init__(
        self,
        analog_pattern_inset: np.ndarray,
        padding_dim: tuple = (0, 0, 0, 0),
        DMD_height: int = 1080,
        DMD_width: int = 1920,
        alpha: float = np.deg2rad(-4.5),
        beta: float = np.deg2rad(-2.5),
        gamma: float = 1.09,
    ):
        """Initialize the binaryMask with an analog pattern inset and optional parameters for
        padding and distortion correction.
        Parameters
        ----------
        analog_pattern_inset : np.ndarray
            2D numpy array representing the analog pattern inset. Values should be between 0 and 0.9.
        padding_dim : tuple, optional
            Tuple of (top, bottom, left, right) padding values. Default is (0, 0, 0, 0).
        DMD_height : int, optional
            Height of the DMD pattern. Default is 1080.
        DMD_width : int, optional
            Width of the DMD pattern. Default is 1920.
        alpha, beta, gamma: float, optional
            Distortion coefficients for the DMD pattern correction.
            Default values are -4.5 degrees for alpha, -2.5 degrees for beta, and 1.09 for gamma.
        """
        self.analog_pattern = analogDMDPattern(analog_pattern_inset)
        self.padding = DMDPadding(padding_dim, DMD_height, DMD_width)
        self.analog_pattern.correct_distortion(
            alpha, beta, gamma
        )  # Apply distortion correction to the analog pattern

    def calibrate_pattern(self, A, B):
        """Calibrate the analog pattern using the coefficients A and B.
        The calibration is done by solving
        the quadratic equation A*x^2 + B*x - y = 0 for each pixel in the analog pattern.

        Parameters
        ----------
        A, B : float
            Coefficients for the quadratic equation used in the calibration.
        """

        img = self.analog_pattern.pattern_corr
        calibrated_image = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                y = img[i, j]
                roots = np.roots([A, B, -y])
                # Return the root that is between 0 and 1
                for root in roots:
                    if root > 0 and root < 1:
                        calibrated_image[i, j] = root
                        break

        self.calibrated_pattern = calibrated_image

    def perform_error_diffusion(
        self,
        ref_image_analog,
        error_diffusion_matrix=ERROR_DIFFUSION_MATRIX,
        save_path: str | Path= "ED Patterns",
    ):
        """ "Perform error diffusion on the calibrated pattern.
        The error diffusion is done using the specified error diffusion matrix,
        which defines the weights and directions of the error diffusion process.
        The reference image is expected to be a 2D numpy array with values between 0 and 0.9,
        representing the analog pattern.
        A stochastic noise is added to the error diffusion process to break the symmetry and avoid artifacts.

        Parameters
        ----------
        ref_image_analog : np.ndarray
            Reference image for error diffusion.

        error_diffusion_matrix : list of tuples, optional
            Error diffusion matrix to use. Default is ERROR_DIFFUSION_MATRIX.

        save_path : str or Path, optional
            Path to save the resulting binary mask. Default is "./ED Patterns/".
            The path should be taken from project root.
        Raises
        ------
        ValueError
            If the reference image is not calibrated or if it does not have positive values for error diffusion.
        TypeError
            If the reference image is not a numpy array or if it is not 2D.
        """

        if not hasattr(self, "calibrated_pattern"):
            raise ValueError(
                "Pattern must be calibrated before performing error diffusion."
            )

        # Re-scale to 0-0.9 range
        scaling_factor = np.max(ref_image_analog) / 0.9
        if scaling_factor <= 0:
            raise ValueError(
                "Reference image must have positive values for error diffusion."
            )

        ref_image_analog = ref_image_analog / scaling_factor

        ref_image_analog = self.padding.apply_padding(
            self.calibrated_pattern
        )  # Apply padding to the reference image BEFORE ED (smoother result)
        Npx, Mpx = ref_image_analog.shape
        q = np.round(ref_image_analog, 0)  # Initialise the 1-bit image
        v = np.zeros_like(
            ref_image_analog
        )  # store the error values during the diffusion process

        for i in range(Npx):  # iterating through y axis
            for j in range(Mpx):  # iteration through x axis
                v_sum = 0

                for di, dj, w in error_diffusion_matrix:
                    ni = (
                        i + di
                    )  # computing the coordinates of the neighbours in the direction (di, dj)
                    nj = j + dj
                    if ni > 0 and nj < Mpx:  # valid bounds
                        v_sum += (
                            v[ni, nj] * w
                        )  # add error from the neighbours, weighted by the weight in error_diffusion_matrix
                if (
                    v_sum + np.random.rand() < 0.5
                ):  # add a random noise to the error diffusion process
                    q[i, j] = 0
                else:
                    q[i, j] = 1
                v[i, j] = v_sum + ref_image_analog[i, j] - q[i, j]

        # If directory does not exist, create it
        save_path = ROOT.joinpath(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the binary mask as a BMP image
        Image.fromarray((q * 255).astype(np.uint8)).save(
            os.path.join(save_path)
        )  # ready to be processed by the DMD's software
        self.binary_mask = q


# if __name__ == "__main__":
