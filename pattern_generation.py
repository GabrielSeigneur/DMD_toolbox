import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import curve_fit

ERROR_DIFFUSION_MATRIX = [
    (0,-1,7/16),
    (-1,0,5/16),
    (-1,1,3/16),
    (-1,-1,1/16)]

def _quadratic(x: np.ndarray, a:float, b:float) -> np.ndarray:
        """Quadratic function for fitting."""
        return a * x**2 + b * x

def _gaussian(array, sigma):
    """Returns a 2D Gaussian of standard deviation sigma, centered on the center of the array."""
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if array.ndim != 2:
        raise ValueError("Input array must be 2D.")
    
    h, w = array.shape
    x = np.linspace(-w//2, w//2, w)
    y = np.linspace(-h//2, h//2, h)
    X, Y = np.meshgrid(x, y)
    
    return np.exp(-(X**2 + Y**2) / (2 * sigma**2))


class DMDPadding:
    """
    Padding for DMD patterns.
    The padding is defined as a tuple of (top, bottom, left, right), and is meant to be fully illuminated by the incident beam.
    Careful about the Y axis: it is point down, so top is the first row and bottom is the last row.
    """
    def __init__(
                self,
                padding_dim: tuple = (0, 0, 0, 0),
                height : int = 1080, width : int = 1920
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
        padding_image[padding_dim[0]:height-padding_dim[1], padding_dim[2]:width-padding_dim[3]] = True
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
        if pattern.shape[0] != self.height-self.padding[0] - self.padding[1] \
            or pattern.shape[1] != self.width - self.padding[2] - self.padding[3]:
            raise ValueError("Pattern dimensions must match the DMD dimensions.")
        
        padded_pattern = np.zeros((self.height, self.width), dtype=pattern.dtype)
        top, bottom, left, right = self.padding
        padded_pattern[top:self.height-bottom, left:self.width-right] = pattern
        return padded_pattern
    
    def plot_padding(self):
        """Plot the padding image."""
        plt.imshow(self.padding_image, cmap='gray')
        plt.title("DMD Padding")
        plt.axis('off')
        plt.show()

class randomPatternSeries:
    """Series of random patterns generated for DMD intensity vs. density of ON mirrors calibration."""
    def __init__(self, num_patterns: int, padding: DMDPadding):
        if not isinstance(num_patterns, int) or num_patterns <= 0:
            raise ValueError("num_patterns must be a positive integer.")
        self.padding = padding
        self.num_patterns = num_patterns
        self.density_array = np.linspace(0, 1, num_patterns)

        self.patterns = np.zeros((num_patterns, self.padding.height, self.padding.width), dtype=np.bool_)
        self.image_path = "./Calibration_Images/" # Where the images will be saved
    
    def build_pattern(self, density: float, dim_tuple: tuple) -> np.ndarray:
        """Generate a random pattern based on the specified density.
        This function creates a 2D numpy array of boolean values, where True represents an ON mirror and False represents an OFF mirror.
        Parameters
        ----------
        density : float
            Density of the pattern, ranging from 0 to 1, where 0 means no mirrors ON and 1 means all mirrors ON.
        dim_tuple : tuple
            Tuple specifying the dimensions of the pattern (height_inset, width_inset).
        
        Returns
        -------
        np.ndarray
            2D numpy array of boolean values representing the pattern.
        
        Raises
        ------
        ValueError
            If density is not between 0 and 1, or if dim_tuple does not have exactly two elements.
        """
        pattern = np.zeros(dim_tuple, dtype=np.bool_)
        for i in range(dim_tuple[0]):
            for j in range(dim_tuple[1]):
                if np.random.rand() < density:
                    pattern[i, j] = True

        return pattern
    
    def generate_patterns(self, save_path: str = './Calibration_Patterns/'):
        """Generate random patterns and optionally save them to files.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the generated patterns. If None, patterns are not saved.
        
        Raises
        ------
        ValueError
            If save_path is not a valid directory or if it does not end with a '/'.
        """
        inset_dimensions = (self.padding.height - self.padding.padding[0] - self.padding.padding[1], self.padding.width - self.padding.padding[2] - self.padding.padding[3])
        print(inset_dimensions)
        for i in range(self.num_patterns):

            pattern_temp = self.build_pattern(self.density_array[i], inset_dimensions)
            self.patterns[i] = self.padding.apply_padding(pattern_temp)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if save_path[-1] != '/':
                save_path += '/'

            if save_path is not None:
                file_path = os.path.join(save_path, f"pattern_{i}.png") 
                Image.fromarray(self.patterns[i].astype(np.uint8) * 255).save(file_path)

class analogBWImage:
    """Type for analog images from an image file or numpy array.
    This class is meant to handle images that are either grayscale (8 bits/pixel) or RGB (24 bits/pixel), like camera images.
    Image should be scaled from 0 to 255 for maximal contrast.
    """
    def __init__(self, img_array: np.ndarray | None = None, img_path: str | None = None):
        """Initialize the analogBWImage with either an image array or a path to an image file.
        Parameters
        ----------
        img_array : np.ndarray, optional
            Numpy array representing the image. If provided, img_path should be None.
        img_path : str, optional
            Path to the image file. If provided, img_array should be None.
        
        Raises
        ------
        ValueError
            If both img_array and img_path are provided or if neither is provided.
        TypeError
            If img_array is not a numpy array or if img_path is not a string.
        FileNotFoundError
            If the specified image file does not exist.
        ValueError
            If the image is not 2D or 3D, or if it is not in the range [0, 1].
        
        Notes
        -----
        The image is converted to grayscale if it is RGB (24 bits/pixel) by selecting the red channel.
        The image is normalized to the range [0, 1] by dividing by 255.0.
        If the image is already in grayscale (8 bits/pixel), it is assumed to be in the range [0, 255] and is converted to [0, 1].
        If the image is in RGB format, it is converted to grayscale by selecting the red channel.
        If the image is in grayscale format, it is assumed to be in the range [0, 255] and is converted to [0, 1].
        """
        if img_array is not None and img_path is not None:
            raise ValueError("Provide either img_array or img_path, not both.")
        if img_array is None and img_path is None:
            raise ValueError("One of img_array or img_path must be provided.")
        
        if img_array is not None:
            if not isinstance(img_array, np.ndarray):
                raise TypeError("img_array must be a numpy array.")
            self.img = img_array
        elif img_path is not None:
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image file {img_path} does not exist.")
            self.img = np.array(Image.open(img_path))
        
        if self.img.ndim != 2 and self.img.ndim != 3:
            raise ValueError("Image must be 2D or 3D (grayscale or RGB).")
        
        # If image is RGB (24 bits/pixel), convert to grayscale (8 bits/pixel) by selecting the red channel
        if self.img.ndim == 3 and self.img.shape[2] == 3:
            self.img = self.img[:, :, 0]

        # Make sure that the image is in the range [0, 1]
        self.img = self.img.astype(np.float32) / 255.0

class analogDMDPattern:
    """Type for analog DMD patterns.
    This class is meant to handle pattern NumPy arrays that are in greyscale, with float values between 0 and 1. """
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
        The pattern is expected to be a 2D numpy array with float values between 0 and 1, where 0 represents an OFF mirror and 1 represents an ON mirror.
        The pattern can be used as an input to the generation of binaryMask object, that would eventually be sent over to a DMD.
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
        X_corr = np.tan(beta)/(1 + np.tan(beta)*np.tan(alpha)) * ( X / np.sin(beta) - Y / np.cos(alpha))
        Y_corr = np.tan(alpha)/(1 + np.tan(beta)*np.tan(alpha)) * ( X / np.cos(beta) + Y / np.sin(alpha))
        Y_corr = Y_corr / gamma  # Apply the gamma correction
        
        # Ensure the corrected coordinates are within the bounds of the original pattern
        X_corr = np.clip(X_corr, 0, 1)
        Y_corr = np.clip(Y_corr, 0, 1)

        # Evaluate the spline at the corrected coordinates
        corrected_pattern = spline.ev(Y_corr.flatten(), X_corr.flatten()).reshape(h, w)
        
        # Clip the values to ensure they are between 0 and 1
        corrected_pattern = np.clip(corrected_pattern, 0, 1)


        # self.pattern_corr = corrected_pattern

        # # Let's reconstruct the distortion matrix:
        # A = np.tan(beta) / (1 + np.tan(beta)*np.tan(alpha)) * (1/np.sin(beta))
        # B = -np.tan(beta) / (1 + np.tan(beta)*np.tan(alpha)) * (1/np.cos(alpha))
        # C = np.tan(alpha) / (1 + np.tan(beta)*np.tan(alpha)) * (1/np.cos(beta))
        # D = np.tan(alpha) / (1 + np.tan(beta)*np.tan(alpha)) * (1/np.sin(alpha))

        # D_matrix = np.array([[A, B], [C, D / gamma]])
        # try:
        #     D_inv = np.linalg.inv(D_matrix)
        # except np.linalg.LinAlgError:
        #     raise ValueError("Distortion matrix is singular and cannot be inverted.")

        # coords = np.stack([X.flatten(), Y.flatten()], axis=0)  # shape (2, N)
        # corrected_coords = D_inv @ coords  # shape (2, N)
        # X_src = corrected_coords[0].reshape(h, w)
        # Y_src = corrected_coords[1].reshape(h, w)

        # # Clip to bounds (still in normalized coordinates)
        # X_src = np.clip(X_src, 0, 1)
        # Y_src = np.clip(Y_src, 0, 1)

        # # Evaluate interpolated image
        # corrected_pattern = spline.ev(Y_src, X_src)
        # corrected_pattern = np.clip(corrected_pattern, 0, 1)

        ## Recenter the corrected pattern to the center of the image
        # Compute center of mass of array
        center_x = np.sum(np.arange(w) * np.sum(corrected_pattern, axis=0)) / np.sum(corrected_pattern)
        center_y = np.sum(np.arange(h) * np.sum(corrected_pattern, axis=1)) / np.sum(corrected_pattern)
        roll_axis = (int(h/2 - center_y), int(w/2 - center_x))

        # Roll the array to recenter it
        corrected_pattern = np.roll(corrected_pattern, roll_axis, axis=(0, 1))

        self.pattern_corr = corrected_pattern
        print("Distortion correction applied successfully.")
      
class randomImagesSeries:
    """Series of random images for DMD calibration.
    This class helps analyse the series of images taken by a camera in the image plane of a DMD displaying a sequence of patterns generated by an object `randomPattern Series`.
    """

    def __init__(self,  dim_camera: tuple = (3000, 4000), ROI_dim: tuple | None = None, images_path: str = './Calibration_Images/', density_array: np.ndarray = np.linspace(0, 1, 100)):
        """Initialize the randomImagesSeries with the path to the images and camera dimensions.
        Parameters
        ----------
        dim_camera: tuple
            Dimensions of the camera images (height, width).
        ROI_dim: tuple | None
            Optional dimensions of the Region of Interest (ROI) on the camera images (rows left aside to the top of the ROI, to the bottom, columns left aside to the left, and to the right).
        images_path: str
            Path to the directory containing the calibration images (should be in a lossless format (PNG or BMP) and named e.g. "0.png", "1.png", ..., "num_images-1.png").
        density_array: np.ndarray
            Array of densities corresponding to the patterns displayed on the DMD during the calibration sequence.
        Notes
        -----
        Only images should be in the directory, no other files.
        Note that the ROI doesn't need to include the whole beam, it can be a smaller cutout region that is constantly illuminated throughout the calibration sequence (e.g. a 100x100 pixel square in the center of the image).
    """
        if images_path[-1] != '/':
            images_path += '/'
        self.images_path = images_path
        self.dim_camera = dim_camera # Dimensions of the camera images ; (height, width) of the camera CCD
        self.num_images = len(os.listdir(images_path))
        self.density_array = density_array
        if ROI_dim is not None: # ROI dimension on the camera images (speeds up the analysis if it is smaller than the camera image)
            self.ROI_dim = ROI_dim

        # Load and format images from the specified path
        self.img_intensities = np.empty(self.num_images, dtype=np.uint8)        
        image_format = os.path.splitext(os.listdir(images_path)[0])[-1]

        for n in range(self.num_images):
            tmp_img_path = os.path.join(images_path, f"{n}.{image_format}")
            tmp_img = analogBWImage(img_path=tmp_img_path).img
            if tmp_img.size != dim_camera:
                raise ValueError(f"Image {n} has incorrect dimensions: {tmp_img.size}. Expected: {dim_camera}.")
            
            if tmp_img.ndim == 3 and tmp_img.shape[2] == 3:  # If RGB, convert to grayscale
                tmp_img = tmp_img[:, :, 0]  # Use the red channel for grayscale representation
            
            if ROI_dim is not None:
                if len(ROI_dim) != 4:
                    raise ValueError("ROI_dim must be a tuple of 4 integers (top, bottom, left, right).")
                if not all(isinstance(x, int) for x in ROI_dim):
                    raise TypeError("All elements of ROI_dim must be integers.")
                if any(x < 0 for x in ROI_dim):
                    raise ValueError("Elements of ROI_dim must be non-negative.")
                if ROI_dim[0] + ROI_dim[1] >= dim_camera[0] or ROI_dim[2] + ROI_dim[3] >= dim_camera[1]:
                    raise ValueError("ROI dimensions exceed camera image dimensions.")
                else:
                    tmp_img = tmp_img[ROI_dim[0]:dim_camera[0]-ROI_dim[1], ROI_dim[2]:dim_camera[1]-ROI_dim[3]]
            
            # Compute the average intensity in the ROI
            self.img_intensities[n] = np.mean(tmp_img)
            
    def get_group_images_by_density(self, intensity_thresh: float = 0.7):
        """Group images by the density of mirrors ON of the pattern they're associated to. The grouping is done according to the total intensity in the ROI of the image.
        Parameters
        ---------
        intensity_thresh: float
            Threshold for grouping images. If the difference in average intensity between two consecutive images is greater than this value, they are considered to belong to different groups.
        """

        self.next_d = [0]
        for i in range(self.num_images-1):
            if abs(self.img_intensities[i] - self.img_intensities[i+1]) > intensity_thresh:
                self.next_d.append(i+1)
        
        # Check I have as many groups as density values
        assert len(self.next_d) == len(self.density_array)-1, "Number of groups does not match the number of densities. Adjust the intensity_thresh parameter."

        self.group_indices = {}
        for i in range(self.density_array.size):
            self.group_indices[self.density_array[i]] = (self.next_d[i], self.next_d[i+1])

        # Average intensity in each group
        self.group_intensities = {}
        for density, indices in self.group_indices.items():
            group_intensity = np.mean(self.img_intensities[indices[0]:indices[1]])
            self.group_intensities[density] = group_intensity

        # Normalise to the intensity reached at density d = 1
        max_intensity = self.group_intensities[self.density_array[-1]]
        for density in self.group_intensities:
            self.group_intensities[density] /= max_intensity

    def plot_intensity_evolution(self, save_path: str | None = None):
        """Plots the average intensity in the ROI vs. the image index.
        Parameters
        ----------
        save_path: str | None
            Optional path to save the plot. If None, the plot is displayed but not saved.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.img_intensities, color='b')

        for idx in self.next_d:
            plt.axvline(x=idx, color='r', linestyle='--')

        plt.title("Average Intensity in ROI vs. Image Index", usetex = True)
        plt.xlabel("Image Index", usetex = True)
        plt.ylabel("Average Intensity (8-bit encoding)", usetex = True)
        plt.grid(True)

        if save_path is not None:
            if save_path[-1] != '/':
                save_path += '/'
            plt.savefig(os.path.join(save_path, "intensity_evolution.png"))
        else:
            plt.show()

    def get_fit_coefficients(self):
        """Fit a quadratic function to the average intensity vs. density data and return the coefficients.
        Returns
        -------
        A tuple (a, b) where the fitted function is a * density**2 + b*x.
        """
        if not hasattr(self, 'group_intensities'):
            self.get_group_images_by_density()
        
        # Fit the quadratic function
        popt, popcov = curve_fit(_quadratic, np.array(list(self.group_intensities.keys())), np.array(list(self.group_intensities.values())), p0=[1, 0], bounds=(0, [np.inf, np.inf]))
        print(f"Fitted coefficients: {popt}")
        self.fit_coeff = popt
        self.fit_cov = popcov

        # Compute the R-squared value
        residuals = np.array(list(self.group_intensities.values())) - _quadratic(np.array(list(self.group_intensities.keys())), *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((np.array(list(self.group_intensities.values())) - np.mean(np.array(list(self.group_intensities.values()))))**2)
        self.r_squared = 1 - (ss_res / ss_tot)
        print(f"R-squared: {self.r_squared}")
    
    def plot_intensity_vs_density(self, save_path: str | None = None):
        """Plot the average intensity in the ROI vs. the density of mirrors ON.
        Parameters
        ----------
        save_path: str | None
            Optional path to save the plot. If None, the plot is displayed but not saved.
        """
        if not hasattr(self, 'fit_coeff'):
            self.get_fit_coefficients()

        densities = np.array(list(self.group_intensities.keys()))
        intensities = np.array(list(self.group_intensities.values()))
        
        plt.figure(figsize=(10, 5))
        plt.scatter(densities, intensities, color='r', label='Measured Intensities')
        plt.plot(densities, _quadratic(densities, *self.fit_coeff), color='r', label='Fitted Quadratic Function')

        plt.plot(densities, densities, color='black', linestyle='--')
        plt.plot(densities, densities**2, color='black', linestyle='--')

        # Add a fill in-between with the uncertainty values from popcov
        a, b = self.fit_coeff
        uncertainties = np.sqrt(np.diag(self.fit_cov))
        lower_bound = _quadratic(densities, a - uncertainties[0], b - uncertainties[1])
        upper_bound = _quadratic(densities, a + uncertainties[0], b + uncertainties[1])
        
        plt.fill_between(densities, lower_bound, upper_bound, color='red', alpha=0.5, label='Uncertainty Bounds')

        plt.grid(True)
        plt.title(f"Average Intensity in ROI vs. Density of Mirrors ON.$R^2 = {self.r_squared}$", usetex = True)
        plt.xlabel("Density of Mirrors ON", usetex = True)
        plt.ylabel("Average Intensity in ROI (normalized)", usetex = True)

        if save_path is not None:
            if save_path[-1] != '/':
                save_path += '/'
            plt.savefig(os.path.join(save_path, "intensity_vs_density.png"))
        else:
            plt.show()

class binaryMask:
    """Type for binary masks ready to send to the DMD. Calibrates, corrects for distortion and performs error-diffusion on the input analog pattern inset.
    Note that the analog pattern should be a 2D numpy array with values between 0 and 0.9.
    """
    def __init__(self, analog_pattern_inset: np.ndarray, padding_dim: tuple = (0, 0, 0, 0), \
                 DMD_height: int = 1080, DMD_width: int = 1920, \
                alpha:float = np.deg2rad(-4.5), beta:float = np.deg2rad(-2.5), gamma:float = 1.09):
        """Initialize the binaryMask with an analog pattern inset and optional parameters for padding and distortion correction.
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
            Distortion coefficients for the DMD pattern correction. Default values are -4.5 degrees for alpha, -2.5 degrees for beta, and 1.09 for gamma.
        """
        self.analog_pattern = analogDMDPattern(analog_pattern_inset)
        self.padding = DMDPadding(padding_dim, DMD_height, DMD_width)
        self.analog_pattern.correct_distortion(alpha, beta, gamma) # Apply distortion correction to the analog pattern

    def calibrate_pattern(self, A, B):
        """Calibrate the analog pattern using the coefficients A and B.
        The calibration is done by solving the quadratic equation A*x^2 + B*x - y = 0 for each pixel in the analog pattern.

        Parameters
        ----------
        A, B : float
            Coefficients for the quadratic equation used in the calibration.
        """

        img = self.analog_pattern.pattern_corr
        calibrated_image = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                y = img[i,j]
                roots = np.roots([A, B, -y])
                # Return the root that is between 0 and 1
                for root in roots:
                    if root > 0 and root < 1:
                        calibrated_image[i,j] = root
                        break


        self.calibrated_pattern = calibrated_image

    def perform_error_diffusion(self, ref_image_analog, error_diffusion_matrix = ERROR_DIFFUSION_MATRIX, save_path: str = "./ED Patterns/"):
        """"Perform error diffusion on the calibrated pattern.
        The error diffusion is done using the specified error diffusion matrix, which defines the weights and directions of the error diffusion process.
        The reference image is expected to be a 2D numpy array with values between 0 and 0.9, representing the analog pattern.
        A stochastic noise is added to the error diffusion process to break the symmetry and avoid artifacts.
        
        Parameters
        ----------
        ref_image_analog : np.ndarray
            Reference image for error diffusion.

        error_diffusion_matrix : list of tuples, optional
            Error diffusion matrix to use. Default is ERROR_DIFFUSION_MATRIX.

        save_path : str, optional
            Path to save the resulting binary mask. Default is "./ED Patterns/".
        Raises
        ------
        ValueError
            If the reference image is not calibrated or if it does not have positive values for error diffusion.
        TypeError
            If the reference image is not a numpy array or if it is not 2D.
        """

        if not hasattr(self, 'calibrated_pattern'):
            raise ValueError("Pattern must be calibrated before performing error diffusion.")

        # Re-scale to 0-0.9 range
        scaling_factor = np.max(ref_image_analog) / 0.9
        if scaling_factor <= 0:
            raise ValueError("Reference image must have positive values for error diffusion.")
        
        ref_image_analog = ref_image_analog / scaling_factor
    
        ref_image_analog = self.padding.apply_padding(self.calibrated_pattern) # Apply padding to the reference image BEFORE ED (smoother result)
        Npx, Mpx = ref_image_analog.shape
        q = np.round(ref_image_analog, 0) # Initialise the 1-bit image
        v = np.zeros_like(ref_image_analog) # store the error values during the diffusion process

        for i in range(Npx): # iterating through y axis

            for j in range(Mpx): # iteration through x axis
                v_sum = 0

                for di,dj,w in error_diffusion_matrix:
                    ni = i+di # computing the coordinates of the neighbours in the direction (di, dj)
                    nj = j+dj
                    if ni>0 and nj<Mpx: # valid bounds
                        v_sum += v[ni,nj]*w # add error from the neighbours, weighted by the weight in error_diffusion_matrix
                if v_sum + np.random.rand() < 0.5: # add a random noise to the error diffusion process
                    q[i, j] = 0
                else:
                    q[i, j] = 1
                v[i,j] = v_sum+ ref_image_analog[i,j] - q[i,j]

        # If directory does not exist, create it
        directory = save_path.split('/')[:-1]
        if not os.path.exists(os.path.join(*directory)):
            os.makedirs(os.path.join(*directory))
            
        # Save the binary mask as a BMP image
        Image.fromarray((q * 255).astype(np.uint8)).save(os.path.join(save_path)) # ready to be processed by the DMD's software
        self.binary_mask = q


if __name__ == "__main__":
    # # Example usage for distortion correction - Gaussian Potential
    # height, width = 1080, 1920
    # x, y = np.meshgrid(np.linspace(-width//2, width//2, width), np.linspace(-height//2, height//2, height))
    # gaussian_potential = _gaussian(np.zeros((height, width)), sigma=100)

    # # Create a analogDMDpattern instance
    # gaussian_pattern = analogDMDPattern(gaussian_potential)

    # # Correct for distortion with gaussian_corr
    # alpha, beta, gamma = np.deg2rad(-4.5), np.deg2rad(-2.5), 1.09
    # gaussian_pattern.correct_distortion(alpha, beta, gamma)

    # fig, ax = plt.subplots(1, 2, figsize=(10, 20))
    # ax[0].imshow(gaussian_potential, cmap='gray')
    # ax[0].set_title("Original Gaussian Potential", usetex = True)
    # ax[0].axis('off')

    # ax[1].imshow(gaussian_pattern.pattern_corr, cmap='gray')
    # ax[1].set_title(r"Corrected Gaussian Potential (analogDMDPattern) $\alpha="+f"{np.rad2deg(alpha):.2f}$," +r"$\beta="+f"{np.rad2deg(beta):.2f}$,"+ r"$\gamma="+f"{gamma:.2f}$", usetex = True)
    # ax[1].axis('off')

    # # Example usage for ED (on Cicero)
    cicero_image = analogBWImage(img_path="Test_Analog_Images/Cicero_BW.jpg")

    # normalise cicero_image to 0-0.9 range (image is encoded in 8 bits, so values are between 0 and 255)

    DMD_height, DMD_width = 1080, 1920

    # Setup the padding for the Cicero image on the DMD array
    padding_cicero_dim = (
        cicero_image.img.shape[0] // 2,  # Top padding
        DMD_height - cicero_image.img.shape[0] // 2 - cicero_image.img.shape[0],  # Bottom padding
        cicero_image.img.shape[1] // 2,  # Left padding
        DMD_width - cicero_image.img.shape[1] // 2 - cicero_image.img.shape[1],  # Right padding
    )
    padding_cicero = DMDPadding(padding_cicero_dim, DMD_height, DMD_width)

    # Crreate an analogDMDPattern instance for the Cicero image and correct for distortion
    cicero_pattern = analogDMDPattern(cicero_image.img)
    # Create a binary mask from the cicero pattern (assuming A = 0.5, B = 0.5 for calibration)
    cicero_mask = binaryMask(cicero_pattern.pattern, padding_cicero_dim, DMD_height, DMD_width)
    cicero_mask.calibrate_pattern(0.5, 0.5)

    cicero_mask.perform_error_diffusion(cicero_image.img, save_path="./ED Patterns/Cicero_test/Cicero_image_ED.bmp")

    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    ax[0, 0].imshow(cicero_image.img, cmap='gray')
    ax[0, 0].set_title("Original Cicero Image", usetex = True)
    ax[0, 0].axis('off')

    ax[0, 1].imshow(cicero_mask.analog_pattern.pattern_corr, cmap='gray')
    ax[0, 1].set_title(r"Corrected Cicero Image (analogDMDPattern) $\alpha="+f"{np.rad2deg(-4.5):.2f}$," +r"$\beta="+f"{np.rad2deg(-2.5):.2f}$,"+ r"$\gamma=1.09$", usetex = True)
    ax[0, 1].axis('off')

    ax[1, 0].imshow(cicero_mask.calibrated_pattern, cmap='gray')
    ax[1, 0].set_title("Calibrated Cicero Pattern", usetex = True)
    ax[1, 0].axis('off')

    ax[1, 1].imshow(cicero_mask.binary_mask, cmap='gray')
    ax[1, 1].set_title("Binary Mask after Error Diffusion", usetex = True)
    ax[1, 1].axis('off')

    plt.tight_layout()
    plt.show()