import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
import os

def _quadratic(x: np.ndarray, a:float, b:float) -> np.ndarray:
        """Quadratic function for fitting."""
        return a * x**2 + b

class DMDPadding:
    """
    Padding for DMD patterns.
    The padding is defined as a tuple of (top, bottom, left, right), and is meant to be fully illuminated by the incident beam.
    Careful about the Y axis: it is point down, so top is the first row and bottom is the last row.
    """
    def __init__(
                self,
                top: int,bottom: int,
                left: int, right: int, 
                height : int = 1080, width : int = 1920
                ):
        
        """
        Initialize the DMD padding with specified dimensions and padding values.
        :param height: Height of the DMD pattern.
        :param width: Width of the DMD pattern.
        :param top: Number of rows to pad at the top.
        :param bottom: Number of rows to pad at the bottom.
        :param left: Number of columns to pad on the left.
        :param right: Number of columns to pad on the right.
        """

        if not all(isinstance(x, int) for x in [top, bottom, left, right]):
            raise TypeError("Padding values must be integers.")
        
        
        self.padding = (top, bottom, left, right)
        self.height = height
        self.width = width

        padding_image = np.zeros((height, width), dtype=np.bool_)
        padding_image[top:height-bottom, left:width-right] = True
        self.padding_image = padding_image


    def apply_padding(self, pattern: np.ndarray) -> np.ndarray:
        """Apply padding to a given pattern."""
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
    """Series of random patterns for DMD calibration."""
    def __init__(self, num_patterns: int, padding: DMDPadding):
        if not isinstance(num_patterns, int) or num_patterns <= 0:
            raise ValueError("num_patterns must be a positive integer.")
        self.padding = padding
        self.num_patterns = num_patterns
        self.density_array = np.linspace(0, 1, num_patterns)

        self.patterns = np.zeros((num_patterns, self.padding.height, self.padding.width), dtype=np.bool_)
        self.image_path = "./Calibration_Images/" # Where the images will be saved
    
    def build_pattern(self, density: float, dim_tuple: tuple) -> np.ndarray:
        """
        Generate a random pattern based on the specified density.
        :param density: Density of the pattern (0 to 1).
        :param dim_tuple: Tuple specifying the dimensions of the pattern (height_inset, width_inset).
        """
        pattern = np.zeros(dim_tuple, dtype=np.bool_)
        for i in range(dim_tuple[0]):
            for j in range(dim_tuple[1]):
                if np.random.rand() < density:
                    pattern[i, j] = True

        return pattern
    
    def generate_patterns(self, save_path: str = './Calibration_Patterns/'):
        """Generate random patterns and optionally save them to files."""
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
    """Type for analog images from an image file or numpy array."""
    def __init__(self, img_array: np.ndarray | None = None, img_path: str | None = None):
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
        
class randomImagesSeries:
    """Series of random images for DMD calibration.
    This class helps analyse the series of images taken by a camera in the image plane of a DMD displaying a sequence of patterns generated by an object `randomPattern Series`.
    Arguments:
    - dim_camera: Dimensions of the camera images (height, width).
    - ROI_dim: Optional dimensions of the Region of Interest (ROI) on the camera images (rows left aside to the top of the ROI, to the bottom, columns left aside to the left, and to the right).
    - images_path: Path to the directory containing the calibration images (should be in a lossless format (PNG or BMP) and named e.g. "0.png", "1.png", ..., "num_images-1.png").
    - density_array: Array of densities corresponding to the patterns displayed on the DMD during the calibration sequence.
    Only images should be in the directory, no other files.
    Note that the ROI doesn't need to include the whole beam, it can be a smaller cutout region that is constantly illuminated throughout the calibration sequence (e.g. a 100x100 pixel square in the center of the image).
    """

    def __init__(self,  dim_camera: tuple = (3000, 4000), ROI_dim: tuple | None = None, images_path: str = './Calibration_Images/', density_array: np.ndarray = np.linspace(0, 1, 100)):

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
        Arguments:
        - intensity_thresh: Threshold for grouping images. If the difference in average intensity between two consecutive images is greater than this value, they are considered to belong to different groups.
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
        Arguments:
        - save_path: Optional path to save the plot. If None, the plot is displayed but not saved.
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
        Returns:
        - A tuple (a, b) where the fitted function is a * density^2 + b.
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
        Arguments:
        - save_path: Optional path to save the plot. If None, the plot is displayed but not saved.
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
    """Type for binary masks ready to send to the DMD."""
    def __init__(self, mask: np.ndarray):
        if not isinstance(mask, np.ndarray):
            raise TypeError("Mask must be a numpy array.")
        if mask.ndim != 2:
            raise ValueError("Mask must be a 2D array.")
        self.mask = mask
        
        raise NotImplementedError("This class is not implemented yet.")