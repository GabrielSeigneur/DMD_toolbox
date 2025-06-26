import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

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


# class analogImage:
#     """Type for analog images from an image file or numpy array."""
#     def __init__(self, img_array: np.ndarray | None = None, img_path: str | None = None):
#         if img_array is not None and img_path is not None:
#             raise ValueError("Provide either img_array or img_path, not both.")
#         if img_array is None and img_path is None:
#             raise ValueError("One of img_array or img_path must be provided.")
        
#         if img_array is not None:
#             if not isinstance(img_array, np.ndarray):
#                 raise TypeError("img_array must be a numpy array.")
#             self.img = img_array
#         else:
#             if not os.path.isfile(img_path):
#                 raise FileNotFoundError(f"Image file {img_path} does not exist.")
#             self.img = np.array(Image.open(img_path))
        
#         if self.img.ndim != 2 and self.img.ndim != 3:
#             raise ValueError("Image must be 2D or 3D (grayscale or RGB).")

        

# class binaryMask:
#     """Type for binary masks ready to send to the DMD."""
#     def __init__(self, mask: np.ndarray):
#         if not isinstance(mask, np.ndarray):
#             raise TypeError("Mask must be a numpy array.")
#         if mask.ndim != 2:
#             raise ValueError("Mask must be a 2D array.")
#         self.mask = mask
