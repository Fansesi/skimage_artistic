import matplotlib.pyplot as plt
import numpy as np
from loguru import logger as lg
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from utils import show, compare
from typing import Union

from skimage.io import imread
from skimage.filters import threshold_local, sobel, threshold_otsu
from skimage.feature import canny
from skimage.measure import find_contours
from skimage.restoration import denoise_tv_chambolle
from scipy.ndimage import binary_dilation
from skimage.color import rgb2gray
from skimage.draw import polygon


class ImageSelector:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.selected_polygon = None
        self.cropped_image = None
        self.remaining_image = None
        self.selection_mode = False
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image)

        # Connect the key press event
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        plt.show()
        # self.show_selected()

    def select(self):
        # Set up the figure and axis
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image)

        # Connect the key press event
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        plt.show()
        self.show_selected()
        return self.cropped_image, self.remaining_image

    def on_key_press(self, event):
        # Toggle selection mode with 'c' key
        if event.key == "c":
            self.selection_mode = not self.selection_mode
            if self.selection_mode:
                print("Selection mode enabled. Click to select points.")
                self.selector = PolygonSelector(
                    self.ax,
                    self.on_select,
                    # lineprops=dict(color="r", linewidth=2, alpha=0.5),
                )
                # Customize the line appearance
                # self.selector.line.set_color("r")
                # self.selector.line.set_linewidth(2)
                # self.selector.line.set_alpha(0.5)
            else:
                print("Selection mode disabled.")
                if hasattr(self, "selector"):
                    self.selector.disconnect_events()

        # Finalize the selection with 'enter' key
        if event.key == "enter" and hasattr(self, "selector"):
            self.selector.disconnect_events()
            self.selection_mode = False
            self.crop_to_selection()

    def on_select(self, verts):
        # Store the vertices of the polygon
        self.selected_polygon = Path(verts)

    def crop_to_selection(self):
        if not self.selected_polygon:
            return

        # Create a mask where the selected polygon is True
        mask = np.zeros(self.image.shape[:2], dtype=bool)
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                if self.selected_polygon.contains_point((j, i)):
                    mask[i, j] = True

        # Extract the bounding box of the polygon
        x, y = np.nonzero(mask)
        if x.size == 0 or y.size == 0:
            return

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        # Create the cropped image
        self.cropped_image = np.zeros_like(self.image)
        self.cropped_image[x_min : x_max + 1, y_min : y_max + 1] = self.image[
            x_min : x_max + 1, y_min : y_max + 1
        ]
        self.cropped_image[~mask] = 0

        # Create the remaining image
        self.remaining_image = np.copy(self.image)
        self.remaining_image[mask] = 0

        # NOTE: just setting self.cropped_image and self.remaining_image
        # return self.cropped_image, self.remaining_image
        # self.show_selected()

    def show_selected(self):
        # Display the cropped image and the remaining part
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(self.cropped_image)
        ax1.set_title("Selected Part")
        ax2.imshow(self.remaining_image)
        ax2.set_title("Remaining Part")
        plt.show()


class ImageOps:
    """Image operations based on skimage. This class is for artistic purposes."""

    def __init__(self, image: Union[np.ndarray, str], mode: str = "show") -> None:
        """
        ## Params:
        * `mode`: should be `show` or `compare`."""
        if isinstance(image, str):
            self.image = imread(image)
        else:
            self.image = image
        # storing gray image too
        self.gray_image = rgb2gray(self.image)

        self.mode = mode

    def basic_threshold(self, params: Union[dict, None] = None):
        if params == None:
            local_thresh = threshold_local(
                image=self.image,
                block_size=3,
                method="gaussian",
                offset=0.004,
                mode="reflect",
                param=None,
                cval=0,
            )
        else:
            local_thresh = threshold_local(image=self.image, **params)

        binary_flower = self.gray_image > local_thresh
        self.mode_run(binary_flower)

        return binary_flower

    def sobel_edge_detection(self, params: Union[dict, None] = None):
        if params == None:
            # TODO: play with sobel parameters and look for pleasing parameters.
            image_edges = sobel(
                image=self.gray_image,
                mask=None,
                axis=None,
                mode="reflect",
                cval=0,
            )
        else:
            image_edges = sobel(self.gray_image, **params)

        image_edges = ImageOps.thicker(2, image_edges)

        self.mode_run(image_edges)
        return image_edges

    def canny_edge_detection(self, params: Union[dict, None] = None):
        if params == None:
            # TODO: play with canny parameters and look for pleasing parameters.
            image_edges = canny(
                image=self.gray_image,
                sigma=1,
                low_threshold=None,
                high_threshold=None,
                mask=None,
                use_quantiles=False,
                mode="constant",
                cval=0,
            )
        else:
            image_edges = canny(self.gray_image, **params)

        image_edges = ImageOps.thicker(2, image_edges)

        self.mode_run(image_edges)
        return image_edges

    @classmethod
    def find_contours(cls, gray_image: np.ndarray, params: Union[dict, None] = None):
        # Find optimal threshold
        thresh = threshold_otsu(gray_image)

        # Mask
        binary_image = gray_image > thresh
        if params == None:
            # TODO: play with find_contours parameters and look for pleasing parameters.
            contours = find_contours(
                image=binary_image,
                level=None,
                fully_connected="low",
                positive_orientation="low",
                mask=None,
            )
        else:
            contours = find_contours(binary_image, **params)

        return contours

    def draw_contours(self, params: Union[dict, None] = None):
        image = ImageOps.denoise(self.image)

        # Create a black background
        black_background = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Find contours
        found_contours = self.find_contours(rgb2gray(image))

        # Draw contours on the black background
        for contour in found_contours:
            # Convert float coordinates to integers
            rr, cc = polygon(contour[:, 0], contour[:, 1], shape=black_background.shape)
            # Clip coordinates to be within image boundaries
            # rr = np.clip(rr, 0, black_background.shape[0] - 1)
            # cc = np.clip(cc, 0, black_background.shape[1] - 1)
            black_background[rr, cc] = 255  # Set contour points to white

        self.mode_run(black_background)
        return black_background

    @classmethod
    def denoise(cls, image: np.ndarray, params: Union[dict, None] = None):
        if params == None:
            return denoise_tv_chambolle(
                image=image,
                weight=0.1,
                eps=0.0002,
                max_num_iter=200,
                channel_axis=2,
            )
        else:
            return denoise_tv_chambolle(image, **params)

    @classmethod
    def thicker(cls, thickness: int, image: np.ndarray):
        """Dilation operation."""
        # Define a structuring element to control the thickness
        struct_element = np.ones((thickness, thickness), dtype=bool)

        # Dilate the boolean array to increase thickness
        image = binary_dilation(image, structure=struct_element)

        return np.uint8(image) * 255

    def mode_run(self, processed_image: np.ndarray, params: dict = {}):
        """`show` or `compare` based on `self.mode`."""
        if self.mode == "show":
            show(processed_image, **params)
        elif self.mode == "compare":
            compare(original=self.image, processed=processed_image, **params)
        elif self.mode == "don't":
            pass
        else:
            lg.error(f"self.mode is {self.mode}, couldn't run anything.")

    def cut(self):
        def tight_crop(frame):
            """Crops the selection to are minimum."""
            # Find rows that are not completely black
            non_black_rows = np.any(frame != [0, 0, 0], axis=(1, 2))

            # Find columns that are not completely black
            non_black_cols = np.any(frame != [0, 0, 0], axis=(0, 2))

            # Remove black rows and columns
            cropped_img = frame[non_black_rows][:, non_black_cols]
            return cropped_img

        lg.warning("Cutting tool might not work with huge resolutions.")
        selector = ImageSelector(self.image)

        return tight_crop(selector.cropped_image)
