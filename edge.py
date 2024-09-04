from skimage.io import imread
from utils import show, compare
import numpy as np

# from skimage.filters import threshold_local
from skimage.color import rgb2gray
from skimage.filters import sobel, threshold_otsu
from skimage.feature import canny
from skimage.measure import find_contours
from skimage.restoration import denoise_tv_chambolle
from matplotlib import pyplot as plt
from scipy.ndimage import binary_dilation

# image = imread("2.jpg")
image = imread("1.jpg")


def mark_contours(image):
    """A function to find contours from an image"""
    gray_image = rgb2gray(image)
    # Find optimal threshold
    thresh = threshold_otsu(gray_image)
    # Mask
    binary_image = gray_image > thresh

    contours = find_contours(binary_image)

    return contours


def plot_image_contours(
    image: np.ndarray, denoise: bool = True, only_edges: bool = True
):
    if denoise:
        image = denoise_tv_chambolle(image, channel_axis=2)

    fig, ax = plt.subplots()
    if only_edges:
        black_mask = np.zeros_like(image)
        ax.imshow(black_mask, cmap=plt.cm.gray)
        _c = "white"
    else:
        ax.imshow(image, cmap=plt.cm.gray)
        _c = "black"

    found_contours = mark_contours(image)

    for contour in found_contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color=_c)

    ax.axis("off")
    plt.show()
    plt.figure(figsize=(12, 9))


# plot_image_contours(image)


def sobel_detection(img):
    img_gray = rgb2gray(img)

    img_edges = sobel(img_gray)
    compare(img, img_edges, "Images of coins with edges detected")


def canny_detection(img, sigma: float = 1.0, thickness: int = 1):
    img_gray = rgb2gray(img)
    canny_edges = canny(img_gray, sigma=sigma)

    if thickness > 1:
        # Define a structuring element to control the thickness
        struct_element = np.ones((thickness, thickness), dtype=bool)

        # Dilate the boolean array to increase thickness
        canny_edges = binary_dilation(canny_edges, structure=struct_element)

    _transformed = np.uint8(canny_edges) * 255

    compare(img, _transformed, "Edges detected with Canny algorithm")


# canny_detection(image, 1.5, 3)
