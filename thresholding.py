# from skimage import data, segmentation, color, graph
from skimage.io import imread
from utils import show, compare
from skimage.filters import threshold_local
from skimage.color import rgb2gray

img = imread("1.jpg")
img_gray = rgb2gray(img)

local_thresh = threshold_local(
    img_gray, block_size=3, offset=0.0004
)  # play with the settings.

binary_flower = img_gray > local_thresh
params = {}
compare(img, binary_flower, title_processed="Tresholded flower image", **params)
