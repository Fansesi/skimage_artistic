from skimage.color import label2rgb
from skimage.segmentation import slic
from skimage.io import imread
from collections import Counter
import numpy as np
from utils import show, compare


# img = data.coffee()
img = imread("1.jpg")


def segment(image, n_segments=100, _kind: str = "most_frequent"):
    def most_frequent_color(image, segment_mask):
        """
        Calculate the most frequent color in each segment of the image.

        Args:
        - image: Original image as a NumPy array.
        - segment_mask: Mask with segment labels from SLIC.

        Returns:
        - result_image: Image where each segment is filled with the most frequent color.
        """
        result_image = np.zeros_like(image)

        for segment_label in np.unique(segment_mask):
            mask = segment_mask == segment_label
            # Extract colors from the segment
            colors = image[mask]

            # Find the most frequent color
            if len(colors) > 0:
                most_common_color = Counter(map(tuple, colors)).most_common(1)[0][0]
                result_image[mask] = most_common_color

        return result_image

    # Obtain superpixels / segments
    superpixels = slic(
        image=image,
        n_segments=n_segments,
        compactness=15.0,
        max_num_iter=20,
        sigma=0,
        spacing=None,
        convert2lab=None,
        enforce_connectivity=True,
        min_size_factor=0.5,
        max_size_factor=3,
        slic_zero=False,
        start_label=1,
        mask=None,
        channel_axis=-1,
    )

    if _kind == "most_frequent":
        # Fill each segment with the most frequent color
        segmented_image = most_frequent_color(image, superpixels)
    else:
        # All the other kinds can be found by looking at label2rgb
        segmented_image = label2rgb(label=superpixels, image=img, kind=_kind)

    return segmented_image


# segmented_img = segment(img, 1500, "most_frequent")
# show(segmented_img, "SLIC")


# this example is from the docs: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_rag_mean_color.html#sphx-glr-auto-examples-segmentation-plot-rag-mean-color-py
# labels1 = slic(img, compactness=30, n_segments=200, start_label=1)
# out1 = color.label2rgb(labels1, img, kind="avg", bg_label=0)

# g = graph.rag_mean_color(img, labels1)
# labels2 = graph.cut_threshold(labels1, g, 29)
# out2 = label2rgb(labels2, img, kind="avg", bg_label=0)


# fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

# ax[0].imshow(out1)
# ax[1].imshow(out2)

# for a in ax:
#     a.axis("off")

# plt.tight_layout()
