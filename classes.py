import matplotlib.pyplot as plt
import numpy as np


class ImageOps:
    """Image operations based on skimage. This class is for artistic purposes."""

    def __init__(self) -> None:
        pass


class ImageSelector:
    def __init__(self, image):
        self.image = image
        self.rect = None
        self.start_point = None
        self.end_point = None
        self.cropped_image = None
        self.remaining_image = None

        # Set up the figure and axis
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image)

        # Connect the mouse events
        self.cid_press = self.fig.canvas.mpl_connect(
            "button_press_event", self.on_press
        )
        self.cid_release = self.fig.canvas.mpl_connect(
            "button_release_event", self.on_release
        )
        self.cid_motion = self.fig.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )

        plt.show()

    def on_press(self, event):
        # Record the starting point of the rectangle
        self.start_point = (int(event.xdata), int(event.ydata))

        # Remove any previous rectangle
        if self.rect:
            self.rect.remove()
            self.rect = None

    def on_release(self, event):
        # Record the ending point of the rectangle
        self.end_point = (int(event.xdata), int(event.ydata))

        # Draw the final rectangle
        self.rect = plt.Rectangle(
            self.start_point,
            self.end_point[0] - self.start_point[0],
            self.end_point[1] - self.start_point[1],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        self.ax.add_patch(self.rect)
        self.fig.canvas.draw()

        # Crop the image and show the result
        self.crop_image()
        self.show_result()

    def on_motion(self, event):
        # Update the rectangle as the mouse moves
        if self.start_point:
            x0, y0 = self.start_point
            x1, y1 = int(event.xdata), int(event.ydata)

            # Remove the previous rectangle if it exists
            if self.rect:
                self.rect.remove()

            # Draw a new rectangle
            self.rect = plt.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="r", facecolor="none"
            )
            self.ax.add_patch(self.rect)
            self.fig.canvas.draw()

    def crop_image(self):
        # Ensure start_point and end_point are properly defined
        if not self.start_point or not self.end_point:
            return

        x0, y0 = self.start_point
        x1, y1 = self.end_point

        # Handle the case where the selection might be reversed
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])

        # Crop the selected part
        self.cropped_image = self.image[y0:y1, x0:x1]

        # Create the remaining part
        self.remaining_image = np.copy(self.image)
        self.remaining_image[y0:y1, x0:x1] = (
            0  # Set the selected part to black or any other color
        )

    def show_result(self):
        # Display the cropped image and the remaining part
        if self.cropped_image is not None and self.remaining_image is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(self.cropped_image)
            ax1.set_title("Cropped Part")
            ax2.imshow(self.remaining_image)
            ax2.set_title("Remaining Part")
            plt.show()


image = np.random.rand(
    300, 400, 3
)  # Replace with imread('your_image.jpg') for an actual image
selector = ImageSelector(image)
