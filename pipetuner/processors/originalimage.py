"""
An image process to just show the original image
"""
import cv2

from imgprocessor import ImageProcessor

class OriginalImage (ImageProcessor):
    """
    This class is an ImageProcess which used to render the original image with
    no extra effects
    """
    def __init__(self, name, image=None, on_image_change=None):

        super().__init__(name, image, on_image_change)

    def _render(self):
        self.setWindow(self._name)

        self._processed_image = self.image
        cv2.imshow(self._name, self._processed_image)
