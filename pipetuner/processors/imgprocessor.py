"""
The parent processor class
"""

import os
import cv2

class ImageProcessor:
    """
    This is the parent class for all images processors
    """
    def __init__(self, name, image=None, image_info = None, on_image_change=None):
        self._name = name
        self._image_info = image_info
        self._config = {}
        self.image = image
        self._on_image_change = on_image_change
        self._processed_image = None
        self._win_Ctrl = 'Controls'
        self._pipline = None


    def __str__( self ) :
        return "( " + "name: " + str( self._name ) + ", " + str(self._config) + ")\n"

    def onImageChange(self, image_processor):
        self.setImage(image_processor.processedImage(), image_processor.imageInfo())

    def setWinControl (self, win_ctrl_name):
        self._win_Ctrl = win_ctrl_name
        # we either use WINDOW_NORMAL or WINDOW_AUTOSIZE
        cv2.namedWindow(self._win_Ctrl, cv2.WINDOW_NORMAL)

    def setWindow (self, win_name):
        # We use WINDOW_KEEPRATIO for image window.
        cv2.namedWindow(win_name, cv2.WINDOW_KEEPRATIO)

    def setImage (self, image, image_info = None):
        self.image = image
        self._image_info = image_info
        self._ysize = self.image.shape[0]
        self._xsize = self.image.shape[1]
        self.refresh()

    """
    This actually means adding the method onImageChange as a callback
    to be called when an image change happend.
    """
    def addProcessor (self, image_processor):
        self._on_image_change = image_processor

    def setPipeline (self, pipeline):
        self._pipline = pipeline

    def onCloseWindow(self):
        cv2.destroyWindow(self._name)

    def processedImage (self):
        return self._processed_image

    def imageInfo (self):
        return self._image_info

    def name (self):
        return self._name

    def _render(self):
        pass

    def setParameter(self, name, value):
        if self._config is not None:
            self._config [name] = value

    def getParameter (self, name):
        if self._config is not None:
            return self._config [name]
        else:
            return None

    def configuration(self):
        return self._config

    def setConfig (self, config):
        self._config = config
        for key in config:
            cv2.setTrackbarPos(key, self._win_Ctrl, config[key])     

    def saveProcessedImage(self, image_filename, out_folder):
        (head, tail) = os.path.split(image_filename)
        (root, ext) = os.path.splitext(tail)

        out_filename = os.path.join(out_folder, root + '-' + self.name() + ext)
        cv2.imwrite(out_filename, self.processedImage())

    def refresh(self):
        """Enforce refresh of the image to invoke the following in the pipeline
        """        
        self._render()
        # Notify the following processor in the pipline with the update 
        # to force refresh to following Processors
        if self._on_image_change and callable(self._on_image_change):
            self._on_image_change(self)
