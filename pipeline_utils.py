"""
Utilities to build pipelines to identify
lanes in images and videos
"""
import logging

# create logger
logger = logging.getLogger(__name__)


import cv2
import os
import json

from guiutils import *

class ImageProcessorPipeline:

    def __init__(self, name, image_filename, out_folder, config_file):
        self._name = name
        self._image_filename = image_filename
        self._out_folder = out_folder
        self._config_file = config_file
        self._processors = []
        self._config = self.loadConfig ()

        cv2.namedWindow('input', cv2.WINDOW_KEEPRATIO)
        self.image = cv2.imread(image_filename)   
        cv2.imshow('input', self.image)


    def __str__( self ) :
        return "( " + "name: " + str( self._name ) + ", " + str(self.getConfig()) + ")\n"

    def addProcessor (self, processor):
        if isinstance (processor, ImageProcessor):
            if len(self._processors) > 0:
                self._processors[-1].addProcessor (processor.onImageChange)
            self._processors.append(processor)
            processor.setPipeline (self)
        else:
            logger.error('Invalid Processor type: %s', type(processor))
    
    def loadConfig(self):
        with open(self._config_file) as configFile:
            return json.load(configFile)


    def saveConfig(self):
        (head, tail) = os.path.split(self._image_filename)
        (root, ext) = os.path.splitext(tail)
        out_filename = os.path.join(self._out_folder, root + '.json')

        config = self.getConfig()
        with open(out_filename, 'w', encoding='utf-8') as out_file:
            json.dump(config, out_file, ensure_ascii=False, indent=4)

        logger.info('Configuration saved to file: %s', out_filename) 

    def getConfig (self):
        config = {}
        for processor in self._processors:
            config[type(processor).__name__] = processor.configuration()
        return config

    def setConfig (self, config):
        for processor in self._processors:
            processor.setConfig (config[type(processor).__name__])

    def setImage (self, image):
        """
        This method will call the setImage of the first processor which in turn will 
        set the input image and call refresh for the rest of the chain
        """
        self._processors[0].setImage(image) 

    def getImage (self):
        """
        This method will get the original image before any processing
        """
        return self._processors[0].image 

    def saveProcessedImages(self):
        for processor in self._processors:
            processor.saveProcessedImage(self._image_filename, self._out_folder)
            
    def save(self):
        self.saveProcessedImages()
        self.saveConfig()

    def _render(self):
        for processor in self._processors:
            processor._render()

    def refresh (self):
        """
        This method will call the refresh of the first processor which in trun will 
        call refresh for the rest of the chain
        """
        self._processors[0].refresh()

class FindLanesPipeline (ImageProcessorPipeline):

    def __init__(self, name, image_filename, out_folder, config_file):

        super().__init__(name, image_filename, out_folder, config_file)

        blur_image = SmoothImage('Bluring Config', self.image, average_filter_size=5, gaussian_filter_size=1, median_filter_size=1, bilateral_filter_size=1)
        self.addProcessor (blur_image)

        edge_finder = EdgeFinder('Edge Finder Config', blur_image.processedImage(), min_threshold=28, max_threshold=115)
        self.addProcessor (edge_finder)
        
        region_mask = RegionMask('Region Masked dimensions', edge_finder.processedImage())
        self.addProcessor (region_mask)
        
        hough_lines = HoughLines('Hough Lines Config', region_mask.processedImage())
        self.addProcessor (hough_lines)
        
        image_blender = ImageBlender('Image Mix Config', hough_lines.processedImage(), self.image)
        self.addProcessor (image_blender)

        if self._config:
            self.setConfig(self._config)

        self.refresh()

class VideoProcessorPipeline (ImageProcessorPipeline):

    def __init__(self, name, video_filename, out_folder, config_file):
        self._name = name
        self.image = None
        self._video_filename = video_filename
        self._out_folder = out_folder
        self._config_file = config_file
        self._processors = []
        self._config = self.loadConfig ()
        self._speed = 25

        # cv2.namedWindow('input', cv2.WINDOW_KEEPRATIO)
        # self.image = cv2.imread(image_filename)   
        # cv2.imshow('input', self.image)

        blur_image = SmoothImage('Bluring Config', self.image, average_filter_size=5, gaussian_filter_size=1, median_filter_size=1, bilateral_filter_size=1)
        self.addProcessor (blur_image)

        edge_finder = EdgeFinder('Edge Finder Config', blur_image.processedImage(), min_threshold=28, max_threshold=115)
        self.addProcessor (edge_finder)
        
        region_mask = RegionMask('Region Masked dimensions', edge_finder.processedImage())
        self.addProcessor (region_mask)
        
        hough_lines = HoughLines('Hough Lines Config', region_mask.processedImage())
        self.addProcessor (hough_lines)
        
        image_blender = ImageBlender('Image Mix Config', hough_lines.processedImage(), self.image)
        self.addProcessor (image_blender)

        if self._config:
            self.setConfig(self._config)

    def videoCapture (self):

        frame_counter = 0

        # Iterate while the cap is open, i.e. while we still get new frames.
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()

            frame_counter += 1
            #If the last frame is reached, reset the capture and the frame_counter
            if frame_counter == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                frame_counter = 0 #Or whatever as long as it is the same as next line
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
            
            self.setImage (frame)
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # cv2.imshow('frame',gray)
            if cv2.waitKey(self._speed) & 0xFF == ord('q'):
                break
    
    def __enter__(self): 
        self.cap = cv2.VideoCapture(self._video_filename)
        return self
  
    def __exit__(self, exception_type, exception_value, traceback):
        logger.exception ("Error in Video Capture")
        self.cap.release()
        cv2.destroyAllWindows()