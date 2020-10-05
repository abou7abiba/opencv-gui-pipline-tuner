"""
How to run:
python find_lanes.py <image path>
"""

import argparse
import cv2
import os
import logging
import json

from guiutils import SmoothImage, EdgeFinder, RegionMask, HoughLines, ImageBlender

# create logger
logger = logging.getLogger(__name__)

def __init_logger():
    #Set level to logging.DEBUG to see CRITICAL, ERROR, WARNING, INFO and DEBUG statements
    #Set level to logging.ERROR to see the CRITICAL & ERROR statements only
    logger.setLevel(logging.DEBUG)

    fileHandler = None
    consoleHandler = None
    
    if len(logger.handlers) > 0:
        for handler in logger.handlers:
            # makes sure no duplicate handlers are added
            if isinstance(handler, logging.FileHandler) and not isinstance(handler, logging.StreamHandler):
                fileHandler = handler
            elif isinstance(handler, logging.StreamHandler):
                fileHandler = handler
                    
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler which logs even info messages
    if fileHandler is None:
        fileHandler = logging.FileHandler('Tuning_info_out.log')
        fileHandler.setLevel(logging.INFO)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    # create console handler and set level to debug
    if consoleHandler is None:
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.ERROR)
        consoleHandler.setFormatter(formatter)
        logger.addHandler(consoleHandler)

def main():
    __init_logger()
    parser = argparse.ArgumentParser(description='Visualizes the line for hough transform.')
    parser.add_argument('filename')

    args = parser.parse_args()

    # img = cv2.imread(args.filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(args.filename)   
    cv2.imshow('input', img)

    blur_image = SmoothImage('Bluring Config', img, average_filter_size=5, gaussian_filter_size=1, median_filter_size=1, bilateral_filter_size=1)
    
    edge_finder = EdgeFinder('Edge Finder Config', blur_image.processedImage(), min_threshold=28, max_threshold=115)
    blur_image.addProcessor (edge_finder.onImageChange)
    
    region_mask = RegionMask('Region Masked dimensions', edge_finder.processedImage())
    edge_finder.addProcessor (region_mask.onImageChange)
    
    hough_lines = HoughLines('Hough Lines Config', region_mask.processedImage())
    region_mask.addProcessor (hough_lines.onImageChange)
    
    image_blender = ImageBlender('Image Mix Config', hough_lines.processedImage(), img)
    hough_lines.addProcessor (image_blender.onImageChange)
    
    # will inforce refresh for all pipeline.
    blur_image.refresh()

    cv2.waitKey(0)

    print ("Adjust the parameters as desired.  Hit any key to close. Your configuration will be in Tuning_info_out.log")

    logger.info("Configuration parameters:\n %s %s %s %s %s", blur_image, edge_finder, region_mask, hough_lines, image_blender)
    logger.debug("Configuration parameters:\n %s %s %s %s %s", blur_image, edge_finder, region_mask, hough_lines, image_blender)

    (head, tail) = os.path.split(args.filename)
    (root, ext) = os.path.splitext(tail)

    smoothed_filename = os.path.join("output_images", root + blur_image.name() + ext)
    edge_filename = os.path.join("output_images", root + edge_finder.name() + ext)
    maskeed_filename = os.path.join("output_images", root + region_mask.name() + ext)
    hough_filename = os.path.join("output_images", root + hough_lines.name() + ext)
    blended_filename = os.path.join("output_images", root + image_blender.name() + ext)

    cv2.imwrite(smoothed_filename, blur_image.processedImage())
    cv2.imwrite(edge_filename, edge_finder.processedImage())
    cv2.imwrite(maskeed_filename, region_mask.processedImage())
    cv2.imwrite(hough_filename, hough_lines.processedImage())
    cv2.imwrite(blended_filename, image_blender.processedImage())

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
