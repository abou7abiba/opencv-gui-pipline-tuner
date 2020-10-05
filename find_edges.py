"""
How to run:
python find_edges.py <image path>
"""

import argparse
import cv2
import os

from guiutils import SmoothImage, EdgeFinder


def main():
    parser = argparse.ArgumentParser(description='Visualizes the line for hough transform.')
    parser.add_argument('filename')

    args = parser.parse_args()

    # img = cv2.imread(args.filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(args.filename)   
    cv2.imshow('input', img)

    blur_image = SmoothImage('Blur', img, average_filter_size=5, gaussian_filter_size=1, median_filter_size=1, bilateral_filter_size=1)
    
    edge_finder = EdgeFinder('Edge', blur_image.processedImage(), min_threshold=28, max_threshold=115)
    blur_image.addProcessor (edge_finder.onImageChange)
    
    # will inforce refresh for all pipeline.
    blur_image.refresh()
    
    cv2.waitKey(0)

    # print ("Edge parameters: %s")
    # print ("GaussianBlur Filter Size: %f" % edge_finder.filterSize())
    # print ("Threshold2: %f" % edge_finder.threshold2())

    (head, tail) = os.path.split(args.filename)

    (root, ext) = os.path.splitext(tail)

    smoothed_filename = os.path.join("output_images", root + "-smoothed" + ext)
    edge_filename = os.path.join("output_images", root + "-edges" + ext)

    cv2.imwrite(smoothed_filename, edge_finder.processedImage())
    cv2.imwrite(edge_filename, edge_finder.processedImage())

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
