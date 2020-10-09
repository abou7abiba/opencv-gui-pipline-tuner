"""
usage: 
python pipeline_utils.py [-h] [-c CONFIG_FILE] [-o OUTPUT_FOLDER] [-d] filename

Visualizes the different stages for image processing pipelines and control the needed
parameters.

positional arguments:
  filename              Input file either text, image or video

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_FILE, --config CONFIG_FILE
                        a JSON file to load the initial configuration
  -o OUTPUT_FOLDER, --out OUTPUT_FOLDER
                        The folder name for the output of the tool.
  -d, --debug           Enable the debug mode for logging debug statements.
"""
import logging
import logging.config

logging.config.fileConfig("logging.conf")
logger = logging.getLogger(__name__)

import argparse
import cv2

from pipeline_utils import FindLanesPipeline

# create logger
logger = logging.getLogger(__name__)

def __init_logger(debug = False):
    #Set level to logging.DEBUG to see CRITICAL, ERROR, WARNING, INFO and DEBUG statements
    #Set level to logging.ERROR to see the CRITICAL & ERROR statements only
    logger.setLevel(logging.DEBUG)

    debugLevel = logging.ERROR
    if debug:
        debugLevel = logging.DEBUG

    for handler in logger.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel (debugLevel)


def main():
    parser = argparse.ArgumentParser(description='Visualizes the different stages for image processing pipelines and control the needed parameters.')
    parser.add_argument('filename', help="Input file either text, image or video")
    parser.add_argument('-c', '--config', default='pipeline-config.json', dest='config_file', help="a JSON file to load the initial configuration " )
    parser.add_argument('-o', '--out', default='output_images', dest='output_folder', help="The folder name for the output of the tool." )
    parser.add_argument('-d', '--debug', action="store_true", help="Enable the debug mode for logging debug statements." )

    args = parser.parse_args()


    __init_logger(args.debug)
    logger.debug("arguments are: %s", args)

    pipeline = FindLanesPipeline ('Find Lanes Pipeline', args.filename, args.output_folder, args.config_file)
    pipeline.processImage()
    
    cv2.waitKey(0)

    print ("Adjust the parameters as desired.  Hit any key to close. Your configuration will be in Tuning_info_out.log")

    pipeline.save()

    logger.info("Configuration parameters:\n %s ", pipeline)
    logger.debug("Configuration parameters:\n %s ", pipeline)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
