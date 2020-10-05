"""
Example to explain argument parser in python
"""

import argparse

def main():
    parser = argparse.ArgumentParser(description='This example for a tool to process a file and configure the tool using a config file.')
    parser.add_argument('filename', help="Input file either text, image or video")
    # parser.add_argument('config_file', help="a JSON file to load the initial configuration ")
    # parser.add_argument('-c', '--config_file', help="a JSON file to load the initial configuration ", default='configFile.json', required=False)
    parser.add_argument('-c', '--config', default='configFile.json', dest='config_file', help="a JSON file to load the initial configuration " )
    parser.add_argument('-d', '--debug', action="store_true", help="Enable the debug mode for logging debug statements." )

    args = parser.parse_args()
    
    filename = args.filename
    configfile = args.config_file

    print("The file to be processed is", filename)
    print("The config file is", configfile)

    if args.debug:
        print("Debug mode enabled")
    else:
        print("Debug mode disabled")

    print("and all arguments are: ", args)


if __name__ == '__main__':
    main()
