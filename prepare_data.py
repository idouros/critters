import os
import shutil

import argparse
import configparser
import csv

import numpy as np
import matplotlib.pyplot as plt

def prepare_dataset(config):
    """prepare"""

    data_location_in = config['training_data']['location_original']
    data_location_out = config['training_data']['location_aligned']
    training_data = config['training_data']['classes'].split(',')
    orientations_file = config['training_data']['orientations']

    # Wipe previous
    if os.path.isdir(data_location_out):
        shutil.rmtree(data_location_out)

    # Build a dictionary of orientations
    orientations = {}
    with open(data_location_in + '/' + orientations_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            orientations[row[0]] = row[1]

    # Load files and pre-process
    class_id = -1
    print("Loading images...")
    os.mkdir(data_location_out)
    for image_class in training_data:
        class_id += 1
        image_directory_in = os.listdir(data_location_in + '/' + image_class)
        image_directory_out = data_location_out + '/' + image_class
        os.mkdir(image_directory_out)
        sample_id = -1
        for image_file in image_directory_in:
            # Read
            image = np.array(plt.imread(data_location_in + '/' + image_class + '/' + image_file))

            # Rotate according to specified orientation
            orientation = int(orientations[image_class + '/' + image_file])
            if orientation == 0:
                image_rotated = image
            else:
                image_rotated = np.rot90(image, (360-orientation)/90)

            # Save
            sample_id += 1
            plt.imsave(image_directory_out + '/' + image_class + '_' + f'{sample_id:04}' + '.jpg', image_rotated)

def main():
    """main"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', dest='config_file',  help='Load the specified config file and train the NN')
    args = parser.parse_args()

    # Data loading
    config = configparser.ConfigParser()
    config.read(args.config_file)
    prepare_dataset(config)

if __name__ == "__main__":
    main()
