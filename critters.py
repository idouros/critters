import os

import argparse
import configparser
import csv
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def convert_to_one_hot(labels):

    one_hot = np.zeros((labels.size, labels.max()+1))
    one_hot[np.arange(labels.size),labels] = 1
    return one_hot.transpose()

def load_dataset(config, sample_size, split_ratio):

    data_location = config['training_data']['location']
    training_data = config['training_data']['classes'].split(',')
    orientations_file = config['training_data']['orientations']

    # Work out the number of classes
    num_classes = len(training_data)
    classes = list(range(0, num_classes))

    # Build a dictionary of orientations
    orientations = {}
    with open(data_location + '/' + orientations_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            orientations[row[0]] = row[1]
    
    # Initialize the data placeholders
    X_in = [] #TODO change types so that slicing becomes easier when splitting between train/test sets
    Y_in = []

    # Load files and pre-process on the fly
    class_id = -1
    print("Loading images...")
    for image_class in training_data:
        class_id += 1
        image_directory = os.listdir(data_location + '/' + image_class)
        for image_file in image_directory:

            # Read
            image = np.array(plt.imread(data_location + '/' + image_class + '/' + image_file))

            # Resize to a standard size
            image_resized = np.array(Image.fromarray(image).resize(size=(sample_size, sample_size)))

            # Rotate according to specified orientation
            orientation = int(orientations[image_class + '/' + image_file])
            if orientation == 0:
                image_rotated = image_resized
            else:
                image_rotated = np.rot90(image_resized, (360-orientation)/90)

            # Add to set
            X_in.append(image_rotated)
            Y_in.append(class_id)

    num_input_samples = len(X_in)
    print("Loaded " + str(num_input_samples) + " images.")

    # Shuffle and split into training and test set
    num_training_set_samples = int(round(num_input_samples * split_ratio))
    num_test_set_samples = num_input_samples - num_training_set_samples
    print(str(num_training_set_samples) + " training samples")
    print(str(num_test_set_samples) + " test samples")

    shuffled_indices = np.arange(0, num_input_samples)
    random.shuffle(shuffled_indices)

    X_train_orig = np.zeros([num_training_set_samples, sample_size, sample_size,3], dtype=int)
    X_test_orig = np.zeros([num_test_set_samples, sample_size, sample_size, 3], dtype=int)
    Y_train_orig = np.zeros([1, num_training_set_samples], dtype=int)
    Y_test_orig = np.zeros([1 ,num_test_set_samples], dtype=int)

    for j in range (0, num_input_samples):
        if j < num_training_set_samples:
            X_train_orig[j] = X_in[shuffled_indices[j]] 
            Y_train_orig[0,j] = Y_in[shuffled_indices[j]]
        else:
            X_test_orig[j-num_training_set_samples] = X_in[shuffled_indices[j]]
            Y_test_orig[0,j-num_training_set_samples] = Y_in[shuffled_indices[j]]   

    # Normalize image vectors
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.
    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig)
    Y_test = convert_to_one_hot(Y_test_orig)

    return X_train, Y_train, X_test, Y_test, classes

def load_and_preprocess(image_file_name, sample_size):

    image = np.array(plt.imread(image_file_name))
    image_resized = np.array(Image.fromarray(image).resize(size=(sample_size, sample_size)))
    image_norm = image_resized/255.0
    image_flattened = image_norm.reshape(1, sample_size*sample_size*3).T
    return image_flattened.astype(np.float32)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', dest='config_file',  help='Load the specified config file and train the NN')
    args = parser.parse_args()

    # Data loading
    config = configparser.ConfigParser()
    config.read(args.config_file)
    sample_size = config.getint('architecture', 'sample_size', fallback = '64') 
    split_ratio = config.getfloat('training_data', 'split_ratio', fallback = '0.8') 
    X_train, Y_train, X_test, Y_test, classes = load_dataset(config, sample_size, split_ratio)


    #TODO Keras



    # Trying the inference: #TODO take this out of here, better user interface
    fnames = ["Data/cat/7.jpeg", "Data/horse/OIP-_6poWqxKgI1r0BVX9xCTaQHaEo.jpeg", "Data/squirrel/OIP-_kiyj8R2JYihtRF0_MURRQHaE8.jpeg", "IMG_20201025_101839.jpg"]
    for fname in fnames:
        test_image = load_and_preprocess(fname, sample_size)
        #predicted_class_id, confidence = predict_class_id(test_image, nn_params, layers)
        #print("Prediction: {} is an image depicting: {}, with {:2.2f}% confidence".format(fname, class_labels[predicted_class_id], confidence*100.0))

if __name__ == "__main__":
    main()