import os
import argparse
import configparser
import csv
import random
import math

import numpy as np
import matplotlib.pyplot as plt

import PIL
from PIL import Image

import tensorflow as tf
from tensorflow.python.framework import ops

def initialize_parameters(layers, sample_size):
   
    tf.random.set_seed(1)             

    parameters = [tf.Variable(tf.initializers.GlorotUniform(seed = 1)(shape = (layers[0],sample_size*sample_size*3)), name = "W1"),
                  tf.Variable(tf.zeros_initializer()(shape = (layers[0],1)), name = "b1")]

    for i in range(1, len(layers)):
        parameters.append(tf.Variable(tf.initializers.GlorotUniform(seed = 1)(shape = (layers[i],layers[i-1])), name = "W"+str(i+1)))
        parameters.append(tf.Variable(tf.zeros_initializer()(shape = (layers[i],1)), name = "b"+str(i+1)))

    return parameters

def forward_propagation(X, parameters, layers):
    
    # Retrieve the parameters from the dictionary
    W = parameters[0]
    b = parameters[1]
    A = X

    for i in range(1, len(layers)):
        Z = tf.add(tf.matmul(W, A), b)   
        A = tf.nn.relu(Z)
        W = parameters[(i*2)]
        b = parameters[(i*2)+1]

    Z_final = tf.add(tf.matmul(W, A), b)
    # (no activation for the final layer, will do softmax later on)
   
    return Z_final

def compute_cost(Z_final, Y, parameters, l2_reg = 0):

    logits = tf.transpose(Z_final)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)) 
    for i in range(0, math.floor(len(parameters)/2)):
        cost += l2_reg * tf.nn.l2_loss(parameters[i*2])
    return cost

def train_model(X_train, Y_train, X_test, Y_test, layers, learning_rate, num_iters, sample_size, l2_reg, print_cost = True):
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.random.set_seed(1)                             # to keep consistent results
    parameters = initialize_parameters(layers, sample_size)
    optimizer = tf.optimizers.Adam(learning_rate = learning_rate)
    for iteration in range(num_iters):
        
        with tf.GradientTape() as t:
            Z_final = forward_propagation(X_train, parameters, layers)
            current_cost = compute_cost(Z_final, Y_train, parameters, l2_reg)
        grads = t.gradient(current_cost, parameters)

        optimizer.apply_gradients(zip(grads, parameters))
        if print_cost == True and iteration % 100 == 0:
            print ("Cost after iteration %i: %f" % (iteration, current_cost.numpy()))
    print ("Network training complete.")

    # Calculate accuracy
    print ("Train Accuracy:", tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Z_final), tf.argmax(Y_train)), "float")).numpy())

    Z_test = forward_propagation(X_test, parameters, layers)
    print ("Test Accuracy:", tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Z_test), tf.argmax(Y_test)), "float")).numpy())
        
    return parameters

def convert_to_one_hot(labels, C):
   
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

    # Flatten the training and test images
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    # Normalize image vectors
    X_train = X_train_flatten/255.
    X_test = X_test_flatten/255.
    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, num_classes)
    Y_test = convert_to_one_hot(Y_test_orig, num_classes)

    return X_train, Y_train, X_test, Y_test, classes

def predict_class_id(image, nn_params, layers):

    logits = forward_propagation(image, nn_params, layers)
    scores = tf.nn.softmax(logits, axis = 0)
    return np.argmax(scores), np.max(scores)

def load_and_preprocess(image_file_name, sample_size):

    image = np.array(plt.imread(image_file_name))
    image_resized = np.array(Image.fromarray(image).resize(size=(sample_size, sample_size)))
    image_norm = image_resized/255.0
    image_flattened = image_norm.reshape(1, sample_size*sample_size*3).T
    return image_flattened.astype(np.float32)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train',   dest='config_file',          help='Load the specified config file and train the NN')
    args = parser.parse_args()

    # Data loading
    config = configparser.ConfigParser()
    config.read(args.config_file)
    sample_size = config.getint('hyperparameters', 'sample_size', fallback = '64') 
    split_ratio = config.getfloat('training_data', 'split_ratio', fallback = '0.8') 
    X_train, Y_train, X_test, Y_test, classes = load_dataset(config, sample_size, split_ratio)

    # Mandatory
    class_labels=config['training_data']['classes'].split(',')
    hidden_layers = config['architecture']['hidden_layers'].split(',')
    layers = list(map(int, hidden_layers))
    layers.append(len(classes))
    print("Model will be trained with " + str(len(layers)) + " layers.")    

    # Optional
    learning_rate = config.getfloat('hyperparameters', 'learning_rate', fallback = '0.001')
    num_iters = config.getint('hyperparameters', 'num_iters', fallback = '1500')
    l2_reg = config.getfloat('hyperparameters', 'l2_reg', fallback = '0.0')

    # Training the NN
    nn_params = train_model(X_train, Y_train, X_test, Y_test, layers, learning_rate, num_iters, sample_size, l2_reg)

    # Trying the inference: #TODO take this out of here, better user interface
    fnames = ["Data/cat/7.jpeg", "Data/horse/OIP-_6poWqxKgI1r0BVX9xCTaQHaEo.jpeg", "Data/squirrel/OIP-_kiyj8R2JYihtRF0_MURRQHaE8.jpeg", "IMG_20201025_101839.jpg"]
    for fname in fnames:
        test_image = load_and_preprocess(fname, sample_size)
        predicted_class_id, confidence = predict_class_id(test_image, nn_params, layers)
        print("Prediction: {} is an image depicting: {}, with {:2.2f}% confidence".format(fname, class_labels[predicted_class_id], confidence*100.0))

if __name__ == "__main__":
    main()