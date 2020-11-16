#TODO Migrate to latest Python version, and latest TensorFlow
#TODO Make it use CUDA
#TODO Improve the test set accuracy

import os
import argparse
import configparser
import csv
import random

import numpy as np
import h5py
import matplotlib.pyplot as plt

import scipy
from scipy import ndimage

import PIL
from PIL import Image

import tensorflow as tf
from tensorflow.python.framework import ops

IM_SIZE = 64

def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape = [n_x, None], name = "X")
    Y = tf.placeholder(tf.float32, shape = [n_y, None], name = "Y")
    return X, Y

def initialize_parameters(layers):
   
    tf.set_random_seed(1)                   
        
    parameters = {"W1": tf.get_variable("W1", [layers[0],IM_SIZE*IM_SIZE*3], initializer = tf.contrib.layers.xavier_initializer(seed = 1)),
                  "b1": tf.get_variable("b1", [layers[0],1], initializer = tf.zeros_initializer())}

    for i in range(1, len(layers)):
        parameters["W"+str(i+1)] = tf.get_variable("W"+str(i+1), [layers[i],layers[i-1]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        parameters["b"+str(i+1)] = tf.get_variable("b"+str(i+1), [layers[i],1], initializer = tf.zeros_initializer())

    return parameters

def forward_propagation(X, parameters, layers):
    
    # Retrieve the parameters from the dictionary
    W = parameters['W1']
    b = parameters['b1']
    A = X

    for i in range(1, len(layers)):
        Z = tf.add(tf.matmul(W, A), b)   
        A = tf.nn.relu(Z)
        W = parameters["W"+str(i+1)]
        b = parameters["b"+str(i+1)]

    Z_final = tf.add(tf.matmul(W, A), b)
    # (no activation for the final layer, will do softmax later on)
   
    return Z_final

def compute_cost(Z_final, Y):

    logits = tf.transpose(Z_final)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    return cost

def model(X_train, Y_train, X_test, Y_test, layers, learning_rate = 0.0001,
          num_iters = 1500, print_cost = True):
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters(layers)
    Z_final = forward_propagation(X, parameters, layers)
    cost = compute_cost(Z_final, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        
        sess.run(init) #TODO introduce minibatches to accelerate training
        for iteration in range(num_iters):
            _ , iteration_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})

            # Print the cost every iteration
            if print_cost == True and iteration % 100 == 0:
                print ("Cost after iteration %i: %f" % (iteration, iteration_cost))

        parameters = sess.run(parameters)
        print ("Network training complete.")

        # Calculate accuracy
        correct_prediction = tf.equal(tf.argmax(Z_final), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
    return parameters

def convert_to_one_hot(labels, C):
   
    one_hot = np.zeros((labels.size, labels.max()+1))
    one_hot[np.arange(labels.size),labels] = 1
    return one_hot.transpose()

def load_dataset(config):

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
            image = np.array(ndimage.imread(data_location + '/' + image_class + '/' + image_file, flatten = False))

            # Resize to a standard size
            image_resized = scipy.misc.imresize(image, size=(IM_SIZE,IM_SIZE))

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
    training_set_indices = []
    r = 0.8 #training_set_ratio
    num_training_set_samples = int(round(num_input_samples * r))
    num_test_set_samples = num_input_samples - num_training_set_samples
    print(str(num_training_set_samples) + " training samples")
    print(str(num_test_set_samples) + " test samples")

    shuffled_indices = np.arange(0, num_input_samples)
    random.shuffle(shuffled_indices)

    X_train_orig = np.zeros([num_training_set_samples,IM_SIZE,IM_SIZE,3], dtype=int)
    X_test_orig = np.zeros([num_test_set_samples,IM_SIZE,IM_SIZE,3], dtype=int)
    Y_train_orig = np.zeros([1,num_training_set_samples], dtype=int)
    Y_test_orig = np.zeros([1,num_test_set_samples], dtype=int)

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
    sm = tf.nn.softmax(logits, axis = 0)
    with tf.Session() as sess:
        scores = sess.run(sm)
        #print(scores)
        return np.argmax(scores), np.max(scores)

def load_and_preprocess(image_file_name):

    image = np.array(ndimage.imread(image_file_name, flatten=False))
    image_resized = scipy.misc.imresize(image, size=(IM_SIZE,IM_SIZE))
    image_norm = image_resized/255.0
    image_flattened = image_norm.reshape(1, IM_SIZE*IM_SIZE*3).T
    return image_flattened.astype(np.float32)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train',   dest='config_file',          help='Load the specified config file and train the NN')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)
    class_labels=config['training_data']['classes'].split(',')
    
    X_train, Y_train, X_test, Y_test, classes = load_dataset(config)
    
    hidden_layers = config['architecture']['hidden_layers'].split(',')
    layers = hidden_layers
    layers.append(len(classes))
    print("Model will be trained with " + str(len(layers)) + " layers.")    
    learning_rate = float(config['hyperparameters']['learning_rate'])
    num_iters = int(config['hyperparameters']['num_iters'])

    # Optimization loop
    nn_params = model(X_train, Y_train, X_test, Y_test, layers, learning_rate, num_iters)

    # Trying the inference: #TODO take this out of here, better user interface
    fnames = ["Data/cat/7.jpeg", "Data/horse/OIP-_6poWqxKgI1r0BVX9xCTaQHaEo.jpeg", "Data/squirrel/OIP-_kiyj8R2JYihtRF0_MURRQHaE8.jpeg", "IMG_20201025_101839.jpg"]
    for fname in fnames:
        test_image = load_and_preprocess(fname)
        predicted_class_id, confidence = predict_class_id(test_image, nn_params, layers)
        print("Prediction: {} is an image depicting: {}, with {:2.2f}% confidence".format(fname, class_labels[predicted_class_id], confidence*100.0))

if __name__ == "__main__":
    main()