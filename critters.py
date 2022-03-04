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

def load_dataset(config, sample_size, split_ratio):
    """load"""
    data_location = config['training_data']['location_aligned']
    training_data = config['training_data']['classes'].split(',')

    # Work out the number of classes
    num_classes = len(training_data)
    classes = list(range(0, num_classes))

    # Initialize the data placeholders
    X_in = []
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

            # Add to set
            X_in.append(image_resized)
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

    X_train = np.zeros([num_training_set_samples, sample_size, sample_size,3], dtype=int)
    X_test = np.zeros([num_test_set_samples, sample_size, sample_size, 3], dtype=int)
    Y_train = np.zeros([num_training_set_samples], dtype=int)
    Y_test = np.zeros([num_test_set_samples], dtype=int)

    for j in range (0, num_input_samples):
        if j < num_training_set_samples:
            X_train[j] = X_in[shuffled_indices[j]]/255.
            Y_train[j] = Y_in[shuffled_indices[j]]
        else:
            X_test[j-num_training_set_samples] = X_in[shuffled_indices[j]]/255.
            Y_test[j-num_training_set_samples] = Y_in[shuffled_indices[j]]

 #   print(Y_train[:5])
    return X_train, Y_train, X_test, Y_test, classes
    
def load_and_preprocess(image_file_name, sample_size):
    """ load and preprocess"""
    image = np.array(plt.imread(image_file_name))
    image_resized = np.array(Image.fromarray(image).resize(size=(sample_size, sample_size)))
    image_norm = image_resized/255.0
    return image_norm.astype(np.float32)

def main():
    """main"""
    print("TensorFlow version:", tf.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', dest='config_file',  help='Load the specified config file and train the NN')
    args = parser.parse_args()

    # Data loading
    config = configparser.ConfigParser()
    config.read(args.config_file)
    sample_size = config.getint('training_data', 'sample_size', fallback = '64')
    split_ratio = config.getfloat('training_data', 'split_ratio', fallback = '0.8')
    class_labels = config.get('training_data', 'classes')
    X_train, Y_train, X_test, Y_test, classes = load_dataset(config, sample_size, split_ratio)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(sample_size, sample_size, 3)),
        #tf.keras.layers.Dense(2048, activation='relu'),
        #tf.keras.layers.Dropout(0.4),
        #tf.keras.layers.Dense(1024, activation='relu'),
        #tf.keras.layers.Dropout(0.2),
        #tf.keras.layers.Dense(1024, activation='relu'),
        #tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.025),
        tf.keras.layers.Dense(3)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy']
        )
    model.fit(X_train, Y_train, epochs=5)
    model.evaluate(X_test,  Y_test, verbose=2)
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    
#    predictions = model(X_train[:5]).numpy()
#    print(predictions)
#    print(tf.nn.softmax(predictions).numpy())

    # Trying the inference
    fnames = [
                "Data/Aligned/cat/cat_0000.jpg",
                "Data/Aligned/horse/horse_0000.jpg",
                "Data/Aligned/squirrel/squirrel_0000.jpg",
                "IMG_20201025_101839.jpg"
                ]
    
    for fname in fnames:
        test_image = load_and_preprocess(fname, sample_size)
        test_image = np.expand_dims(test_image, axis=0)
        predictions = probability_model(test_image).numpy()
        confidence = np.max(predictions)
        predicted_class_id = np.argmax(predictions)
        print(predictions)
        print("Prediction: {} is an image depicting: {}, with {:2.2f}% confidence".format(
            fname, 
            class_labels[predicted_class_id], 
            confidence*100.0
            ))

if __name__ == "__main__":
    main()
