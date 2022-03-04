"""builds model for identifying between three different species"""
import os

import argparse
import configparser

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_dataset(config, sample_size, split_ratio):
    """load"""
    data_location = config['training_data']['location_aligned']
    training_data = config['training_data']['classes'].split(',')

    # Initialize the data placeholders
    num_samples_train = [0] * len(training_data)
    num_samples_test = [0] * len(training_data)
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    # Load files and pre-process on the fly
    class_id = -1
    print("Loading images...")
    for image_class in training_data:
        class_id += 1
        image_directory = os.listdir(data_location + '/' + image_class)

        num_samples = len(image_directory)
        num_samples_train[class_id] = int(round(num_samples * split_ratio))
        num_samples_test[class_id] = num_samples - num_samples_train[class_id]

        image_id = -1
        for image_file in image_directory:
            # Read
            image = np.array(plt.imread(data_location + '/' + image_class + '/' + image_file))

            # Resize to a standard size
            image_resized = np.array(Image.fromarray(image).resize(size=(sample_size, sample_size)))

            # Add to the correct set
            image_id += 1
            if image_id < num_samples_train[class_id]:
                x_train.append(image_resized/255.0)
                y_train.append(class_id)
            else:
                x_test.append(image_resized/255.0)
                y_test.append(class_id)

    print("Training samples:")
    print(num_samples_train)
    print("Test samples:")
    print(num_samples_test)
    print("Total samples: " + str(sum(num_samples_train)+sum(num_samples_test)))

    print(y_train[:5])
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def load_and_preprocess(image_file_name, sample_size):
    """ load and preprocess"""
    image = np.array(plt.imread(image_file_name))
    image_resized = np.array(Image.fromarray(image).resize(size=(sample_size, sample_size)))
    image_norm = image_resized/255.0
    return image_norm

def main():
    """main"""
    print("TensorFlow version:", tf.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', dest='config_file',  help='config file')
    args = parser.parse_args()

    # Data loading
    config = configparser.ConfigParser()
    config.read(args.config_file)
    sample_size = config.getint('training_data', 'sample_size', fallback = '64')
    split_ratio = config.getfloat('training_data', 'split_ratio', fallback = '0.8')
    class_labels = config.get('training_data', 'classes').split(",")
    x_train, y_train, x_test, y_test = load_dataset(config, sample_size, split_ratio)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(sample_size, sample_size, 3)),
        #tf.keras.layers.Dense(2048, activation='relu'),
        #tf.keras.layers.Dropout(0.4),
        #tf.keras.layers.Dense(1024, activation='relu'),
        #tf.keras.layers.Dropout(0.2),
        #tf.keras.layers.Dense(1024, activation='relu'),
        #tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(256, activation='relu'),
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
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test,  y_test, verbose=2)
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    print(model(x_train[:5]).numpy())
    print(probability_model(x_train[:5]).numpy())

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
        print(f"Prediction: {fname} is an image depicting: { class_labels[predicted_class_id]}, with {(confidence*100.0):2.2f}% confidence")

if __name__ == "__main__":
    main()
