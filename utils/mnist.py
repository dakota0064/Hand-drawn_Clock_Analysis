from os import path
import os
import time
import cv2
import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

''' Trains an MNIST model for digit prediction. This shouldn't need to be rerun unless you're conducting your
    own experiments and attempting to improve the digit recognition capabilities
'''

###########################################################################################################

import numpy as np
import matplotlib.pyplot as plt

def section_print():
    '''Memorized function keeping track of section number'''
    section_number = 0

    def __inner(message):
        nonlocal section_number
        section_number += 1
        print('Section {}: {}'.format(section_number, message))
    print('Section {}: Initializing section function'.format(section_number))
    return __inner


def get_bounding_box(grad, threshold):
    """Get the bounding box around a digit, expressed as x and y coordinates
    """
    exceeds_threshold = np.absolute(grad) > threshold
    diff = np.diff(exceeds_threshold)
    boundaries = []
    for i, (e, d) in enumerate(zip(exceeds_threshold, diff)):
        breaks = np.where(d)[0]
        assert breaks.shape[0] > 0
        if e[0]:
            breaks = np.array([0, breaks[0]])
        if e[-1]:
            breaks = np.array([breaks[0], d.shape[0]])
        breaks[0] = breaks[0] + 1
        boundary = (breaks[0], breaks[-1])
        boundaries.append(boundary)
    return np.array(boundaries)


def get_data_to_box(df, threshold=.1):
    """get bounding boxes for all digits in dataset df
    """
    z_grad, y_grad, x_grad = np.gradient(df)
    y_grad_1d = y_grad.sum(axis=2)
    x_grad_1d = x_grad.sum(axis=1)

    y_bounds = get_bounding_box(y_grad_1d, threshold)
    x_bounds = get_bounding_box(x_grad_1d, threshold)

    return np.hstack([x_bounds, y_bounds])


def plot_bounding_box(data, bounds, index):
    """Plot the image and bounding box indexed by index
    """

    # Calculate bounding box
    print(index)
    x_bound = bounds[index, 0:2] * data.shape[1]
    y_bound = bounds[index, 2:4] * data.shape[2]

    # Plot image
    img = plt.imshow(data[index], cmap='Greys')

    # Plot calculated bounding box
    plt.plot([x_bound[0]] * 2, y_bound, color='r')
    plt.plot(x_bound, [y_bound[0]] * 2, color='r')
    plt.plot([x_bound[1]] * 2, y_bound, color='r')
    plt.plot(x_bound, [y_bound[1]] * 2, color='r')

    return img


def plot_bounding_grid(df, subplot_shape, bounding_boxes):
    fig, axes = plt.subplots(*subplot_shape)
    for ax in axes.ravel():
        rand_index = np.random.randint(0, df.shape[0])
        ax.imshow(df[rand_index], cmap='Greys')
        ax.axis('off')

        # Calculate bounding box
        x_bound = bounding_boxes[rand_index, 0:2] * df.shape[1]
        y_bound = bounding_boxes[rand_index, 2:4] * df.shape[2]

        # Plot calculated bounding box
        ax.plot([x_bound[0]] * 2, y_bound, color='r')
        ax.plot(x_bound, [y_bound[0]] * 2, color='r')
        ax.plot([x_bound[1]] * 2, y_bound, color='r')
        ax.plot(x_bound, [y_bound[1]] * 2, color='r')

    fig.tight_layout()
    fig.subplots_adjust(wspace=-.18, hspace=-.18)
    return fig


def reshape_to_img(df):
    """Reshape 1d images to 2d
    """
    m = df.shape[0]
    i = np.int(np.sqrt(df.shape[1]))
    return df.reshape((m, i, i))


def get_bounding_boxes(mnist):
    """Get bounding boxes for the MNIST train, validation and test sets
    """
    train = reshape_to_img(mnist.train.images)
    validation = reshape_to_img(mnist.validation.images)
    test = reshape_to_img(mnist.test.images)

    bounds_train = get_data_to_box(train) * 1. / 28
    bounds_validation = get_data_to_box(validation) * 1. / 28
    bounds_test = get_data_to_box(test) * 1. / 28

    return bounds_train, bounds_validation, bounds_test

###########################################################################################################

def build_bounding_model(num_classes):
    model = Sequential(
        [
            Input(shape=(28, 28, 1)),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(num_classes, activation="softmax")
        ]
    )
    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    return model

def train_model(model, train_data, test_data, train_labels, test_labels):
    model.fit(train_data, train_labels, batch_size=128, epochs=30, validation_split=0.1)

    score = model.evaluate(test_data, test_labels, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

##################################################################################################################

# the data, split between train and test sets
num_classes = 10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#print("Generating Garbage")
#garbage_images = generate_garbage()
#cv2.imshow("garbage", garbage_images[0])
#cv2.waitKey(0)
#random.shuffle(garbage_images)
#garbage_labels = np.ones((len(garbage_images))) * 10

#x_train = np.concatenate((x_train, garbage_images[:-1000, :, :]), axis=0)
#y_train = np.concatenate((y_train, garbage_labels[:-1000]), axis=0)

#x_test = np.concatenate((x_test, garbage_images[-1000:, :, :]), axis=0)
#y_test = np.concatenate((y_test, garbage_labels[-1000:]), axis=0)

print("Shuffling")
train = list(zip(x_train, y_train))
random.shuffle(train)
x_train, y_train = zip(*train)

test = list(zip(x_test, y_test))
random.shuffle(test)
x_test, y_test = zip(*test)

x_train = np.array(x_train)
x_test = np.array(x_test)

for i in range(len(x_train)):
    blurred = cv2.GaussianBlur(x_train[i], (3, 3), 0)
    x_train[i] = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)[1]

for i in range(len(x_test)):
    blurred = cv2.GaussianBlur(x_test[i], (3, 3), 0)
    x_test[i] = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)[1]

cv2.imshow("train", x_train[0])
cv2.waitKey(0)

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
# print("x_train shape:", x_train.shape)
# print(x_train.shape[0], "train samples")
# print(x_test.shape[0], "test samples")

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
model = build_bounding_model(num_classes)

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

train_model(model, x_train, x_test, y_train, y_test)
model.save("data/mnist_threshed_classifier.h5")