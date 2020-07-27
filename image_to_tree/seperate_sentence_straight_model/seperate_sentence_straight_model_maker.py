
import variables
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import SGD

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
import seperate_sentence_straight_model_variables as SSSMV
from useful_function import *
from variables import*
import cv2



"""
model = Sequential()
model.add(Conv2D(100, (5, 5), padding="same", input_shape=input_image_size))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(50, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))
"""

def SSSM():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(100, (5,5), padding="same", input_shape=2),
        tf.keras.layers.Conv2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense()
    ])
    return model

def make_learn_model():

    return 0

def resize_image(image):
    return image

def load_images_as_np(path):
    counter = 0
    image_numpy = np.array([])
    for root, dirs, files in os.walk(path):
        root_path = os.path.join(os.path.abspath(path), root)
        for file in files:
            file_path = os.path.join(root_path, file)
            image = cv2.imread(file_path)
            image = resize_image(image)
            counter = counter + 1
            if counter > SSSMV.NUMBER_OF_IMAGE_LOAD:
                return


if __name__ == "__main__":
    image_numpy = load_images_as_np(SSSM)

    return 0
