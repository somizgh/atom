from character_vocabulary import CharVocabulary
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Activation, Flatten, Dense
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import variables as VARIABLES
import useful_function as UF
from separate_sentence_straight_model_variables import *

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

def SSSM(input_image_size,out_put_size):
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

    model.add(Dense(out_put_size))
    model.add(Activation("softmax"))
    return model


def learn_SSSM(dataX, dataY):
    input_image_size = (IMAGE_WIDTH, IMAGE_HEIGHT, 1)
    output_layer_len = 3
    dataX = dataX/255
    dataX = dataX.reshape((len(dataX), IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    train_dataX, test_dataX, train_dataY, test_dataY = train_test_split(dataX, dataY, test_size=0.3)

    model = SSSM(input_image_size, output_layer_len)
    model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.0001), metrics=["accuracy"])

    model_dir = VARIABLES.SSSM_MODEL_PREFIX + "\\" + UF.time_format()
    UF.create_folder(model_dir)

    cb_checkpoint = ModelCheckpoint(filepath=model_dir, monitor='val_loss',
                                    verbose=1, save_best_only=False)

    cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    hist = model.fit(train_dataX, train_dataY, batch_size=32, epochs=EPOCH,
                     validation_data=(test_dataX, test_dataY), verbose=2, callbacks=[cb_checkpoint, cb_early_stopping])

    loss, accuracy = model.evaluate(test_dataX, test_dataY, batch_size=64, verbose=1)

    print("Accuracy = {}%, ".format(accuracy * 100))

    #json.dump(hist.history, open(os.path.join(model_dir, VARIABLES.SSSM_HISTORY_JSON_NAME), 'w'))
    #json.dump(model.to_json(), open(os.path.join(model_dir, VARIABLES.SSSM_JSON_MODEL_NAME), 'w'))
    model.save(os.path.join(model_dir, VARIABLES.SSSM_H5_NAME))
    """
    model_json = model.to_json()
    with open(os.path.join(model_dir, VARIABLES.SSSM_JSON_MODEL_NAME), "w") as json_file:
        json_file.write(model_json)
    """
    fig, loss_ax = plt.subplots()

    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

    acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    plt.show()

    return 0


def from_image_save_numpy(dir_path, number_of_images_use_option):
    data_x = [[[]]]
    data_y = [[[]]]
    numpy_files = os.listdir(dir_path)
    if number_of_images_use_option is "max":
        number_of_file_use = int(len(numpy_files)/2)
    elif number_of_images_use_option/IMAGES_PER_FILE > len(numpy_files)/2:
        print("number of images use {} more than files {}".format(number_of_images_use_option, len(numpy_files)/2))
    else:
        number_of_file_use = math.ceil(number_of_images_use_option/IMAGES_PER_FILE)
    for i in range(number_of_file_use):
        datax = np.load(os.path.join(dir_path, INPUT_DATA_X_NAME+INPUT_DATA_COUNTER_FORMAT.format(i+1)))
        datay = np.load(os.path.join(dir_path, INPUT_DATA_Y_NAME+INPUT_DATA_COUNTER_FORMAT.format(i+1)))
        if i == 0:
            data_x = datax
            data_y = datay
            continue
        data_x = np.concatenate((data_x, datax), axis=0)
        data_y = np.concatenate((data_y, datay), axis=0)
    return data_x, data_y


if __name__ == "__main__":
    print("a")

