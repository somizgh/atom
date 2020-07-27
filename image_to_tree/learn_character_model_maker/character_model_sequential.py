from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import variables
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation, Flatten, Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf



input_image_size = (variables.LENGTH_OF['RESIZED_IMAGE_HORIZONTAL'], variables.LENGTH_OF['RESIZED_IMAGE_VERTICAL'], 1)

dataX = np.load("./learn_character_model_data/data_X.npy")
dataY = np.load("./learn_character_model_data/data_Y.npy")



scaler=MinMaxScaler()
dataX[:] = scaler.fit_transform(dataX[:])
dataY[:] = scaler.fit_transform(dataY[:])

dataX=dataX.reshape((len(dataX),variables.LENGTH_OF['RESIZED_IMAGE_HORIZONTAL'],variables.LENGTH_OF['RESIZED_IMAGE_VERTICAL'],1))


print("dataX.shape : ", dataX.shape)
print("dataY.shape : ", dataY.shape)


train_dataX, test_dataX, train_dataY, test_dataY = train_test_split(dataX, dataY, test_size=0.3)

print(train_dataX[0],train_dataY[0])

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

#model.add(Dense(200))
#model.add(Activation("relu"))

model.add(Dense(variables.NUMBER_OF['TOTAL']))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.0001), metrics=["accuracy"])

cb_checkpoint = ModelCheckpoint(filepath="./learn_character_model", monitor='val_loss',
                                verbose=1, save_best_only=True)

cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10)

hist = model.fit(train_dataX, train_dataY, batch_size=32, epochs=variables.EPOCH, validation_data=(test_dataX, test_dataY), verbose=1, callbacks=[cb_checkpoint, cb_early_stopping])

loss, accuracy = model.evaluate(test_dataX, test_dataY, batch_size = 64, verbose = 1)
print("Accuracy = {}%, ".format(accuracy*100))

model.save('character106_model.h5')

model_json = model.to_json()
with open("character106_model.json", "w") as json_file:
    json_file.write(model_json)


fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()





