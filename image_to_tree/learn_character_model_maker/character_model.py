import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from variables import *
import numpy as np
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

input_image_size = (LENGTH_OF['RESIZED_IMAGE_HORIZONTAL'], LENGTH_OF['RESIZED_IMAGE_VERTICAL'], 1)

dataX = np.load("./learn_character_model_data/data_X.npy")
dataY = np.load("./learn_character_model_data/data_Y.npy")


scaler=MinMaxScaler()
dataX[:] = scaler.fit_transform(dataX[:])
dataY[:] = scaler.fit_transform(dataY[:])

dataX=dataX.reshape((len(dataX),LENGTH_OF['RESIZED_IMAGE_HORIZONTAL'],LENGTH_OF['RESIZED_IMAGE_VERTICAL'],1))
dataY=dataY.reshape((len(dataY),1))

print("dataX.shape : ",dataX.shape)
print("dataY.shape : ",dataY.shape)


train_dataX, test_dataX, train_dataY, test_dataY = train_test_split(dataX, dataY, test_size=0.3)

def mapping_fn(X,Y=None):
    input, out = {'x': X}, Y
    return input, out


def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((train_dataX, train_dataY))

    dataset = dataset.shuffle(buffer_size=len(train_dataX))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(mapping_fn)
    dataset = dataset.repeat(count=EPOCH)

    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()


def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((test_dataX, test_dataY))
    dataset = dataset.shuffle(buffer_size=len(test_dataX))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(mapping_fn)

    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()

def model_fn(features, labels, mode):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT


    conv_layer1 = tf.keras.layers.Conv2D(filters=10, kernel_size=(4, 4), activation=tf.nn.relu)(features['x'])
    pooling_layer1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3))(conv_layer1)
    flatten_layer1 = tf.keras.layers.Flatten()(pooling_layer1)
    hidden_layer1 = tf.keras.layers.Dense(units=30, activation=tf.nn.sigmoid)(flatten_layer1)
    output_layer = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)(hidden_layer1)

    if PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'prob': tf.nn.sigmoid(output_layer)})

    loss = tf.losses.sigmoid_cross_entropy(labels, output_layer)
    if EVAL:
        pred = tf.nn.sigmoid(output_layer)
        accuracy = tf.metrics.accuracy(output_layer, pred)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={'acc':accuracy})
    if TRAIN:
        global_step = tf.train.get_global_step()
        train_op = tf.train.AdadeltaOptimizer(learning_rate=0.1).minimize(loss, global_step)
        return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss=loss)


model_dir = "E:/Ai_projects_data/atom_data/data_out"
cnn_est = tf.estimator.Estimator(model_fn, model_dir=model_dir)
cnn_est.train(train_input_fn)
cnn_est.evaluate(eval_input_fn)


