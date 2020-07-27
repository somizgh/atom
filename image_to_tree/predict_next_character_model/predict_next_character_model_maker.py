from dictionary.character_vocabulary import CharVocabulary
from file_preprocess import txtfile2npyfile, del_multiple_newline
import jamotools
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from variables import *
from useful_function import *
import predict_next_character_model_variables as PNCMV
import json


def PNCM(batch_size, char_vocabulary):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(char_vocabulary.len_dictionary(), PNCMV.EMBEDDING_DIM, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(PNCMV.RNN_UNITS,
                            return_sequences=True,
                            recurrent_initializer='glorot_uniform',
                            stateful=True),
        tf.keras.layers.Dense(char_vocabulary.len_dictionary())
    ])
    return model


def PNCM_loss_function(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def make_learn_model(text_as_int, vocabulary):
    """
    model 제작의 변수는 predict_next_character_model_variables 에 저장
    일련의 character 를 받아 다음 나올 character 을 예측하는 GRU 모델
    :param text_as_int: 분리시킨 자모를 char_vocabulary 에 맞는 인덱스로 변화시킨 numpy.array
    :param vocabulary: char_vocabulary
    :return: 제작한 모델의 history
    """
    examples_per_epoch = len(text_as_int) // PNCMV.SEQ_LENGTH
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(PNCMV.SEQ_LENGTH+1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)
    steps_per_epoch = examples_per_epoch // PNCMV.BATCH_SIZE

    dataset = dataset.shuffle(PNCMV.BUFFER_SIZE).batch(PNCMV.BATCH_SIZE, drop_remainder=True)

    model = PNCM(PNCMV.BATCH_SIZE, vocabulary)
    model.summary()

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=PNCM_loss_function)

    model_dir = PNCM_MODEL_PREFIX + "/" + time_format()
    create_folder(model_dir)

    checkpoint_prefix = os.path.join(model_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True,
        period=1)

    cb_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    history = model.fit(dataset.repeat(), epochs=PNCMV.EPOCHS, steps_per_epoch=steps_per_epoch,
                        callbacks=[checkpoint_callback, cb_early_stopping], verbose=2)


    history_dict = history.history
    json.dump(history_dict, open(os.path.join(model_dir, PNCM_HISTORY_JSON_NAME), 'w'))
    model.save(os.path.join(model_dir, PNCM_H5_NAME))
    json.dump(model.to_json(), open(os.path.join(model_dir,"model.json"), 'w'))

    plt.plot(history.history['loss'])
    plt.show()
    tf.train.latest_checkpoint(model_dir)
    return history





#삭제 예정
def load_volume(PATH, vocabulary):
    file = open(PATH, "r", encoding='UTF-8')
    text = file.read()
    split = jamotools.split_syllables(text)
    indexes = []
    for i in split:
        c = vocabulary.char2index(i)
        if c == -1:
            continue
        else:
            indexes.append(c)
    return np.array(indexes)


def load_indexes_fromnpy(path):
    index_numpy = np.array([], dtype=np.int32)
    counter = 0
    for root, dirs, files in os.walk(path):
        root_path = os.path.join(os.path.abspath(path), root)
        for file in files:
            file_path = os.path.join(root_path, file)
            temp = np.load(file_path)
            index_numpy = np.concatenate((index_numpy, temp), axis=0)
            counter = counter+1
            if counter > PNCMV.NUMBER_OF_FILE_READ:
                return index_numpy
            print("concate file", counter)
            print("data size", np.shape(index_numpy))
    return index_numpy


def make_index_npy(txt_file_path, npy_file_path, char_dict):
    del_multiple_newline(txt_file_path)
    txtfile2npyfile(txt_file_path, npy_file_path, char_dict)


if __name__ == "__main__":
    char_vocabulary = CharVocabulary()
    print("c")
    char_vocabulary.save_char_vocabulary()
    char_vocabulary.load_char_vocabulary()
    index_array = load_indexes_fromnpy(WIKI_NPY_PATH)
    his = make_learn_model(index_array, char_vocabulary)

