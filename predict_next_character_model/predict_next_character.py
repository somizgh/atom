from dictionary.character_vocabulary import CharVocabulary
from file_preprocess import txtfile2npyfile, del_multiple_newline
import jamotools
import tensorflow as tf
from variables import *
from useful_function import *
import json
from predict_next_character_model_maker import PNCM
from dictionary.character_vocabulary import CharVocabulary


def PNC(model, start_string, vocabulary, num_generate):
    input_eval = [vocabulary.char2index(s) for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated_jamo = []
    temperature = 1.0
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated_jamo.append(vocabulary.index2char(predicted_id))
    text_generated = jamotools.join_jamos(text_generated_jamo)
    return jamotools.join_jamos(start_string) + text_generated


def load_model():
    vocabulary = CharVocabulary()
    vocabulary.load_char_vocabulary()
    model = PNCM(batch_size=1, char_vocabulary=vocabulary)
    with open(PNCM_HISTORY_JSON_PATH, "r") as hist:
        history_dict = json.load(hist)

    min_loss = min(history_dict["loss"])
    index = history_dict["loss"].index(min_loss)

    model.load_weights(PNCM_LATEST_MODEL_PATH + "/ckpt_" + str(index + 1))
    model.build(tf.TensorShape([1, None]))
    model.summary()
    return model


if __name__ == "__main__":
    vocabulary = CharVocabulary()
    vocabulary.load_char_vocabulary()
    model = load_model()
    print(PNC(model, start_string=u"ㅈㅏㅁㅇㅡㄴ ㅇㅏㄴㅇㅗㄱㅗ ", vocabulary=vocabulary, num_generate=1000))
