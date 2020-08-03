from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import variables
import detect_character_model_variables as DCMV
from detect_character_model_maker import load_numpy
from detect_character_model_variables import *
from dictionary.character_vocabulary import CharVocabulary
import os


def detect_character(image, char_dict):
    pixel_img = np.array(image)
    model = load_model(VARIABLES.DCM_LATEST_H5_PATH)
    predict = model.predict(dataX)
    choose = predict.index(max(predict))
    return predict

if __name__ == "__main__":
    char_dictionary = CharVocabulary().load_char_vocabulary().convert_detectable_char_vocabulary()
    input_data_x, input_data_y = load_numpy(CHARACTER_NUMPY_DIR_PATH, 130)
    numbers = os.listdir(CHARACTER_IMAGES_DIR_PATH)
    for i in range(len(input_data_x)):
        ima = numbers
        c = detect_character(input_data_x, char_dictionary)






