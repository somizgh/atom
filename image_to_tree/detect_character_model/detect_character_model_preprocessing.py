import variables as VARIABLES
import os
import numpy as np
from PIL import Image, ImageOps
from detect_character_model_variables import *
from character_vocabulary import CharVocabulary
import re
import math
"""
사진 폴더로부터 문자이미지를 numpy 배열로 변경하고 라벨을 붙힌 후 저장한다.

"""


def character_images_to_numpy(number_of_image_use_option, image_per_file):
    char_vocabulary = CharVocabulary().load_char_vocabulary().convert_detectable_char_vocabulary()
    keys = char_vocabulary.return_keys()
    print(len(keys))
    #input_image_size = (IMAGE_WIDTH, IMAGE_HEIGHT, 1)
    image_files = os.listdir(CHARACTER_IMAGES_DIR_PATH)
    number_of_image_use = number_of_image_use_option
    if number_of_image_use_option is "max":
        number_of_image_use = len(image_files)
    elif number_of_image_use_option > len(image_files):
        print("number of image use you wrote {0} larger than {1}".format(number_of_image_use_option,len(image_files)))
        number_of_image_use = len(image_files)
    else:
        number_of_image_use = number_of_image_use_option

    file_used = 0
    numpy_size = 0
    data_x = []
    data_y = []
    for data_num in range(number_of_image_use):
        if data_num % image_per_file == 0:
            file_used = 0
            numpy_size = image_per_file
            if number_of_image_use < data_num + image_per_file:
                numpy_size = number_of_image_use - data_num
            data_x = np.zeros((numpy_size, IMAGE_HEIGHT, IMAGE_WIDTH))
            data_y = np.zeros((numpy_size, len(keys)))
        image_path = os.path.join(CHARACTER_IMAGES_DIR_PATH, image_files[data_num])
        image = Image.open(image_path)
        image.convert('1')
        image = ImageOps.invert(image)
        pixel_img = np.array(image)
        data_x[file_used] = pixel_img[:, :, 0]
        numbers = re.findall('\d+', image_files[data_num])
        char_num = int(numbers[1])
        data_y[file_used][char_num] = 1
        file_used += 1
        if file_used == numpy_size:
            print("saved {} images".format(data_num+1))
            save_numpy(data_x, data_y, CHARACTER_NUMPY_DIR_PATH, math.ceil((data_num+1)/image_per_file))
    return data_x, data_y


def save_numpy(input_data_x, input_data_y, path, counter):
    print(counter)
    data_x_name = INPUT_DATA_X_NAME + INPUT_DATA_COUNTER_FORMAT.format(counter)
    data_y_name = INPUT_DATA_Y_NAME + INPUT_DATA_COUNTER_FORMAT.format(counter)
    np.save(os.path.join(path, data_x_name), input_data_x)
    np.save(os.path.join(path, data_y_name), input_data_y)
    return 0

if __name__ == "__main__":
    input_dataX, input_dataY = character_images_to_numpy("max", IMAGES_PER_FILE)

