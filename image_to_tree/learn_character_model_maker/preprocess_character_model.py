import variables
import os
import numpy as np
from PIL import Image, ImageOps

import re
"""
사진 폴더로부터 문자이미지를 numpy 배열로 변경하고 라벨을 붙힌 후 저장한다.

"""


input_image_size = (variables.LENGTH_OF['RESIZED_IMAGE_HORIZONTAL'], variables.LENGTH_OF['RESIZED_IMAGE_VERTICAL'], 1)
images = os.listdir(variables.CHARACTER_DATA_PATH)
dataX = np.zeros((len(images), variables.LENGTH_OF['RESIZED_IMAGE_VERTICAL'], variables.LENGTH_OF['RESIZED_IMAGE_HORIZONTAL']))
dataY = np.zeros((len(images), variables.NUMBER_OF['TOTAL']))
for data_num in range(len(images)):
    try:
        if data_num%100 ==0:
            print("data_num",data_num)
        image_path = variables.CHARACTER_DATA_PATH +"/"+ images[data_num]
        image = Image.open(image_path)
        #image = image.resize((25,25))
        image.convert('1')
        image = ImageOps.invert(image)
        pixel_img = np.array(image)
        dataX[data_num] = pixel_img[:, :, 0]
        numbers = re.findall('\d+',images[data_num])
        char_num = int(numbers[1])
        if char_num in variables.BIG_SAME_SMALL:
            char_num = char_num - 26
        elif char_num in variables.ZERO_YOUNG_IEUNG:
            char_num = 21
        elif char_num in variables.l_SAME:
            char_num = 9
        print(char_num)
        dataY[data_num][char_num] = 1
    except IOError:
        print("resizing error")
print("dataX.shape : ", dataX.shape)
dataX = dataX.reshape((len(dataX), 25*25))
print("dataX.shape : ", dataX.shape)
print("dataY.shape : ", dataY.shape)

print("dataX[100] : ", dataX[100])
print("dataY[100] : ", dataY[100])

np.save("./learn_character_model_data/data_X", dataX)
np.save("./learn_character_model_data/data_Y", dataY)
