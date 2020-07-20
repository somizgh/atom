import numpy as np
import copy
from PIL import Image
def ItoT(image):  #image to Text
    image = Image.open('./data/web_page_test.jpg')
    pix = np.array(image)
    print(pix.shape)

    return 0

ItoT(1)