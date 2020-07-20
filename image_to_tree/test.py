import numpy as np

import cv2

image = cv2.imread("original.jpg")
print(image.shape)
height, width, c = image.shape
y_len = 1600
i=0
while i-1 < int(height/y_len):
    im = image[i*y_len:(i+1)*y_len,0:width]
    cv2.imwrite(str(i)+".jpg",im)
    i=i+1