import cv2
import numpy as np
import matplotlib.pyplot as plt
h=255
w=255
c=3
black = np.zeros((h,w,c),np.uint8)

for i in range(h):
    for j in range(w):
        r=1
        g=0.2
        b=0.6
        black[i,j] = [r*j,g*j,b*j]
plt.imshow(black)
plt.show()