import cv2
import variables as VARIABLES

import os
import numpy as np
import matplotlib.pyplot as plt
def pick_pixel(img,row,col):


    return 0

if __name__ =="__main__":
    images = os.listdir(VARIABLES.FULL_NATURE_PATH)
    for image_num in range(len(images)):
        image = cv2.imread(os.path.join(VARIABLES.FULL_NATURE_PATH, images[image_num]))
        cv2.imshow("adsf",image)
        cv2.waitKey(0)
        h, w, c = image.shape
        print(h,w,c)
        new_image = np.zeros((h,w,c),np.uint8)
        print(image[0, 0])
        print(image[100,100])
        for row in range(h):
            for col in range(w):
                r,g,b = image[row, col]
                print(row,col,r,g,b)
                print(type(r))
                sum = int(r)+int(g)+int(b)
                print("r",r)
                print(r+g)
                print("sum",sum)
                if sum==0:
                    r=0
                    g=0
                    b=0
                else:
                    r = int(255*r/sum)
                    g = int(255*g/sum)
                    b = int(255*b/sum)
                print(row,col,r,g,b)
                image[row, col] = [r, g, b]
        rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        plt.imshow(rgb)
        plt.show()
        print(image[0, 0])
        print(image[100, 100])
