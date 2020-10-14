import os
import cv2
import numpy as np
import variables as VARIABLES
import random
import copy
from separate_sentence_straight_model_variables import *
from useful_function import display_n_m_images
import matplotlib.pyplot as plt
"""
A file to create images for model training that separates straights and sentences 

"""
def find_child_outline(con,h,i,outline_list):
    outline_list.append(con[i])
    child_idx = h[0][i][2]
    while child_idx != -1:
        grand_child_idx = h[0][child_idx][2]
        while grand_child_idx != -1:
            print("in")
            outline_list = find_child_outline(con, h, grand_child_idx, outline_list)
            grand_child_idx = h[0][grand_child_idx][0]
        child_idx = h[0][child_idx][0]
    return outline_list

def find_outline(contour, h1):
    outline = []
    for i in range(len(h1[0])):
        if h1[0][i][3]==-1:
            outline = find_child_outline(contour,h1,i,outline)
    return outline

def save_image(image, dir_path, key, imc):
    print(key)
    file_name = "{0:>03}".format(imc)+"c"+str(key-48)+".jpg"
    print(os.path.join(dir_path,file_name))
    cv2.imwrite( os.path.join(dir_path,file_name), image)
    return 0


def make_ss_image(original_image_path, number_of_image_use_option):
    images = os.listdir(original_image_path)
    for image_num in range(len(images)):
        image = cv2.imread(os.path.join(original_image_path, images[image_num]))

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((2, 2), np.uint8)
        erosion = cv2.erode(gray_image, kernel)
        dilation = cv2.dilate(gray_image, kernel)
        morph_gradient = dilation - erosion  # 외곽선 추출


        inv_morph_gradient = cv2.bitwise_not(morph_gradient)
        morph_adthresh = cv2.adaptiveThreshold(inv_morph_gradient, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3) # 이걸로 구별하자
        _, morph_thresh = cv2.threshold(inv_morph_gradient, 127, 255, cv2.THRESH_BINARY_INV)

        cv2.imshow("morph thresh", morph_thresh)
        cv2.waitKey(0)

        kernel = np.ones((3, 3), np.uint8)
        morph_close = cv2.morphologyEx(morph_adthresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        morph_thr_close = cv2.morphologyEx(morph_thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        #morph_close = morph_adthresh

        display_n_m_images(2, 2, [morph_thr_close, dilation, morph_gradient, morph_adthresh],
                           ["morph_thr_close", "dilation", "morph_gradient", "morph adthresh"])
        cv2.imshow("morph close", morph_close)
        cv2.waitKey(0)

        approx_on = False
        convex_on = False
        rectangle_on = False
        iamge, contours, hierachy = cv2.findContours(morph_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _, contours2, hierachy2 = cv2.findContours(morph_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        outline = find_outline(contours,hierachy)

        print("len",len(outline))

        contours = outline

        show_image = copy.deepcopy(image)
        result = []
        AREACUT = False
        for i in range(len(contours)):
            # 각 Contour Line을 구분하기 위해서 Color Random생성
            b = 0
            g = 0
            r = 255
            cnt = contours[i]
            area = cv2.contourArea(cnt)

            if AREACUT is True:
                if area > 30:
                    if approx_on:
                        epsilon = 0.005 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        img = cv2.drawContours(image, [approx], -1, (b, g, r), 2)
                    elif convex_on:
                        hull = cv2.convexHull(cnt)
                        img = cv2.drawContours(image, [hull], -1, (b, g, r), 2)
                    elif rectangle_on:
                        x, y, w, h = cv2.boundingRect(cnt)
                        """
                        copy1 = no_straight.copy()
                        copy2 = copy1[y:y + h, x:x + w]
                        cv2.imshow(copy2)
                        TF = input()
                        cv2.waitKey(0)
                        if TF is not (1 and 0):
                            print("errrrrr")
                        cv2.imwrite("./image_to_sen_out/" + IMG_CODE + "L" + TF + ".jpg", copy2)
                        """
                        cv2.rectangle(image, (x, y), (x + w, y + h), (b, g, r), 2)
                        result.append([x, y, x + w, y + h, w * h])
                    else:
                        show_image = cv2.drawContours(image, [cnt], 1, (b, g, r), 1)
            else:
                show_image = cv2.drawContours(image, [cnt], 0, (b, g, r), 1)
        pltrgb = cv2.cvtColor(show_image,cv2.COLOR_BGR2RGB)
        plt.imshow(pltrgb)
        plt.show()
        """
        result = sorted(result, key=lambda a: -a[4])
        print(result)
        high_density = []
        for i in range(len(result)):
            sx,sy,ex,ey,size = result[i]
            density = np.average(morph_close[sy:ey,sx:ex])
            if density > 120:
                append = True
                for j in range(len(high_density)):
                    hsx, hsy, hex, hey, s,a = high_density[j]
                    if sx >= hsx and sy >= hsy and ex <= hex and ey <= hey:
                        append=False
                        break
                if append:
                    result[i].append(1)
                    print("append ",result[i])
                    high_density.append(result[i])
        print(high_density)
        new_dir_path = os.path.join(SS_GRADIENT_IMAGE_DIR_PATH, images[image_num][:-4])
        os.makedirs(new_dir_path)
        image_counter = 0
        for i in range(len(high_density)):
            sx,sy,ex,ey,s,c = high_density[i]
            ims = morph_gradient[sy:ey, sx:ex]
            cv2.imshow("image",ims)
            key_input = cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(ims, new_dir_path, key_input, image_counter)
            image_counter += 1
        #draw(show_image_c, high_density,"squares",(255,0,0),1)
    """
    return 0


if __name__ == "__main__":

    sample_dir_path = VARIABLES.FULL_CHROME_PATH

    make_ss_image(sample_dir_path, "max")
