
import os
import cv2
import numpy as np
import variables as VARIABLES
import random
from image_to_tree_main import draw
from separate_sentence_straight_model_variables import *
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
        image_height, image_width, ch = image.shape
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((2, 2), np.uint8)
        erosion = cv2.erode(gray_image, kernel)
        dilation = cv2.dilate(gray_image, kernel)
        morph_gradient = dilation - erosion
        cv2.imshow("morph gradient", morph_gradient)
        cv2.waitKey(0)
        morph_adthresh = cv2.adaptiveThreshold(morph_gradient, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 12)
        cv2.imshow("adaptive thers", morph_adthresh)
        cv2.waitKey(0)

        kernel = np.ones((2, 4), np.uint8)
        morph_close = cv2.morphologyEx(morph_adthresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        cv2.imshow("morph close", morph_close)
        cv2.waitKey(0)
        show_image = cv2.cvtColor(morph_gradient, cv2.COLOR_GRAY2RGB)
        approx_on = False
        convex_on = False
        rectangle_on = True
        iamge, contours, hierachy = cv2.findContours(morph_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        show_image_c = cv2.cvtColor(morph_close, cv2.COLOR_GRAY2RGB)
        result = []
        for i in range(len(contours)):
            # 각 Contour Line을 구분하기 위해서 Color Random생성
            b = random.randrange(1, 255)
            g = random.randrange(1, 255)
            r = random.randrange(1, 255)
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if area > 60:
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
                    cv2.rectangle(show_image, (x, y), (x + w, y + h), (b, g, r), 2)
                    result.append([x, y, x + w, y + h, w * h])
                else:
                    img = cv2.drawContours(image, [cnt], -1, (b, g, r), 1)
        cv2.imshow("end", show_image)
        cv2.waitKey(0)
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
        draw(show_image_c, high_density,"squares",(255,0,0),1)

    return 0


if __name__ == "__main__":
    sample_dir_path = VARIABLES.FULL_CHROME_PATH_NAME

    make_ss_image(sample_dir_path, "max")
