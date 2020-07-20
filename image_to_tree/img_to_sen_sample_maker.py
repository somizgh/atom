from square_detection import *
import random
import cv2
import os

CLOSING_KERNEL_WIDTH = 5
CLOSING_KERNEL_HEIGHT = 1
IMAGE_IN_PATH = "./image_data_original"
IMAGE_OUT_PATH = "./image_data_chopped"


def sentence_sample_maker(image, IMG_CODE):
    image_height, image_width, ch = image.shape
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inv_gray_image = cv2.bitwise_not(gray_image)
    kernel = np.ones((2, 2), np.uint8)
    half = cv2.resize(image,dsize=(0,0),fx=0.5,fy=0.5)

    erosion = cv2.erode(inv_gray_image, kernel)
    #cv2.imshow("erosion", erosion)
    #cv2.waitKey(0)
    dilation = cv2.dilate(inv_gray_image, kernel)
    #cv2.imshow("dilation", dilation)
    #cv2.waitKey(0)
    morph_gradient = dilation - erosion
    #cv2.imshow("morph gradient", morph_gradient)
    #cv2.waitKey(0)

    morph_adthresh = cv2.adaptiveThreshold(morph_gradient, 125, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3,12)
    #cv2.imshow("morph adthresh", morph_adthresh)
    #cv2.waitKey(0)

    _, extreme_image = cv2.threshold(morph_adthresh,124,255,cv2.THRESH_BINARY)
    #cv2.imshow("extreme", extreme_image)
    #cv2.waitKey(0)

    straights, _ = find_straight(extreme_image, minStraightLength=40, div=1)
    no_straight = draw(extreme_image,straights,"straights",(0,0,0),1,show=False)

    colorful = cv2.cvtColor(morph_adthresh, cv2.COLOR_GRAY2RGB)
    """
    lines = cv2.HoughLinesP(morph_adthresh, rho=1, theta=np.pi / 180, threshold=80, minLineLength=100, maxLineGap=5)
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(colorful, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.imshow("line", colorful)
    cv2.waitKey(0)
    """

    kernel = np.ones((CLOSING_KERNEL_HEIGHT, CLOSING_KERNEL_WIDTH), np.uint8)
    morph_dilation = cv2.dilate(no_straight,kernel)
    #cv2.imshow("dilation", morph_dilation)
    #cv2.waitKey(0)

    morph_erosion = cv2.erode(morph_dilation, kernel)
    #cv2.imshow("erosion", morph_erosion)
    #cv2.waitKey(0)

    morph_close = morph_erosion
    #cv2.imshow("morph close", morph_close)




    approx_on = False
    convex_on = False
    rectangle_on = True
    contours, hierarchy = cv2.findContours(morph_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    num=0
    for i in range(len(contours)):
        # 각 Contour Line을 구분하기 위해서 Color Random생성
        b = random.randrange(1, 255)
        g = random.randrange(1, 255)
        r = random.randrange(1, 255)
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area > 100:
            num=num+1
            if approx_on:
                epsilon = 0.005*cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                img = cv2.drawContours(image, [approx], -1, (b, g, r), 2)
            elif convex_on:
                hull = cv2.convexHull(cnt)
                img = cv2.drawContours(image, [hull], -1, (b, g, r), 2)

            elif rectangle_on:
                x,y,w,h = cv2.boundingRect(cnt)
                copy1 = no_straight.copy()
                copy2 = copy1[y:y+h, x:x+w]
                cv2.imshow("im", copy2)
                TF=cv2.waitKey(0)
                if TF is 48:
                    TF = 0
                elif TF is 49:
                    TF = 1
                else:
                    print("errr")
                cv2.destroyAllWindows()
                cv2.imwrite(IMAGE_OUT_PATH+"/{0:>03d}{1:>03d}".format(IMG_CODE, num)+"L"+str(TF)+".jpg", copy2)
                cv2.rectangle(image, (x, y), (x+w, y+h),(b, g, r), 2)

            else:
                img = cv2.drawContours(image, [cnt], -1, (b, g, r), 2)

    titles = ['Result']
    images = [image]

    for i in range(1):
        plt.subplot(1, 1, i + 1), plt.title(titles[i]), plt.imshow(images[i])
        plt.xticks([]), plt.yticks([])

    plt.show()

    return 0


def draw(image, square_list, square_type, color, size,color_fix=False, show=True, gray=False, div=1):
    print(image.shape,"##########################")
    if square_type is'straights':

        for i in range(len(square_list)):
            print("i",i,square_list[i])
            cv2.line(image, (square_list[i][0],square_list[i][1]), (square_list[i][2],square_list[i][3]), color, size)
    elif square_type is 'points':
        for i in range(len(square_list)):
             cv2.line(image, (square_list[i][0],square_list[i][1]), (square_list[i][0],square_list[i][1]), color, size)
    elif square_type is 'squares':
        for i in range(len(square_list)):
            sx,sy,ex,ey,s,c = square_list[i]
            if div != 1:
                sx = int(sx/div)
                sy = int(sy/div)
                ex = int(ex/div)
                ey = int(ey/div)
            cv2.line(image, (sx, sy), (sx, ey), color, size)
            cv2.line(image, (sx, sy), (ex, sy), color, size)
            cv2.line(image, (sx, ey), (ex, ey), color, size)
            cv2.line(image, (ex, sy), (ex, ey), color, size)

    elif square_type is 'tree':
        all_squares = square_list.descendants
        sorted_all = sorted(all_squares, key=lambda a: a.position[4])
        for i in sorted_all:
            sx, sy, ex, ey, ssize = i.position
            if div != 1:
                sx = int(sx/div)
                sy = int(sy/div)
                ex = int(ex/div)
                ey = int(ey/div)
            code = i.type
            if code is 'square':
                colors=(255, 0, 0)  # red
            elif code is 'image':
                colors = (0, 0, 255)  # blue
            elif code is 'element':
                colors = (255, 255, 0)  # yellow
            elif code is 'table':
                colors = (0, 255, 0)  # green
            if color_fix:
                colors = color
            cv2.line(image, (sx, sy), (sx, ey), colors, size)
            cv2.line(image, (sx, sy), (ex, sy), colors, size)
            cv2.line(image, (sx, ey), (ex, ey), colors, size)
            cv2.line(image, (ex, sy), (ex, ey), colors, size)
    if show:
        if gray:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)
        plt.show()
    return image




for file in os.listdir(IMAGE_IN_PATH):
    code = int(file[:3])
    image = cv2.imread(IMAGE_IN_PATH+"/"+file)
    sentence_sample_maker(image, code)