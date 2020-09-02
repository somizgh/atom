from square_detection import *
from square_tree import *
from anytree import Node
from pytesseract import *
import time
import random
from variables import *
import os
CLOSING_KERNEL_WIDTH = 5
CLOSING_KERNEL_HEIGHT = 1
import sys
sys.setrecursionlimit(10000)


def image_making_sentence(image, IMG_CODE):
    image_height, image_width, ch = image.shape
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inv_gray_image = cv2.bitwise_not(gray_image)
    kernel = np.ones((2, 2), np.uint8)
    half = cv2.resize(image,dsize=(0,0),fx=0.5,fy=0.5)

    erosion = cv2.erode(inv_gray_image, kernel)
    dilation = cv2.dilate(inv_gray_image, kernel)

    morph_gradient = dilation - erosion
    cv2.imshow("morph gradient", morph_gradient)
    cv2.waitKey(0)

    morph_adthresh = cv2.adaptiveThreshold(morph_gradient, 125, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3,12)
    cv2.imshow("morph adthresh", morph_adthresh)
    cv2.waitKey(0)

    _, extreme_image = cv2.threshold(morph_adthresh,124,255,cv2.THRESH_BINARY)
    cv2.imshow("extreme", extreme_image)
    cv2.waitKey(0)

    straights, _ = find_straight(extreme_image, minStraightLength=40, div=1)
    no_straight = draw(extreme_image,straights,"straights",(0,0,0),1)
    cv2.imshow("no straight", no_straight)
    cv2.waitKey(0)

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
    cv2.imshow("dilation", morph_dilation)
    cv2.waitKey(0)

    morph_erosion = cv2.erode(morph_dilation, kernel)
    cv2.imshow("erosion", morph_erosion)
    cv2.waitKey(0)

    morph_close = morph_erosion
    cv2.imshow("morph close", morph_close)

    approx_on = False
    convex_on = False
    rectangle_on = True
    contours, hierarchy = cv2.findContours(morph_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        # 각 Contour Line을 구분하기 위해서 Color Random생성
        b = random.randrange(1, 255)
        g = random.randrange(1, 255)
        r = random.randrange(1, 255)
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area > 100:
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
                copy2 = copy1[y:y+h,x:x+w]
                cv2.imshow(copy2)
                TF = input()
                cv2.waitKey(0)
                if TF is not(1 and 0):
                    print("errrrrr")
                cv2.imwrite("./image_to_sen_out/"+IMG_CODE+"L"+TF+".jpg",copy2)
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


def from_image_find_sentence(original_image, image, real_straights):
    cv2.imshow("extreme", image)
    cv2.waitKey(0)
    morph_adthresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                           3, 12)
    cv2.imshow("find sen morph adthresh", morph_adthresh)
    cv2.waitKey(0)

    #no_straight = draw(morph_adthresh, real_straights, "straights", (0, 0, 0), 3,show=False)
    #cv2.imshow("no straight", no_straight)
    #cv2.waitKey(0)

    kernel = np.ones((10,3), np.uint8)
    morph_close = cv2.morphologyEx(morph_adthresh, cv2.MORPH_CLOSE,kernel)
    cv2.imshow("morph close", morph_close)
    cv2.waitKey(0)

    approx_on = False
    convex_on = False
    rectangle_on = True
    iamge, contours, hierachy = cv2.findContours(morph_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    result = []
    for i in range(len(contours)):
        # 각 Contour Line을 구분하기 위해서 Color Random생성
        b = random.randrange(1, 255)
        g = random.randrange(1, 255)
        r = random.randrange(1, 255)
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area > 20:
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
                cv2.rectangle(original_image, (x, y), (x + w, y + h), (b, g, r), 2)
                result.append([x,y,x+w,y+h,w*h])
            else:
                img = cv2.drawContours(image, [cnt], -1, (b, g, r), 1)
    cv2.imshow("end", original_image)
    cv2.waitKey(0)

    return result


def webpage_making_imageTree(image):
    image_height, image_width, ch = image.shape
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_RGB2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inv_gray_image = cv2.bitwise_not(gray_image)

    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(inv_gray_image, kernel)
    dilation = cv2.dilate(inv_gray_image, kernel)
    morph_gradient = dilation - erosion
    cv2.imshow("morph gradient", morph_gradient)
    cv2.waitKey(0)

    ret, extreme = cv2.threshold(morph_gradient, 5, 255, cv2.THRESH_BINARY)
    cv2.imshow("extreme", extreme)
    cv2.waitKey(0)
    """
    morph_close = cv2.morphologyEx(extreme, cv2.MORPH_CLOSE, (9, 5))
    cv2.imshow("morph close", morph_close)
    cv2.waitKey(0)
    """

    # 직선을 찾는다.
    t1 = time.time()
    straights, preprocessed_image = find_straight(extreme, minStraightLength=20, div=1)
    t2 = time.time()
    print("{:<60}".format("find_straight end "), "time : {:>10}".format(round(t2-t1, 4)))
    #draw(original_RGB, straights,"straights",(255,0,0),3)
    # 변이 될 직선과 문장이 될 직선을 구별

    # 문장과 변 이미지 맵 제작
    points, straights_image_map = find_points(straights, image_height, image_width)  # stretch 전 이미지 맵 제작 역활
    # points, sentence_image_map = find_points(sentence, image_height, image_width)  # stretch 전 이미지 맵 제작 역활

    # 변의 길이를 늘린다.
    stretched_straights, straights_image_map = stretch_straights(straights, straights_image_map, robust_pixel=6)

    # 2번의 pruning으로 가로,세로 누가먼저하든지 상관 x
    pruned_straights = pruning_straights(stretched_straights, straights_image_map, mingap=6)
    points, straights_image_map = find_points(pruned_straights, image_height, image_width)
    pruned_straights = pruning_straights(pruned_straights, straights_image_map, mingap=6)
    points, straights_image_map = find_points(pruned_straights, image_height, image_width)
    #draw(original_RGB, pruned_straights, "straights", (255,0,0),1)

    real_straights, sentence = separate_sentence_from_straights(extreme, pruned_straights, width=5)
    #draw(original_RGB2, sentence, "straights", (0,255,0),1,show=False)
    #draw(extreme, real_straights,"straights",(0,0,255),1)
    sentence = from_image_find_sentence(original_RGB, morph_gradient, real_straights)  # 이미지로부터 문자열 추출
    #t3 = time.time()
    #print("{:<60}".format("separate_sentence_from_straights end "), "time : {:>10}".format(round(t3-t2, 4)))

    # outline 을 real_straights 에 추가
    real_straights = add_outline(real_straights, image_height, image_width, len(straights))


    #draw(dst, real_straights, "straights", (255, 0, 0), 2, show=False)
    #draw(dst, sentence, "straights", (0, 255, 0), 2, show=True)



    #pruned_sentence = pruning_straights(sentence, sentence_image_map, mingap=8, sentence=True)
    #_, sentence_image_map = find_points(pruned_sentence, image_height, image_width)
    #pruned_sentence = pruning_straights(pruned_sentence, sentence_image_map, mingap=8, sentence=True)
    #_, sentence_image_map = find_points(pruned_sentence, image_height, image_width)


    #draw(dst2, pruned_straights, "straights", (0, 0, 255), 3, show=False)
    #draw(dst2, points, "points", (0, 255, 255), 6, show=True)
    #draw(dst, pruned_sentence, "straights", (0, 255, 0), 3, show=True)

    # 변과 점으로 부터 사각형을 찾는다.
    squares = find_squares(straights_image_map, pruned_straights, points)
    element_map = make_element_map(original_RGB, squares)
    sorted_squares = sorted(squares, key=lambda a: -a[4])

    draw(original_RGB, pruned_straights, "straights", (0, 0, 255), 3,show=False)
    draw(original_RGB, sorted_squares,"squares",(255,0,0),3)

    #sentence_square = sentence_to_square(image, pruned_sentence, sameway_padding=8, padding=16)
    #draw(dst2, sentence_square, "squares", (255, 0, 255), 3, show=True)


    # tree 제작 시작
    tree_root = make_tree(sorted_squares, element_map)
    square_erased_image = draw(preprocessed_image, tree_root, "tree", (0,0,0), 4 ,color_fix=True, show=False, div=3)

    plt.imshow(square_erased_image,cmap="gray")
    plt.show()
    components = divide_pixel_by_pixel(square_erased_image)
    draw(original_RGB, components, "squares", (0,255,0),1)
    #tree_root = add_sentence(tree_root, sentence_square)
    draw_tree(tree_root)

    tree_root = add_component(tree_root,components)

    tree_root = add_image(tree_root, image)
    tree_root = find_string(tree_root)
    return tree_root


def resize_and_find(image, root):
    image_height, image_width, ch = image.shape
    for i in range(10):
        resize_magnification = pow(2,i)
        resized_height = int(image_height/resize_magnification)
        resized_width = int(image_width/resize_magnification)
        if resized_width*resized_height < 2000:
            break
        print(resized_height,resized_width)
        resized_image = cv2.resize(image,dsize=(resized_width,resized_height), interpolation=cv2.INTER_AREA)
    return 0


def draw(image, square_list, square_type, color, size,color_fix=False, show=True, gray=False, div=1):
    if square_type is'straights':

        for i in range(len(square_list)):
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


if __name__ == "__main__":
    pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    a = 1
    source_image = cv2.imread(os.path.join(FULL_IMAGES_PATH,"{0:>03}.jpg".format(a)))

    tree = webpage_making_imageTree(source_image)
    #tree = image_making_sentence(source_image)
    tree = resize_and_find(source_image, tree)

