import numpy as np
import skimage.measure
import cv2
import matplotlib.pyplot as plt
import time
import sys


def canny_prepcocess(canny, division):
    """
    mean = cv2.blur(canny, (3, 3))
    _, max_pooled = cv2.threshold(mean, 50, 255, cv2.THRESH_BINARY)
    mean = cv2.blur(max_pooled, (3, 3))
    _, max_pooled = cv2.threshold(mean, 50, 255, cv2.THRESH_BINARY)

    """
    mean = skimage.measure.block_reduce(canny, (division, division), np.max)
    plt.imshow(mean, cmap="gray")
    plt.show()
    _, mean_pooled = cv2.threshold(mean, 50, 255, cv2.THRESH_BINARY)
    plt.imshow(mean_pooled, cmap="gray")
    plt.show()

    return mean_pooled

def find_straight(image, minStraightLength, div=3):
    """
    이미지에서 minStraightLength보다 긴 직선을 찾는다
    find straights that longer than minStraightLength in image
    :param image: 소스 이미지 canny 검출된 이미지 사용
    :param minStraightLength: 직선으로 인식하는 최소의 픽셀 수
    pooling을 하기에 poolingSize의 배수로 기본 50
    :param poolingSize: 직선을 찾기전 이미지를 풀링할 크기
    :return: 2차원 ndarray, [[직선의 x 시작점, 직선의 y 시작점, 직선의 x끝점, 직선의 y끝점, 직선 번호]*n]
    정렬돼지 않은 결과를 리턴

    image_height, image_width = image.shape
    max_pooled = skimage.measure.block_reduce(image, (poolingSize, poolingSize), np.max)
    pooled_image_height, pooled_image_width = max_pooled.shape
    """
    if div == 1:
        pooled_image_height, pooled_image_width = image.shape
        preprocessed_image = image
    else:
        preprocessed_image = canny_prepcocess(image, div)
        pooled_image_height, pooled_image_width = preprocessed_image.shape
    straights = []

    print(image)
    x=0
    while x < pooled_image_width:
        serial_num = 0
        y0 = 0
        y = 0
        instraight = False
        while y < pooled_image_height:
            if preprocessed_image[y, x] == 255:
                serial_num = serial_num + 1
            else:
                if instraight is True:
                    straights.append([x, y0, x, y-1])
                serial_num = 0
                instraight = False
            if serial_num == int(minStraightLength/div):
                y0 = y - int(minStraightLength/div)+1
                instraight = True
            y = y + 1
        if instraight is True:
            straights.append([x, y0, x, y - 1])
        x = x+1

    y = 0
    while y < pooled_image_height:
        serial_num = 0
        x0 = 0
        x = 0
        instraight = False
        while x < pooled_image_width:
            if preprocessed_image[y, x] == 255:
                serial_num = serial_num + 1
            else:
                if instraight is True:
                    straights.append([x0, y, x-1, y])
                serial_num = 0
                instraight = False
            if serial_num == int(minStraightLength/div):
                x0 = x - int(minStraightLength/div)+1
                instraight = True
            x = x + 1
        if instraight is True:
            straights.append([x0, y, x-1, y])
        y = y+1
    recovered_straights = np.array(straights) * div
    if len(recovered_straights) == 0:
        return [], preprocessed_image
    recovered_straights = np.concatenate((recovered_straights, [[i] for i in range(1, len(straights)+1)]), axis=1)
    return recovered_straights, preprocessed_image


def add_outline(real_straights, image_height, image_width, num):

    out_line = [[image_width - 1, 0, image_width - 1, image_height - 1, num+1],
                [0, 0, 0, image_height - 1, num + 2],
                [0, 0, image_width - 1, 0, num + 3],
                [0, image_height - 1, image_width - 1, image_height - 1, num + 4]]
    real_straights = np.concatenate((real_straights, out_line), axis=0)
    return real_straights

def stretch_straights(straights, image_map, robust_pixel=3):
    """
    직선의 양옆을 비율이나 지정된 픽셀만큼 늘려준다. 이미지를 틔어나가지 않도록 조정됨
    :param image: 이미지의 가로와 세로를 알기위해 필요
    :param straights: 직선의 2차원 리스트[[직선의 x 시작점, 직선의 y 시작점, 직선의 x끝점, 직선의 y끝점]*n]
    :param robust_pixel: 직선을 늘릴 픽셀 수 자연수 추천, 양 끝방향으로 늘어남 ex) 3픽셀이면 양끝 3픽셀 총 6
    :return: 늘어난 직선의 2차원 list, [[직선의 x 시작점, 직선의 y 시작점, 직선의 x끝점, 직선의 y끝점, 직선 번호]*n]
    """
    finish_straights = [[False, False] for _ in range(len(straights))]
    image_height = len(image_map)
    image_width = len(image_map[0])
    for i in range(robust_pixel):
        for j in range(len(straights)):
            sx, sy, ex, ey, num = straights[j]
            if sx == ex:  # vertical
                if sy-1 >= 0 and finish_straights[j][0] is False: # 선의 시작점
                    original = image_map[sy-1][sx]
                    vertical_code = int(original % 10000)
                    if vertical_code is 0:
                        image_map[sy - 1][sx] = original + num
                        straights[j][1] = sy-1
                    else:
                        finish_straights[j][0] = True
                else:
                    finish_straights[j][0] = True

                if ey+1 < image_height and finish_straights[j][1] is False: # 선의 끝점
                    original = image_map[ey + 1][sx]
                    vertical_code = int(original % 10000)
                    if vertical_code is 0:
                        image_map[ey + 1][sx] = original + num
                        straights[j][3] = ey + 1
                    else:
                        finish_straights[j][1] = True
                else:
                    finish_straights[j][1] = True

            elif sy == ey:  # horizontal
                if sx - 1 >= 0 and finish_straights[j][0] is False:  # 선의 시작점
                    original = image_map[sy][sx-1]
                    horizontal_code = int(original/10000)
                    if horizontal_code is 0:
                        image_map[sy][sx-1] = original + num*10000
                        straights[j][0] = sx - 1
                    else:
                        finish_straights[j][0] = True
                else:
                    finish_straights[j][0] = True

                if ex + 1 < image_width and finish_straights[j][1] is False:  # 선의 끝점
                    original = image_map[ey][ex+1]
                    horizontal_code = int(original/10000)
                    if horizontal_code is 0:
                        image_map[ey][ex+1] = original + num*10000
                        straights[j][2] = ex + 1
                    else:
                        finish_straights[j][1] = True
                else:
                    finish_straights[j][1] = True
            else:   # 수직도 아니고 수평도 아닌 에러
                print("error not a straight")

    return straights, image_map


def split_straight_vertical_horizontal(mixed_straights):
    """
    직선들중 수직선과 수평선을 분류한다.
    :param mixed_straights: 수직선과 수평선이 섞인 2차원 ndarray, [[직선의 x 시작점, 직선의 y 시작점, 직선의 x끝점, 직선의 y끝점, 직선 번호]*n]
    :return: 수직선만 있는 2차원 list vertical_straights, 수평선만 있는 2차원 horizontal_straights
    """
    vertical_straights = []
    horizontal_straights = []
    for i in range(len(mixed_straights)):
        sx, sy, ex, ey, num = mixed_straights[i]
        if sx == ex:
            vertical_straights.append([sx, sy, ex, ey, num])
        elif sy == ey:
            horizontal_straights.append([sx, sy, ex, ey, num])
    return vertical_straights, horizontal_straights


def pruning_straights(straights, image_map, mingap=5, sentence=False):
    #거리가 mingap이하이면 합쳐질 가능성 있다.
    vertical_straights, horizontal_straights = split_straight_vertical_horizontal(straights)
    sorted_vertical_straights = sorted(np.array(vertical_straights), key=lambda a: a[0])
    sorted_horizontal_straights = sorted(np.array(horizontal_straights), key=lambda a: a[1])

    sorted_vertical_straights = [list(i) for i in sorted_vertical_straights]
    sorted_horizontal_straights = [list(i) for i in sorted_horizontal_straights]

    #for i in range(len(sorted_vertical_straights)):
    i = 0
    while i < len(sorted_vertical_straights):
        sx, sy, ex, ey, num = sorted_vertical_straights[i]
        del_this_straight = False
        for j in range(len(sorted_vertical_straights)):
            if del_this_straight is True:
                break
            nsx, nsy, nex, ney, nnum = sorted_vertical_straights[j]
            if sx is nsx:
                continue
            if abs(sx-nsx) <= mingap:
                len_now = abs(ey-sy)
                len_next = abs(ney - nsy)
                if len_now <= len_next:
                    if ney >= ey >= nsy or ney >= sy >= nsy: # 겹치는 부분이 있다.
                        check = True
                        if sentence is False:
                            for k in range(len_now):
                                if int(image_map[sy+k][sx]/10000) != int(image_map[sy+k][nsx]/10000):
                                    check = False
                        if check is True:
                            del_this_straight = True
                            sorted_vertical_straights[j] = [nsx, min(sy, nsy), nex, max(ey, ney), nnum]
                            sorted_vertical_straights.remove([sx,sy,ex,ey,num])
                            i = i-1
            elif nsx-sx > mingap:
                break
        i = i+1
    #for i in range(len(sorted_horizontal_straights)):
    i = 0
    while i < len(sorted_horizontal_straights):
        sx, sy, ex, ey, num = sorted_horizontal_straights[i]
        del_this_straight = False
        for j in range(len(sorted_horizontal_straights)):
            if del_this_straight is True:
                break

            nsx, nsy, nex, ney, nnum = sorted_horizontal_straights[j]
            if sy is nsy:
                continue
            if abs(sy-nsy) <= mingap:
                len_now = abs(ex-sx)
                len_next = abs(nex - nsx)
                if len_now <= len_next:
                    if nex >= ex >= nsx or nex >= sx >= nsx : # 겹치는 부분이 있다.
                        check = True
                        if sentence is False:
                            for k in range(len_now):
                                if int(image_map[sy][sx+k]%10000) != int(image_map[nsy][sx+k]%10000):
                                    check = False
                        if check is True:
                            del_this_straight = True
                            sorted_horizontal_straights[j] = [min(sx, nsx), nsy, max(ex, nex), ney, nnum]
                            sorted_horizontal_straights.remove([sx,sy,ex,ey,num])
                            i=i-1
            elif nsy-sy > mingap:
                break
        i=i+1
    sorted_vertical_straights.extend(sorted_horizontal_straights)
    return sorted_vertical_straights

def flood_fill(numpy):


    return True

def separate_sentence_from_straights(canny, straights, width=3, too_long=0.5):
    imheight, imwidth = canny.shape
    thres_imheight = int(imheight*too_long)
    thres_imwidth = int(imwidth*too_long)
    real_straights= []
    sentence = []
    numpy_map = np.array(canny) # 흰색은 255
    for i in range(len(straights)):
        sx,sy,ex,ey,num = straights[i]
        if sx == ex: # 수직선
            if abs(ey-sy) > thres_imheight:
                real_straights.append([sx, sy, ex, ey, num])
                continue
            ssx = max(0, sx-width)
            esx = min(imwidth-1, sx+width)
            area = numpy_map[sy:ey+1, ssx:esx+1]
            mean_list = np.mean(area,axis=0)
            sub_list = [abs(mean_list[i]-mean_list[i+1]) for i in range(len(mean_list)-1)]
            if max(sub_list) >= 220:
                real_straights.append([sx,sy,ex,ey,num])
            else:
                sentence.append([sx,sy,ex,ey,num])

        else: #수평선
            if abs(ex-sx) > thres_imwidth:
                real_straights.append([sx, sy, ex, ey, num])
                continue
            ssy = max(0, sy - width)
            esy = min(imheight - 1, sy + width)
            area = numpy_map[ssy:esy + 1,sx:ex + 1]
            mean_list = np.mean(area, axis=1)
            sub_list = [abs(mean_list[i] - mean_list[i + 1]) for i in range(len(mean_list) - 1)]
            if max(sub_list) >= 220:
                real_straights.append([sx, sy, ex, ey, num])
            else:
                sentence.append([sx, sy, ex, ey, num])
    return real_straights, sentence


def find_points(straights, image_height, image_width):
    """
    점의 코드 : 수평선과 수직선을 순서에 따라 코드를 부여하고 만나는 점은 수평선*10000+수직선의 코드를 부여
    ex) 58번 수평선과 145번 수직선이 만나는 점의 코드 : 580145
    :param image: 이미지의 가로와 세로를 알기위해 필요
    :param straights: 점을 구할 직선 2차원 ndarray, [[직선의 x 시작점, 직선의 y 시작점, 직선의 x끝점, 직선의 y끝점, 직선 번호]*n]
    :return: 점의 2차원 list와 선이 지나가는 곳에 선의 번호가 기입된 map 출력,
    [[점의 x좌표, 점의 y좌표, 점의 코드]*n],
    image_map[이미지 y높이][이미지 x높이]
    """
    image_map = [[0 for _ in range(image_width)] for _ in range(image_height)]
    points = []
    # line 끼리 만나는 점 point 계산
    if len(straights) > 10000:
        print("too many lines")
        return 0, 0
    for i in range(len(straights)):
        x0, y0, x1, y1, num = straights[i]
        if x0 == x1:  # 수직선
            y_len = abs(y0-y1)+1
            ys = min(y0, y1)
            for j in range(y_len):
                if 0 <= ys+j <= image_height-1:
                    original = image_map[ys+j][x0]
                    horizontal_code= int(original/10000)
                    vertical_code = int(original % 10000)
                    if vertical_code is not 0:
                        continue
                    if horizontal_code > 0:
                        points.append([x0, ys+j, original+num])
                    image_map[ys + j][x0] = image_map[ys + j][x0] + num  # 수직 라인이 10000이하의 수
        elif y0 == y1:
            x_len = abs(x0-x1)+1
            xs = min(x0, x1)
            for j in range(x_len):
                if 0 <= xs+j <= image_width-1:
                    original = image_map[y0][xs+j]
                    horizontal_code = int(original / 10000)
                    vertical_code = int(original % 10000)
                    if horizontal_code is not 0:
                        continue
                    if vertical_code > 0:
                        points.append([xs+j, y0, original+num*10000])
                    image_map[y0][xs+j] = image_map[y0][xs+j] + num*10000  # 수평라인이 10000이상의 수
    sorted_points = sorted(points, key=lambda point: (point[0], point[1]))
    return sorted_points, image_map


def matching_line(x, y, image_map, horizontal_point,vertical_point, minsquaresize):
    candidate_square = []

    for i in range(len(horizontal_point)): # x좌표, 정수
        for j in range(len(vertical_point)):
            code = 'square'
            if image_map[vertical_point[j][0]][horizontal_point[i][0]] == (vertical_point[j][1])*10000+(horizontal_point[i][1]):
                abs_x = abs(x-horizontal_point[i][0])
                abs_y = abs(y-vertical_point[j][0])
                if abs_x < 10 or abs_y < 10:
                    continue
                size = abs_x*abs_y
                if i is 0 and j is 0:
                    code = 'element'
                if size > minsquaresize:
                    candidate_square.append([x, y, horizontal_point[i][0], vertical_point[j][0], size, code])
    '''
    if len(candidate_square) == 0:
        return 'None', [-1, -1, -1, -1, 0]
    #print(candidate_square)
    candidate_square = np.array(candidate_square)
    # 제일 큰 사각형 1개
    candidate_square = sorted(candidate_square, key=lambda can: -can[4])
    candidate_square = [list(a) for a in candidate_square]
    end = len(candidate_square)-1
    biggest = candidate_square[0]
    smallest = candidate_square[end]

    if biggest[4] < minsquaresize:
        return [-1, -1, -1, -1, 0, "too small"], [-1, -1, -1, -1, 0, "too small"]
    if len(candidate_square) == 1:
        return candidate_square[0].append("single square"), candidate_square[0].append("single square")
    if candidate_square[0][4]/len(candidate_square) > table_pro:
        return candidate_square[0].append("table"), candidate_square[end].append("")
    return 'Image', candidate_square[0]
    '''
    return candidate_square


def find_squares(image_map, straights, sorted_points, minSqurareSize=200):
    squares = []
    straights_num_index = [straights[i][4] for i in range(len(straights))]
    for i in range(len(sorted_points)):
        x, y, point_code = sorted_points[i]
        horizontal_line = int(point_code / 10000)  # 라인 번호
        vertical_line = int(point_code % 10000)
        hori_index = straights_num_index.index(horizontal_line)
        verti_index = straights_num_index.index(vertical_line)
        from_end_x = max(straights[hori_index][0], straights[hori_index][2])
        from_end_y = max(straights[verti_index][1], straights[verti_index][3])
        horizontal_cross_list = []
        for j in range(x+1, from_end_x+1):
            code = image_map[y][j]
            vertical_code = int(code % 10000)
            if vertical_code is not 0:
                horizontal_cross_list.append([j, vertical_code])
        vertical_cross_list = []
        for j in range(y+1, from_end_y+1):
            code = image_map[j][x]
            horizontal_code = int(code / 10000)
            if horizontal_code is not 0:
                vertical_cross_list.append([j, horizontal_code])
        cells = matching_line(x, y, image_map, horizontal_cross_list, vertical_cross_list, minSqurareSize)
        if len(cells) > 0:
            squares.extend(cells)
    return squares


def make_element_map(image, squares, ):
    image_height, image_width, c = image.shape
    element_map = [[0 for _ in range(image_width)] for _ in range(image_height)]
    for i in range(len(squares)):
        xs, ys, xe, ye, size, code = squares[i]
        if code is 'element':
            for j in range(xs, xe+1):
                for k in range(ys, ye+1):
                    element_map[k][j] = 1
    return element_map


def sentence_to_square(image, sentence, sameway_padding = 5, padding=10):
    image_height, image_width, c = image.shape
    square = []
    for i in range(len(sentence)):
        sx, sy, ex, ey, num = sentence[i]
        if sy == ey:
            ssx = max(sx - sameway_padding, 0)
            ssy = max(sy - padding, 0)
            eey = min(ey + padding, image_height - 1)
            eex = min(ex + sameway_padding, image_width - 1)
            size = abs(eex - ssx) * abs(eey - ssy)
            code = "sentence"
        elif sx == ex:
            ssx = max(sx - padding, 0)
            ssy = max(sy - sameway_padding, 0)
            eey = min(ey + sameway_padding, image_height - 1)
            eex = min(ex + padding, image_width - 1)
            size = abs(eex - ssx) * abs(eey - ssy)
            code = "vertical_sentence"
        square.append([ssx, ssy, eex, eey, size, code])
    return square

def spread(map, y, x, area):
    area.append([x, y])
    map[y, x] = 0
    if y-1 >= 0 and map[y-1, x] == 255:
        map, area = spread(map, y-1, x, area)
    if x+1 <= len(map[0])-1 and map[y, x+1] == 255:
        map, area = spread(map, y, x+1, area)
    if y+1 <= len(map[0]) and map[y+1, x] == 255:
        map, area = spread(map, y+1, x, area)
    if x-1 >= 0 and map[y, x-1] == 255:
        map, area = spread(map, y, x-1, area)
    return map, area

def divide_pixel_by_pixel(erased_image, padding =1):
    sys.setrecursionlimit(10000)
    """
    erased_image = cv2.blur(erased_image, (2,1))
    _, erased_image = cv2.threshold(erased_image, 50, 255, cv2.THRESH_BINARY)
    plt.imshow(erased_image,cmap="gray")
    plt.show()
    """
    image_height, image_width = erased_image.shape
    image_map = erased_image.copy()
    area_list = []
    for x in range(image_width):
        for y in range(image_height):
            if image_map[y,x] == 255:
                image_map, area = spread(image_map, y, x, [])
                area_list.append(area)

    square_list = []
    for i in range(len(area_list)):
        number_of_points = len(area_list[i])
        if number_of_points < 4:
            print("just noise")
            continue
        array = np.array(area_list[i])
        sx, sy = np.min(array, axis=0)
        ex, ey = np.max(array, axis=0)
        sx = int(max(0,sx-padding)*3)
        sy = int(max(0,sy-padding)*3)
        ex = int(min(image_width-1, ex+padding+1)*3)
        ey = int(min(image_height-1, ey+padding+1)*3)
        size = abs(ex-sx)*abs(ey-sy)
        square_list.append([sx, sy, ex, ey, size, "component"])
    return square_list






