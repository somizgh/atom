from anytree import Node, RenderTree
from anytree.exporter import DotExporter
from pytesseract import *
import cv2
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import matplotlib.pyplot as plt
import copy
import numpy as np
import skimage



def find_best_node(parent, square, fun_code):
    xs, ys, xe, ye, size, code = square
    for i in range(len(parent.children)):
        pxs, pys, pxe, pye, psize = parent.children[i].position
        if xs-pxs >= 0 and ys-pys >= 0 and pxe-xe >= 0 and pye-ye >= 0: # 같거나 포함되면
            find_best_node(parent.children[i], square,fun_code)
            return 0
        elif pxs < xs < pxe and fun_code is "make_tree": # 이전의 사각형과 겹친다.
            if pys < ys < pye or pys < ye < pye:
                return 0
        elif pxs < xe < pxe and fun_code is "make_tree": # 겹친다.
            if pys < ys < pye or pys < ye < pye:
                return 0

    pxs, pys, pxe, pye, psize = parent.position
    pro = round(size/psize, 3)

    if fun_code is "add_component":
        n = Node("s({},{}),e({},{}),size{},type:{},pro{}".format(xs, ys, xe, ye, size, code, pro), parent=parent,
                 type=square[5], position=square[:5], proportion=pro, data=[], image=None, no_straight_image=None)
        return 0

    elif pro > 0.5 and square[5] == 'square':
        square[5] = 'image'
    n = Node("s({},{}),e({},{}),size{},type:{},pro{}".format(xs, ys, xe, ye, size, code, pro), parent=parent,
            type=square[5], position=square[:5], proportion=pro,  data=[], image=None, no_straight_image=None)
    return 0



def make_tree(squares, element_map):
    if len(squares) > 0:
        root = Node("root", parent=None, type="square", position=squares[0][:5], proportion=1, data=[], image=None, no_straight_image=None)
    else:
        print("no square")
        return 0
    for i in range(1, len(squares)):
        append, code = not_a_middle(squares[i], element_map)  # 표같은 것에서 중간에 낀 작은 사각형들을 제거
        if append is True:
            squares[i][5] = code
            find_best_node(root, squares[i], "make_tree")


    return root


def not_a_middle(square, element_map):
    sx, sy, ex, ey, size, code = square
    if code is 'element':
        return True, 'element'

    #  내부검사
    for i in range(sx, ex+1):
        for j in range(sy, ey+1):
            if element_map[j][i] is not 1:
                return True, 'square'
    # 외곽검사
    if sx-1 >= 0:
        out_sum = 0
        for i in range(sy, ey+1):
            out_sum = out_sum + element_map[i][sx-1]
        if out_sum == ey-sy+1:
            return False, 'None'
    if ex+1 <= len(element_map[0])-1:
        out_sum = 0
        for i in range(sy, ey+1):
            out_sum = out_sum + element_map[i][ex+1]
        if out_sum == ey-sy+1:
            return False, 'None'
    if sy-1 >= 0:
        out_sum = 0
        for i in range(sx, ex+1):
            out_sum = out_sum + element_map[sy-1][i]
        if out_sum == ex-sx+1:
            return False, 'None'
    if ey+1 <= len(element_map)-1:
        out_sum = 0
        for i in range(sx, ex+1):
            out_sum = out_sum + element_map[ey+1][i]
        if out_sum == ex-sx+1:
            return False, 'None'
    return True, 'table'


def append_tree(squares, root):
    for i in range(0, len(squares)):
        find_best_node(root, squares[i], "append_tree")
    return root


def add_image(root, image):
    descendants = root.descendants
    root.image = image
    for child in descendants:
        sx, sy, ex, ey, size = child.position
        new_image = image.copy()
        new_image = new_image[sy:ey, sx:ex]
        child.image = new_image

    return root

def add_component(root, squares):
    for i in range(len(squares)):
        find_best_node(root, squares[i], "add_component")
    return root

def node_image_to_string(node):
    sx,sy,ex,ey,size = node.position
    originalBGR = node.image
    gray = cv2.cvtColor(originalBGR, cv2.COLOR_BGR2GRAY)

    sharp = gray.copy()
    image_enhanced = gray.copy()
    canny = gray.copy()
    min_pooling = gray.copy()
    max_pooling = gray.copy()
    sharpX2 = gray.copy()
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])


    # 커널 적용
    image_sharp = cv2.filter2D(sharp, -1, kernel)
    image_enhanced = cv2.equalizeHist(image_enhanced)
    canny = cv2.Canny(canny, 0, 50)
    thres = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #blur = cv2.medianBlur(gray, 10)
    sharpX2 = image_sharp.copy()
    sharpX2 = cv2.filter2D(sharpX2,-1,kernel)
    bitwise_not = cv2.bitwise_not(originalBGR)
    sharp_bitwise_not = cv2.bitwise_not(sharp)


    print("original",pytesseract.image_to_string(originalBGR, lang='kor+eng'))
    print("sharp",pytesseract.image_to_string(image_sharp, lang='kor+eng'))
    print("enhence",pytesseract.image_to_string(image_enhanced, lang='kor+eng'))
    print("canny",pytesseract.image_to_string(canny, lang='kor+eng'))
    print("thres", pytesseract.image_to_string(thres, lang='kor+eng'))
    #print("max", pytesseract.image_to_string(blur, lang='kor+eng'))
    print("sharpX2", pytesseract.image_to_string(sharpX2, lang='kor+eng'))
    print("bitwise", pytesseract.image_to_string(bitwise_not, lang='kor+eng'))
    print("sharp bitwise", pytesseract.image_to_string(sharp_bitwise_not, lang='kor+eng'))


    print("data", pytesseract.image_to_data(originalBGR,lang='kor+eng'))
    fig, axes = plt.subplots(nrows=3, ncols=3)
    axes[0][0].imshow(originalBGR)
    axes[0][0].set_title("original")
    axes[0][1].imshow(image_sharp,cmap="gray")
    axes[0][1].set_title("sharp")
    axes[0][2].imshow(image_enhanced,cmap = "gray")
    axes[0][2].set_title("enhanced")
    axes[1][0].imshow(canny,cmap="gray")
    axes[1][0].set_title("canny")
    axes[1][1].imshow(thres, cmap="gray")
    axes[1][1].set_title("thres")
    #axes[1][2].imshow(blur, cmap="gray")
    #axes[1][2].set_title("blur")
    axes[2][0].imshow(sharpX2, cmap="gray")
    axes[2][0].set_title("sharpX2")
    axes[2][1].imshow(bitwise_not, cmap="gray")
    axes[2][1].set_title("bitwise")
    axes[2][2].imshow(sharp_bitwise_not, cmap="gray")
    axes[2][2].set_title("sharp_bitwise")
    plt.show()

    #print("osd", pytesseract.image_to_osd(only_this_node))







    return 0

def find_string(root):
    if root.image is None:
        print("need original image")
        return 0

    for node in root.descendants:
        print("node", node.type)
        if node.type is "component":
            node_image_to_string(node)



    '''

    txt = pytesseract.image_to_string(source_image, lang='kor')
    print(txt)
    print("box", pytesseract.image_to_boxes(source_image, lang='eng+kor'))
    print("data", pytesseract.image_to_data(source_image))
    print("osd", pytesseract.image_to_osd(source_image))

'''

    return root


def delete_squares_image(image, root):
    descendants = root.descendants
    color = (0, 0, 0)
    size = 10
    for i in range(len(descendants)):
        sx, sy, ex, ey, square_size, code = descendants[i].position
        cv2.line(image, (sx, sy), (sx, ey), color, size)
        cv2.line(image, (sx, sy), (ex, sy), color, size)
        cv2.line(image, (sx, ey), (ex, ey), color, size)
        cv2.line(image, (ex, sy), (ex, ey), color, size)
    return image


def draw_tree(root):
    for pre, fill, node in RenderTree(root):
        print("%s%s"%(pre, node.name))
    return 0