from draw_directory_map_config import *
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

class File:
    def __init__(self, path, type, layer, len, parent):
        self.name = os.path.basename(path)
        self.type = type
        self.layer = layer
        self.len = len
        self.file_names = []
        self.parent = parent
        if self.type is "file":
            self.load_file_function()
        if self.type is "file_pack":
            self.file_names.append(self.name)

        self.sx = self.layer * (WB_WIDTH_BETWEEN_CELLS + CELL_WIDTH) + WB_LEFT_PADDING
        self.sy = 0
        self.ex = self.sx + CELL_WIDTH
        self.ey = 0

        self.tsx = self.sx
        self.tsy = 0
        self.tex = self.ex
        self.tey = 0

        self.bsx = self.sx
        self.bsy = 0
        self.bex = self.ex
        self.bey = 0

        self.psy = 0
        self.pey = 0

    def load_file_function(self):
        if self.name[-2:] is "py":
            self.funcion_list = self.python(self)

    def python(self):
        f = open(self.path, "r")
        for line in f:
            if line.startswith("def"):
                print(line)

    def set_position(self, parent_sy, parent_ey):
        self.psy = parent_sy
        self.pey = parent_ey

        self.sy = self.pey + WB_HEIGHT_BETWEEN_CELLS
        if self.type is "dir":
            self.ey = self.sy + CELL_STANDARD_HEIGHT
        elif self.type is "file_pack":
            self.ey = self.sy + max(CELL_STANDARD_HEIGHT, CELL_TITLE_HEIGHT +
                                    len(self.file_names) * (CELL_BODY_FONT_SIZE + CELL_HEIGHT_BETWEEN_TEXTS))
        self.tsy = self.sy
        self.tey = self.tsy + CELL_TITLE_HEIGHT
        self.bsy = self.tey + DDMC.CELL_BODY_UPPER_PADDING
        self.bey = self.bsy
        return self.sy, self.ey


    def draw(self, white_board):
        print("name", self.name)
        print("my",self.layer, self.len)
        print("parent", self.parent)
        print(" ")
        # 직선 먼저 연결
        if self.parent[1] == self.len:  # 부모 파일과 같은 len
            cv2.line(white_board, (self.tsx-DDMC.CELL_BETWEEN_WIDTH_PADDING, int((self.tsy + self.tey)/2)),
                     (self.tsx, int((self.tsy + self.tey)/2)), (0, 0, 0), 2)
        else:  # 부모 파일 아래로 떨어져있다.
            cv2.line(white_board, (self.tsx - int(DDMC.CELL_BETWEEN_WIDTH_PADDING/2), int((self.tsy + self.tey) / 2)),
                     (self.tsx, int((self.tsy + self.tey) / 2)), (0, 0, 0), 2)
            cv2.line(white_board, (self.tsx - int(DDMC.CELL_BETWEEN_WIDTH_PADDING/2), self.psy+int(DDMC.CELL_TITLE_HEIGHT/2)),
                     (self.tsx - int(DDMC.CELL_BETWEEN_WIDTH_PADDING/2), int((self.tsy + self.tey) / 2)), (0, 0, 0), 2)

        if self.type is "dir":
            color = DDMC.DIR_COLOR
            cv2.rectangle(white_board, (self.sx, self.sy), (self.ex, self.ey), color, -1)
            color = (144, 144, 144)
            cv2.rectangle(white_board, (self.tsx, self.tsy), (self.tex, self.tey), color, -1)
            cv2.putText(white_board, str(self.name),
                        (self.tsx+DDMC.CELL_TITLE_LEFT_PADDING, self.tsy+DDMC.CELL_TITLE_UPPER_PADDING),
                        DDMC.CELL_TITLE_FONT, DDMC.CELL_TITLE_FONT_SIZE, (0, 0, 0), 1, cv2.LINE_AA)

        elif self.type is "file":
            color = DDMC.FILE_COLOR
            cv2.rectangle(white_board, (self.sx, self.sy), (self.ex, self.ey), color, -1)
            color = (144, 144, 144)
            cv2.rectangle(white_board, (self.tsx, self.tsy), (self.tex, self.tey), color, -1)
            cv2.putText(white_board, str(self.name),
                        (self.tsx + DDMC.CELL_TITLE_LEFT_PADDING, self.tsy + DDMC.CELL_TITLE_UPPER_PADDING),
                        DDMC.CELL_TITLE_FONT, DDMC.CELL_TITLE_FONT_SIZE, (0, 0, 0), 1, cv2.LINE_AA)

        elif self.type is "file_pack":
            color = DDMC.FILE_PACK_COLOR
            cv2.rectangle(white_board, (self.sx, self.sy), (self.ex, self.ey), color, -1)
            color = (144, 144, 144)
            cv2.rectangle(white_board, (self.tsx, self.tsy), (self.tex, self.tey), color, -1)
            cv2.putText(white_board, str(self.name),
                        (self.tsx + DDMC.CELL_TITLE_LEFT_PADDING, self.tsy + DDMC.CELL_TITLE_UPPER_PADDING),
                        DDMC.CELL_TITLE_FONT, DDMC.CELL_TITLE_FONT_SIZE, (0, 0, 0), 1, cv2.LINE_AA)
            for i in range(len(self.file_names)):
                cv2.putText(white_board, "* " + self.file_names[i], (self.bsx, self.bsy+i*DDMC.CELL_BODY_INTERVAL), DDMC.CELL_BODY_FONT,
                            DDMC.CELL_BODY_FONT_SIZE, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.line(white_board, (self.bsx, self.bsy+i*DDMC.CELL_BODY_INTERVAL+DDMC.CELL_BODY_UNDERLINE_PADDING),
                         (self.bex, self.bsy+i*DDMC.CELL_BODY_INTERVAL+DDMC.CELL_BODY_UNDERLINE_PADDING), (0, 0, 0), 1)



    def add_file(self, file):
        self.file_names.append(file)

    def return_layer_file_list(self):
        return self.layer, len(self.file_names)

    def return_long_layer(self):
        return self.len, self.layer


def sort_tree_to_top():
    return 0


def go_to_dir(path, file_list, root_len, root_layer):
    long = 0
    child_max_layer = root_layer
    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)) and file not in DDMC.DIRECTORY_IGNORED:
            print(file)
            dir = File(os.path.join(path, file), "dir", root_layer+1, root_len + long, [root_layer, root_len])
            file_list.append(dir)
            _, child_long, child_layer = go_to_dir(os.path.join(path, file), file_list, root_len + long, root_layer+1)
            child_max_layer = max(child_max_layer, child_layer)
            long = long + child_long

    make_pack = False
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            if DDMC.DISPLAY_ALL_SOURCE_CODE:
                py = File(os.path.join(path, file), "file", root_layer+1, root_len + long, [root_layer, root_len])
                file_list.append(py)
                long = long + 1
            elif not make_pack:
                pack = File(os.path.join(path, file), "file_pack", root_layer + 1, root_len + long, [root_layer, root_len])
                file_list.append(pack)
                long = long + 1
                make_pack = True
            else:
                pack.add_file(file)
    if long < 1:
        return file_list, 1, child_max_layer
    return file_list, long, child_max_layer

def draw(tree_map, long, layer):
    print(long,layer)

    white_board_size = (long * DDMC.CELL_POSITION_HEIGHT_COEFFICIENT + DDMC.UPPER_PADDING*2,
                        layer*DDMC.CELL_POSITION_WIDTH_COEFFICIENT + DDMC.LEFT_PADDING*2, 3)
    white_board = np.zeros(white_board_size, np.uint8)
    white_board.fill(255)
    sy_ey_list = [[0, 0] for _ in range(layer)]  # psy, pey, long_count
    if DDMC.DRAW_DENSE_MAP:
        for i in range(len(tree_map)):
            c = tree_map[i]
            long, layer = c.return_long_layer()
            psy, pey = sy_ey_list[layer]
            psy, pey = c.set_position(psy, pey)
            sy_ey_list[layer] = [psy, pey]
            c.draw(white_board)
    else:
        long_previous = 0
        for i in range(len(tree_map)):
            c = tree_map[i]
            long, layer = c.return_long_layer()
            if long_previous != long:
                max_previous = max([a[1] for a in sy_ey_list])
                for j in range(len(sy_ey_list)):
                    sy_ey_list[j][1] = max_previous
                    long_previous = long
            psy, pey = sy_ey_list[layer]
            psy, pey = c.set_position(psy, pey)
            sy_ey_list[layer] = [psy, pey]
            c.draw(white_board)
    plt.imshow(white_board)
    plt.show()

def draw_directory_tree(path):
    root = File(path, "dir", 0, 0, [0, 0])
    file_list = [root]
    file_list, long, layer = go_to_dir(path, file_list, root_len=0, root_layer=0)
    draw(file_list,long,layer+1)




if __name__ == "__main__":
    draw_directory_tree("D:/atom")
