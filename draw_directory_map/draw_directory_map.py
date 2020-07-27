import draw_directory_map_config as DDMC
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

    def load_file_function(self):
        if self.name[-2:] is "py":
            self.funcion_list = self.python(self)

    def python(self):
        f = open(self.path, "r")
        for line in f:
            if line.startswith("def"):
                print(line)

    def draw(self, white_board):
        sx = self.layer * DDMC.CELL_POSITION_WIDTH_COEFFICIENT + DDMC.LEFT_PADDING
        sy = self.len * DDMC.CELL_POSITION_HEIGHT_COEFFICIENT + DDMC.UPPER_PADDING
        ex = sx + DDMC.CELL_WIDTH
        ey = sy + DDMC.CELL_HEIGHT

        tsx = sx + DDMC.CELL_TITLE_LEFT_PADDING  # title start x
        tsy = sy + DDMC.CELL_TITLE_UPPER_PADDING

        bsy = tsy + DDMC.CELL_BODY_UPPER_PADDING  # body start x

        color = (255,255,255)

        if self.parent[1] == self.len:
            cv2.line(white_board, (sx-DDMC.CELL_WIDTH_PADDING, sy + DDMC.CELL_LINE_PADDING_Y),
                     (sx+DDMC.CELL_LINE_PADDING_X, sy + DDMC.CELL_LINE_PADDING_Y), (0, 0, 0), 2)
        else:
            px = int(sx - DDMC.CELL_WIDTH_PADDING/2)
            y_apart = self.len - self.parent[1]
            cv2.line(white_board, (px, sy + DDMC.CELL_LINE_PADDING_Y),
                     (sx + DDMC.CELL_LINE_PADDING_X, sy + DDMC.CELL_LINE_PADDING_Y), (0, 0, 0), 2)
            cv2.line(white_board, (px, sy + DDMC.CELL_LINE_PADDING_Y),
                     (px, sy + DDMC.CELL_LINE_PADDING_Y - y_apart*(DDMC.CELL_POSITION_HEIGHT_COEFFICIENT)), (0, 0, 0), 2)

        if self.type is "dir":
            color = DDMC.DIR_COLOR
            cv2.rectangle(white_board, (sx, sy), (ex, ey), color, -1)
            cv2.putText(white_board, str(self.name), (tsx, tsy), DDMC.CELL_TITLE_FONT, DDMC.CELL_TITLE_FONT_SIZE,
                        (0, 0, 0), 1, cv2.LINE_AA)

        elif self.type is "file":
            color = DDMC.FILE_COLOR
            cv2.rectangle(white_board, (sx, sy), (ex, ey), color, -1)
            cv2.putText(white_board, str(self.name), (tsx, tsy), DDMC.CELL_TITLE_FONT, DDMC.CELL_TITLE_FONT_SIZE,
                        (0, 0, 0), 1, cv2.LINE_AA)

        elif self.type is "file_pack":
            color = DDMC.FILE_PACK_COLOR
            cv2.rectangle(white_board, (sx, sy), (ex, ey), color, -1)
            cv2.putText(white_board, "source file", (tsx, tsy), DDMC.CELL_TITLE_FONT, DDMC.CELL_TITLE_FONT_SIZE,
                        (0, 0, 0), 1, cv2.LINE_AA)
            for i in range(len(self.file_names)):
                cv2.putText(white_board, "* " + self.file_names[i], (tsx, bsy+i*DDMC.CELL_BODY_INTERVAL), DDMC.CELL_BODY_FONT,
                            DDMC.CELL_BODY_FONT_SIZE, (0, 0, 0), 1, cv2.LINE_AA)
        color = (255,255,0)
        cv2.rectangle(white_board, (sx, sy), (ex, sy+DDMC.CELL_TITLE_HEIGHT), color, 2)



    def add_file(self, file):
        self.file_names.append(file)


def sort_tree_to_top():
    return 0


def go_to_dir(path, layer, root_len, file_list, tree_map):
    long = 0
    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)) and file not in DDMC.DIRECTORY_IGNORED:
            dir = File(os.path.join(path, file), "dir", layer+1, root_len + long, [layer, root_len])
            file_list.append(dir)
            tree_map[root_len+long][layer+1] = 1
            child_long, _ = go_to_dir(os.path.join(path, file), layer+1, root_len + long, file_list, tree_map)
            long = long + child_long

    make_pack = False
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            if DDMC.DISPLAY_ALL_SOURCE_CODE:
                py = File(os.path.join(path, file), "file", layer+1, root_len + long, [layer, root_len])
                file_list.append(py)
                tree_map[root_len + long][layer + 1] = 1
                long = long + 1
            elif not make_pack:
                pack = File(os.path.join(path, file), "file_pack", layer + 1, root_len + long, [layer, root_len])
                file_list.append(pack)
                tree_map[root_len + long][layer + 1] = 1
                long = long + 1
                make_pack = True
            else:
                pack.add_file(file)
    if long < 1:
        return 1, file_list
    return long, file_list

def draw(tree_map, long):
    white_board_size = (long * DDMC.CELL_POSITION_HEIGHT_COEFFICIENT + DDMC.UPPER_PADDING*2, 1500,3)
    white_board = np.zeros(white_board_size, np.uint8)
    white_board.fill(255)
    for i in range(len(tree_map)):
        c = tree_map[i]
        c.draw(white_board)
    plt.imshow(white_board)
    plt.show()

def draw_directory_tree(path):
    tree_map = [[0 for _ in range(100)] for _ in range(100)]
    root = File(path, "dir", 0, 0, [0, 0])
    file_list = [root]
    long, file_list = go_to_dir(path, 0, 0, file_list, tree_map)
    print(long)
    draw(file_list, long)




if __name__ == "__main__":
    draw_directory_tree("D:/atom")
