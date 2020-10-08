import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt

def write_model_txt(model, path):
    f = open(path+"/model_description.txt",'w')
    return True


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            return True
    except OSError:
        print("Error: making directory {}".format(directory))
    return False


def time_format():
    return time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))

def draw_2d_list(numpy):
    for i in range(len(numpy)):
        for j in range(len(numpy[0])):
            print("{0:03}".format(numpy[i][j]), end =" ")
        print("")


def display_n_m_images(rows,cols,image_list,title_list):
    fig = plt.figure()

    axis_list = []
    for i in range(rows*cols):
        axn = fig.add_subplot(rows,cols,i+1)

        if len(image_list[i].shape) == 3:
            axn.imshow(image_list[i])
        else:
            axn.imshow(image_list[i],cmap="gray",)
        axn.set_title(title_list[i])
        axn.axis("off")
    fig.tight_layout()
    plt.show()



