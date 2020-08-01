import os
import time
import tensorflow as tf

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



