import os
import time
import tensorflow as tf

def write_model_txt(model, path):

    f = open(path+"/model_description.txt",'w')
    tf.summary.create_file_writer



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