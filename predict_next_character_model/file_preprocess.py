import numpy as np
import jamotools
from character_vocabulary import CharVocabulary
import os
dictionary = CharVocabulary()
def del_multiple_newline(path):
    for root, dirs, files in os.walk(path):
        rootpath = os.path.join(os.path.abspath(path), root)
        for file in files:
            serial_newline = False
            print("del_multiple_newline from",file)
            filepath = os.path.join(rootpath,file)
            f = open(filepath,'r',encoding="UTF-8")
            reader = f.read()
            f.close()
            w =  open(filepath,'w',encoding="UTF-8")
            for c in reader:
                if c =="\n":
                    if serial_newline is True:
                        continue
                    else:
                        serial_newline=True
                else:
                    serial_newline=False
                w.write(c)
            w.close()

def txtfile2npyfile(input_path, output_path, vocabulary):

    file_counter = 1
    for root, dirs, files in os.walk(input_path):
        rootpath = os.path.join(os.path.abspath(input_path), root)
        for file in files:
            int_array = []
            filepath = os.path.join(rootpath, file)
            f = open(filepath, encoding="UTF-8")
            for line in f:
                split = jamotools.split_syllables(line)
                for c in split:
                    index = vocabulary.char2index(c)
                    if index > 0:
                        int_array.append(index)
            file_name = "wiki_korean_{0:04}".format(file_counter) + ".npy"
            save_path =os.path.join(os.path.abspath(output_path),file_name)
            np.save(save_path, np.array(int_array))
            print("txtfile2npyfile convert to : {0}".format(file_name)+" done ")
            file_counter = file_counter + 1
