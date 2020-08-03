from character_vocabulary_variables import *
import numpy as np
import variables as VARIABLES
import json

class CharVocabulary:

    def __init__(self):
        self.vocab_file_name = "char_vocabulary_list.npy"
        self.char2index_vocab = {}
        for v in CHARACTERS:
            if v not in self.char2index_vocab.keys():
                self.char2index_vocab[v] = len(self.char2index_vocab) + 1

    def char2index(self, char):
        try:
            return self.char2index_vocab[char]
        except KeyError:
            return -1
        return self.char2index_vocab[char]

    def index2char(self, index):
        if index > len(self.char2index_vocab.keys()):
            return "None"
        return list(self.char2index_vocab.keys())[int(index)-1]

    def chars2indexes(self, numpy_list):
        list = []
        for c in numpy_list:
            a = self.char2index(c)
            if a == -1:
                continue
            list.append(a)
        return np.array(list)

    def indexes2chars(self, numpy_list):
        key_list = list(self.char2index_vocab.keys())
        return np.array([key_list[int(i)-1] for i in numpy_list])

    def load_char_vocabulary(self):
        with open(VARIABLES.CHAR_VOCABULARY_JSON_PATH, "r") as vocab:
            self.char2index_vocab = json.load(vocab)
        return self

    def save_char_vocabulary(self):
        print("save char vocabulary to "+str(VARIABLES.CHAR_VOCABULARY_JSON_PATH))
        json.dump(self.char2index_vocab, open(VARIABLES.CHAR_VOCABULARY_JSON_PATH, 'w'))
        return True

    def add_char_vocabulary(self, char):
        print("add char" + char + "to char vocabulary", self.char2index_vocab.keys())
        if char not in self.char2index_vocab.keys():
            print("dictionary add "+char+" index "+str(len(self.char2index_vocab)+1))
            self.char2index_vocab[char] = len(self.char2index_vocab)+1
            return True
        return False

    def len_dictionary(self):
        return len(self.char2index_vocab)

    def return_keys(self):
        return list(self.char2index_vocab.keys())

    def return_characters(self):
        return list(self.char2index_vocab.keys())

    def convert_detectable_char_vocabulary(self):
        keys = list(self.char2index_vocab.keys())
        for i in range(len(keys)):
            if keys[i] in CHARACTERS_NOT_USED_IN_DCM:
                del self.char2index_vocab[keys[i]]
            elif keys[i] in CHARACTERS_CANNOT_DISTINGUISH:
                del self.char2index_vocab[keys[i]]
        keys=list(self.char2index_vocab.keys())
        for k in enumerate(keys):
            self.char2index_vocab[k[1]] = k[0]
        print("success convert to detectable char vocabulary", self.char2index_vocab)
        return self

