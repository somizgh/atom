import pickle
import os
import numpy as np
from variables import *
import json

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
INITIAL_SOUND_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
MIDDLE_SOUND_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ',
                     'ㅢ', 'ㅣ']
FINAL_SOUND_LIST = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ',
                    'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
CAPITAL_ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y', 'Z']
SMALL_ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                  'u', 'v', 'w', 'x', 'y', 'z']
SPECIAL_CHARACTER = ['`', '-', '=', '~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+','[', ']', '\\', '{', '}', '|',
                     ';', '\'', ':', '\"', ',', '.', '/', '<', '>', '?', ' ', '\n', '\t']


CHARACTERS = []
CHARACTERS.extend(NUMBER)
CHARACTERS.extend(INITIAL_SOUND_LIST)
CHARACTERS.extend(MIDDLE_SOUND_LIST)
CHARACTERS.extend(FINAL_SOUND_LIST)
CHARACTERS.extend(CAPITAL_ALPHABET)
CHARACTERS.extend(SMALL_ALPHABET)
CHARACTERS.extend(SPECIAL_CHARACTER) # 148개


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
        with open(CHAR_VOCABULARY_JSON_PATH, "r") as vocab:
            self.char2index_vocab = json.load(vocab)
        return True

    def save_char_vocabulary(self):
        print("a")
        json.dump(self.char2index_vocab, open(CHAR_VOCABULARY_JSON_PATH, 'w'))
        return True

    def add_char_vocabulary(self, char):
        print(char, self.char2index_vocab.keys())
        if char not in self.char2index_vocab.keys():
            print("dictionary add "+char+" index "+str(len(self.char2index_vocab)+1))
            self.char2index_vocab[char] = len(self.char2index_vocab)+1
            return True
        return False

    def len_dictionary(self):
        return len(self.char2index_vocab)

    def print_key_value(self):
        for i in range(1,len(self.char2index_vocab)+1):
            print("key: ",i, "  value : ",self.index2char(i))

