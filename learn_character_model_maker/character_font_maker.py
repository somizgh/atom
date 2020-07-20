import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


font_path = "E:/Ai_projects_data/atom_data/hangul_fonts/"
save_path = "E:/Ai_projects_data/atom_data/character_data_HES3/"
fonts = os.listdir(font_path)
ttf_fonts = [font for font in fonts if font.endswith(".ttf")]
print(ttf_fonts)


korean_vowel = "ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅒㅔㅖ"
korean_consonant = "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ"
english_capital = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
english_small = "abcdefghijklmnopqrstuvwxyz"
special = "0123456789(){}[]~/!@#$%^&*"

char = korean_vowel+korean_consonant+english_capital+english_small+special

for idx, ttf in enumerate(ttf_fonts):
    print("idx", idx, ttf)
    font = ImageFont.truetype(font=font_path + ttf, size=100)
    font_name = font_manager.FontProperties(fname=font_path+ttf).get_name()
    matplotlib.rc('font', family=font_name)
    plt.rc('font', family=font_name)
    for i in range(len(char)):
        print(char[i])
        x, y = font.getsize(char[i])
        print(x, y)
        the_Image = Image.new('RGB', (125, 125), color="white")
        theDrawPad = ImageDraw.Draw(the_Image)
        theDrawPad.text((int((120-x)/2), int((120-y)/2)), char[i], font=font, fill='black')
        resized = the_Image.resize((25, 25))
        resized.save(save_path+"/font{0:03d}num{1:03d}.jpg".format(int(idx), int(i)))

