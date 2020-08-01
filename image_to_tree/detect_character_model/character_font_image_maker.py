import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from detect_character_model_variables import *
from dictionary.character_vocabulary import CharVocabulary


def make_character_image(number_of_font_use_option, font_path, image_save_path, font_size=16):
    char_vocab = CharVocabulary().load_char_vocabulary().convert_detectable_char_vocabulary()
    char = char_vocab.return_characters()
    fonts = os.listdir(font_path)
    ttf_fonts = [font for font in fonts if font.endswith(".ttf")]
    if number_of_font_use_option is "max":
        number_of_font_use = len(ttf_fonts)
    elif number_of_font_use_option > len(ttf_fonts):
        print("number of font use option {0}, exceed max fonts number {1}".format(number_of_font_use_option, len(ttf_fonts)))
        number_of_font_use = len(ttf_fonts)
    else:
        number_of_font_use = number_of_font_use_option
    ttf_fonts = ttf_fonts[:number_of_font_use]

    for idx, ttf in enumerate(ttf_fonts):
        print("idx", idx, ttf)
        font = ImageFont.truetype(font=os.path.join(font_path, ttf), size=font_size)
        font_name = font_manager.FontProperties(fname=os.path.join(font_path, ttf)).get_name()
        matplotlib.rc('font', family=font_name)
        plt.rc('font', family=font_name)
        for i in range(len(char)):
            x, y = font.getsize(char[i])
            the_Image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), color="white")
            theDrawPad = ImageDraw.Draw(the_Image)
            theDrawPad.text((int((IMAGE_WIDTH - x) / 2), int((IMAGE_HEIGHT - y) / 2)), char[i], font=font, fill='black')
            the_Image.save(os.path.join(image_save_path, "font{0:03d}num{1:03d}.jpg".format(int(idx), int(i))))
    return 0


if __name__ == "__main__":
    make_character_image(1000, FONT_DIR_PATH, CHARACTER_IMAGES_DIR_PATH, FONT_SIZE)
