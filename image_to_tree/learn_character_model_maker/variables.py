
NUMBER_OF = {
        'HANGUL_VOWEL': 14,
        'HANGUL_CONSONANT': 14,
        'ENGLISH_CHARACTER': 52,
        'SPECIAL_CHARACTER': 16,
        'NUMBER_CHARACTER': 10,
        'TOTAL': 106

}
LENGTH_OF = {
    'RESIZED_IMAGE_HORIZONTAL' : 25,
    'RESIZED_IMAGE_VERTICAL' : 25

}


EPOCH = 2000

TRAIN_SPLIT = 0.7
CHARACTER_DATA_PATH = "E:/Ai_projects_data/atom_data/character_data_HES3"
BATCH_SIZE = 1

#  겹치는 문자들
ZERO_YOUNG_IEUNG = [21, 42, 68, 80]  # 동그라미들.  이응, 대문자 O, 소문자 o, 숫자 0
BIG_SAME_SMALL = [56, 69, 72, 75, 76, 77, 79]  # 대소문자 같은 소문자들 -26을 한다. C,P,S,V,W,X,Z
l_SAME = [9, 36, 65]  # 세로 막대기. 모음 ㅣ,대문자 I, 소문자 l,
