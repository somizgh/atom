import variables as VARIABLES

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

CHARACTERS_NOT_USED_IN_DCM = ['ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ',
                              'ㅘ', 'ㅙ', 'ㅚ', 'ㅝ', 'ㅟ', 'ㅞ', 'ㅟ', 'ㅢ',
                              'ㄳ', 'ㄵ', 'ㄶ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅄ',
                              ' ', '\n', '\t']
CHARACTERS_CANNOT_DISTINGUISH ={'c':'C', 'l':'ㅣ','s':'S','v':'V','w':'W','x':'X','z':'Z'}


