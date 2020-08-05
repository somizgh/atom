from os.path import join

E_DATA_PATH = "E:\\Ai_projects_data\\atom_data"

# Predict Next Character Model ####################################
PNCM_DIR_NAME = "PNCM"
PNCM_PATH = join(E_DATA_PATH, PNCM_DIR_NAME)

PNCM_MODEL_PREFIX = join(PNCM_PATH, 'PNCM_checkpoints')
PNCM_LATEST_MODEL_DIR = '2020-07-20-06-56'
PNCM_LATEST_MODEL_PATH = join(PNCM_MODEL_PREFIX, PNCM_LATEST_MODEL_DIR)

PNCM_HISTORY_JSON_NAME = "history.json"
PNCM_LATEST_HISTORY_JSON_PATH = join(PNCM_LATEST_MODEL_PATH, PNCM_HISTORY_JSON_NAME)

PNCM_H5_NAME = "my_model.h5"
PNCM_LATEST_H5_PATH = join(PNCM_LATEST_MODEL_PATH, PNCM_H5_NAME)

# Detect Character Model ############################################
DCM_DIR_NAME = "DCM"
DCM_PREFIX = join(E_DATA_PATH, DCM_DIR_NAME)

DCM_MODEL_PREFIX = join(DCM_PREFIX, 'DCM_checkpoints')
DCM_LATEST_MODEL_DIR = '2020-07-20-06-56'
DCM_LATEST_MODEL_PATH = join(DCM_MODEL_PREFIX, DCM_LATEST_MODEL_DIR)

DCM_HISTORY_JSON_NAME = "history.json"
DCM_LATEST_HISTORY_JSON_PATH = join(DCM_LATEST_MODEL_PATH, DCM_HISTORY_JSON_NAME)

DCM_H5_NAME = "DCM.h5"
DCM_LATEST_H5_PATH = join(DCM_LATEST_MODEL_PATH, DCM_H5_NAME)

DCM_JSON_MODEL_NAME = "DCM.json"

# Separate Sentence Straight Model ############################################

SSSM_DIR_NAME = "SSSM"
SSSM_PREFIX = join(E_DATA_PATH, SSSM_DIR_NAME)

SSSM_MODEL_PREFIX = join(SSSM_PREFIX, 'DCM_checkpoints')
SSSM_LATEST_MODEL_DIR = '2020-07-20-06-56'
SSSM_LATEST_MODEL_PATH = join(SSSM_MODEL_PREFIX, SSSM_LATEST_MODEL_DIR)

SSSM_HISTORY_JSON_NAME = "history.json"
SSSM_LATEST_HISTORY_JSON_PATH = join(SSSM_LATEST_MODEL_PATH, SSSM_HISTORY_JSON_NAME)

SSSM_H5_NAME = "SSSM.h5"
SSSM_LATEST_H5_PATH = join(SSSM_LATEST_MODEL_PATH, SSSM_H5_NAME)

SSSM_JSON_MODEL_NAME = "SSSM.json"

# char vocabulary ####################################
CHAR_VOCABULARY_JSON_PREFIX = join(E_DATA_PATH, "dictionary\\char_vocabulary")
CHAR_VOCABULARY_JSON_NAME = "char_vocabulary_list.json"
CHAR_VOCABULARY_JSON_PATH = join(CHAR_VOCABULARY_JSON_PREFIX, CHAR_VOCABULARY_JSON_NAME)

# wiki ################################

WIKI_TEXT_PATH = join(E_DATA_PATH, "ko_wiki_extraction")
WIKI_NPY_PATH = join(E_DATA_PATH, "ko_wiki_toindex")

# full_images ###########################################

FULL_IMAGES_DIR_NAME = "full_images"
FULL_IMAGES_PATH_NAME = join(E_DATA_PATH, FULL_IMAGES_DIR_NAME)

FULL_CHROME_DIR_NAME = "chrome_images"
FULL_CHROME_PATH_NAME = join(FULL_IMAGES_PATH_NAME, FULL_CHROME_DIR_NAME)









