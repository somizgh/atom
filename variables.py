from os.path import join

E_DATA_PATH = "E:\\Ai_projects_data\\atom_data"

# Predict Next Character Model ####################################
PNCM_MODEL_PREFIX = join(E_DATA_PATH, 'PNCM_checkpoints')
PNCM_LATEST_MODEL_DIR = '2020-07-20-06-56'
PNCM_LATEST_MODEL_PATH = join(PNCM_MODEL_PREFIX, PNCM_LATEST_MODEL_DIR)

PNCM_HISTORY_JSON_NAME = "history.json"
PNCM_HISTORY_JSON_PATH = join(PNCM_LATEST_MODEL_PATH, PNCM_HISTORY_JSON_NAME)

PNCM_H5_NAME = "my_model.h5"
PNCM_H5_PATH = join(PNCM_LATEST_MODEL_PATH, PNCM_H5_NAME)

# Detect Character Model ############################################
DCM_MODEL_PREFIX = join(E_DATA_PATH, 'DCM_checkpoints')
DCM_LATEST_MODEL_DIR = '2020-07-20-06-56'
DCM_LATEST_MODEL_PATH = join(DCM_MODEL_PREFIX, DCM_LATEST_MODEL_DIR)

DCM_HISTORY_JSON_NAME = "history.json"
DCM_HISTORY_JSON_PATH = join(DCM_LATEST_MODEL_PATH, DCM_HISTORY_JSON_NAME)

DCM_H5_NAME = "DCM.h5"
DCM_H5_PATH = join(DCM_LATEST_MODEL_PATH, DCM_H5_NAME)

# char vocabulary ####################################
CHAR_VOCABULARY_JSON_PREFIX = join(E_DATA_PATH, "dictionary\\char_vocabulary")
CHAR_VOCABULARY_JSON_NAME = "char_vocabulary_list.json"
CHAR_VOCABULARY_JSON_PATH = join(CHAR_VOCABULARY_JSON_PREFIX, CHAR_VOCABULARY_JSON_NAME)

WIKI_TEXT_PATH = join(E_DATA_PATH, "ko_wiki_extraction")
WIKI_NPY_PATH = join(E_DATA_PATH, "ko_wiki_toindex")


IMAGE_WITH_TEXT_NAME = "image_with_text"
IMAGE_WITH_TEXT_PATH = join(E_DATA_PATH, IMAGE_WITH_TEXT_NAME)






