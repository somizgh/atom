import variables as VARIABLES
from os.path import join



IMAGE_WIDTH = 100
IMAGE_HEIGHT = 20

SSSM_DIR_NAME = "SSSM"
SSSM_DIR_PATH = join(VARIABLES.E_DATA_PATH, SSSM_DIR_NAME)


SS_IMAGES_DIR_NAME = "ss_images"
SS_IMAGES_DIR_PATH = join(SSSM_DIR_PATH, SS_IMAGES_DIR_NAME)

SS_GRADIENT_IMAGE_DIR_NAME = "ss_gradient_images"
SS_GRADIENT_IMAGE_DIR_PATH = join(SSSM_DIR_PATH, SS_GRADIENT_IMAGE_DIR_NAME)

SS_NUMPY_DIR_NAME = "SSSM_input_data"
SS_NUMPY_DIR_PATH = join(SSSM_DIR_PATH, SS_NUMPY_DIR_NAME)

CHECKPOINTS_DIR_NAME = "SSSM_checkpoints"
CHECKPOINTS_DIR_PATH = join(SSSM_DIR_PATH, CHECKPOINTS_DIR_NAME)

INPUT_DATA_X_NAME = "dataX"
INPUT_DATA_Y_NAME = "dataY"
INPUT_DATA_COUNTER_FORMAT = "_{0:>04}.npy"


IMAGES_PER_FILE = 1000

EPOCH = 2000

TRAIN_SPLIT = 0.7
BATCH_SIZE = 1
