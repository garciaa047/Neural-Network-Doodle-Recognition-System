# Dir variables (Can be changed to absolute path C:/etc)
FILE_PATH_DIR = ""
DATASET_DIR = "doodleDataset"
CSV_FILE = "doodle_pixels.csv"
NPZ_FILE = "doodle_model.npz"

# Image Variables
NUM_ITEMS = 5                                       # Can be changed to include more or less depending on how many images you have in your dataset
IMAGES_PER_ITEM = 500                               # Number of images per item you would like to train on
IMAGE_SIZE = (30, 30)                               # Image size in pixels, 30 x 30 pixels
NUM_PIXELS = IMAGE_SIZE[0] * IMAGE_SIZE[1]          # 30 x 30 = 900 pixels per image
AUGMENT_COPIES = 2                                  # Number of additional augmented copies per image. 
                                                    # Total number of images will be (1 + AUGMENTED_COPIES) * IMAGES_PER_ITEM

# Training Variables
TEST_SPLIT = 0.2                                    # Amount of data to test with, 0.2 -> 20% test and 80% Train
LEARNING_RATE = 0.001                               # Learning Rate, lower the more stable
HIDDEN_LAYERS = [128, 64]                           # Number of hidden layers and their size, currently (900 -> 128 -> 64 -> 5)
BATCH_SIZE = 64                                     # Batch Size
EPOCHS = 500                                        # Number of epochs
LEAKY_ALPHA = 0.1                                   # Alpha value for Leaky ReLU
LAMBDA_REG = 0.003                                  # Lambda for L2 regularization strength

# Contains the names of the classes, will be updated every time datasetToCsv.py is ran
CLASS_NAMES = ['apple', 'pants', 'airplane', 'clock', 'tree']
