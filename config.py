import torch

# [GENERAL SETTINGS]
DEBUG = False                       # Set to True to enable debugging features
SEED = 42                           # Random seed for reproducibility



# [DATA PATHS]
DATA_DIR = "data/raw/"      # Path to the dataset directory
TRAIN_DATA_DIR = DATA_DIR + "train.csv" # Path to training data
VAL_DATA_DIR = DATA_DIR + "val.csv"   # Path to validation data
TEST_DATA_DIR = DATA_DIR + "test.csv"   # Path to test data
SAVE_MODEL_PATH = "models/"         # Where to save models



# [TEXT PROCESSING]
MAX_SEQ_LENGTH = 50                 # Maximum length of input/output sequences
VOCAB_SIZE = 10000                  # Maximum vocabulary size
PAD_TOKEN = 0                       # Padding token
SOS_TOKEN = 1                       # Start-of-sequence token
EOS_TOKEN = 2                       # End-of-sequence token
UNK_TOKEN = 3                       # Unknown token



# [MODEL ARCHITECTURES PARAMETERS]
EMBEDDING_SIZE = 256                # Embedding size
HIDDEN_SIZE = 512                   # Hidden size
NUM_LAYERS = 3                      # Number of RNN layers
DROPOUT_RATE = 0.1                  # Dropout rate
ENCODER_BIDIRECTIONAL = True        # Use bidirectional encoder


# [TRAINING PARAMETERS]
BATCH_SIZE = 32                     # Number of samples per batch
EPOCHS = 20                         # Number of training epochs
LEARNING_RATE = 0.001               # Initial learning rate
TEACHER_FORCING_RATIO = 0.5         # Probability of using teacher forcing in decoder



# [HARDWARE SETTINGS]
USE_GPU = True                      # Use GPU if available
NUM_WORKERS = 4                     # Number of data loading workers
DEVICE = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")



# [LOGGING AND CHECKPOINTS]
CHECKPOINT_PATH = "checkpoints/"    # Path to save checkpoints
SAVE_EVERY = 5                      # Save model every X epochs
LOG_INTERVAL = 50                   # Print logs every X batches
