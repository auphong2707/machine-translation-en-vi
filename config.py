import torch

# [GENERAL SETTINGS]
DEBUG = False                       # Set to True to enable debugging features
SEED = 42                           # Random seed for reproducibility



# [DATA PATHS]
DATA_DIR = "data/raw/"      # Path to the dataset directory
TRAIN_DATA_DIR = DATA_DIR + "train_50k.csv" # Path to training data
VAL_DATA_DIR = DATA_DIR + "val.csv"   # Path to validation data
TEST_DATA_DIR = DATA_DIR + "test.csv"   # Path to test data
SAVE_MODEL_PATH = "models/"         # Where to save models



# [TEXT PROCESSING]
MAX_SEQ_LENGTH = 50                 # Maximum length of input/output sequences
VOCAB_SIZE = 20000                  # Maximum vocabulary size
PAD_TOKEN = 0                       # Padding token
SOS_TOKEN = 1                       # Start-of-sequence token
EOS_TOKEN = 2                       # End-of-sequence token
UNK_TOKEN = 3                       # Unknown token



# [RNN ARCHITECTURES PARAMETERS]
EMBEDDING_SIZE = 256                # Embedding size (Change this)
HIDDEN_SIZE = 512                   # Hidden size (Change this)
NUM_LAYERS = 3                      # Number of RNN layers (Change this)
DROPOUT_RATE = 0.1                  # Dropout rate (Change this)
ENCODER_BIDIRECTIONAL = True        # Use bidirectional encoder
TEACHER_FORCING_RATIO = 0.5         # Probability of using teacher forcing in decoder (Change this)

# [TRANSFORMER PARAMETERS]
TFM_EMBEDDING_SIZE = 512            # Transformer embedding size
TFM_DROPOUT_RATE = 0.5              # Transformer dropout rate
TFM_NUM_LAYERS = 4                  # Number of transformer layers
TFM_NUM_HEADS = 8                   # Number of attention heads
TFM_DIM_FEED_FORWARD = 2048          # Dimension of feed forward network

# [TRAINING PARAMETERS]
BATCH_SIZE = 256                    # Number of samples per batch
EPOCHS = 50                         # Number of training epochs
LEARNING_RATE = 0.001               # Initial learning rate

# [HARDWARE SETTINGS]
USE_GPU = True                      # Use GPU if available
NUM_WORKERS = 4                     # Number of data loading workers
DEVICE = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")



# [LOGGING AND CHECKPOINTS]
CHECKPOINT_PATH = "checkpoints/"    # Path to save checkpoints
SAVE_EVERY = 5                      # Save model every X epochs
LOG_INTERVAL = 50                   # Print logs every X batches

# [ADDITIONAL SETTINGS]
BEAM_WIDTH = 5                      # Beam width for beam search
EXPERIMENT_NAME = "transformer_one"     # Name of the experiment (Change this)