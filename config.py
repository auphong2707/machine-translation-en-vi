import torch

# [GENERAL SETTINGS]
DEBUG = False                       # Set to True to enable debugging features
SEED = 42                           # Random seed for reproducibility



# [DATA PATHS]
DATA_DIR = "data/"      # Path to the dataset directory
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



# [MODEL ARCHITECTURES PARAMETERS]

# RNN
RNN_EMBEDDING_SIZE = 128                # Embedding size
RNN_HIDDEN_SIZE = 128                   # Hidden size
RNN_NUM_LAYERS = 3                      # Number of RNN layers
RNN_DROPOUT_RATE = 0.1                  # Dropout rate
RNN_ENCODER_BIDIRECTIONAL = True        # Use bidirectional encoder

# RNN with Attention
RNN_ATTN_EMBEDDING_SIZE = 128           # Embedding size
RNN_ATTN_HIDDEN_SIZE = 128              # Hidden size
RNN_ATTN_NUM_LAYERS = 3                 # Number of RNN layers
RNN_ATTN_DROPOUT_RATE = 0.1             # Dropout rate
RNN_ATTN_ENCODER_BIDIRECTIONAL = True   # Use bidirectional encoder


# [TRAINING PARAMETERS]
BATCH_SIZE = 128                    # Number of samples per batch
EPOCHS = 50                         # Number of training epochs
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

# [ADDITIONAL SETTINGS]
BEAM_WIDTH = 3                      # Beam width for beam search
EXPERIMENT_NAME = "experiment_0"     # Name of the experiment (Change this)