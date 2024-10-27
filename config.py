import torch
# config.py

# General settings
DEBUG = False                       # Set to True to enable debugging features
SEED = 42                           # Random seed for reproducibility

# Data paths
DATA_DIR = "data/preprocessed/"      # Path to the dataset directory
TRAIN_DATA = DATA_DIR + "train.txt" # Path to training data
VALID_DATA = DATA_DIR + "val.txt"   # Path to validation data
TEST_DATA = DATA_DIR + "test.txt"   # Path to test data
SAVE_MODEL_PATH = "models/"         # Where to save models

# Text preprocessing
MAX_SEQ_LENGTH = 50                 # Maximum length of input/output sequences
VOCAB_SIZE = 10000                  # Maximum vocabulary size
PAD_TOKEN = 0                       # Padding token
SOS_TOKEN = 1                       # Start-of-sequence token
EOS_TOKEN = 2                       # End-of-sequence token
UNK_TOKEN = "<UNK>"                 # Unknown token

# Model architecture parameters
EMBEDDING_DIM = 256                 # Dimensionality of word embeddings
HIDDEN_DIM = 512                    # Dimensionality of RNN hidden state
NUM_LAYERS = 2                      # Number of layers in RNN
DROPOUT = 0.2                       # Dropout rate between layers
ENCODER_RNN_TYPE = "GRU"            # Type of RNN for encoder (LSTM or GRU)
DECODER_RNN_TYPE = "GRU"            # Type of RNN for decoder (LSTM or GRU)
BIDIRECTIONAL = True                # Use bidirectional encoder RNN

# Training parameters
BATCH_SIZE = 64                     # Number of samples per batch
EPOCHS = 20                         # Number of training epochs
LEARNING_RATE = 0.001               # Initial learning rate
TEACHER_FORCING_RATIO = 0.5         # Probability of using teacher forcing in decoder

# Hardware settings
USE_GPU = True                      # Use GPU if available
NUM_WORKERS = 4                     # Number of data loading workers
DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"

# Logging and checkpointing
CHECKPOINT_PATH = "checkpoints/"    # Path to save checkpoints
SAVE_EVERY = 5                      # Save model every X epochs
LOG_INTERVAL = 50                   # Print logs every X batches
