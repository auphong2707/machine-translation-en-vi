import torch
import torch.nn as nn

class EncoderGRU(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, 
                 num_layers, dropout_rate, bidirectional):
        super(EncoderGRU, self).__init__()

        # [SAVE PARAMETERS]
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # [CREATE LAYERS]
        # Embedding layer
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        # Multi-layer GRU
        self.gru = nn.GRU(embedding_size, 
                          hidden_size, 
                          num_layers=num_layers,
                          dropout=dropout_rate,
                          bidirectional=bidirectional,
                          batch_first=True)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input, hidden_state=None):
        # Input: (BATCH_SIZE, LENGTH)
        
        output = self.dropout(self.embedding(input))
        # Output: (BATCH_SIZE, LENGTH, EMBEDDING_SIZE)
        
        output, hidden = self.gru(output, hidden_state)
        # Output: (BATCH_SIZE, LENGTH, HIDDEN_SIZE * NUM_DIRECTIONS)
        # Hidden: (NUM_LAYERS * NUM_DIRECTIONS, BATCH_SIZE, HIDDEN_SIZE)
        
        return output, hidden


# [TESTING SECTION START]
if __name__ == "__main__":
    import sys
    sys.path.append('../machine-translation-en-vi')
    from config import *
    
    encoder = EncoderGRU(
        input_size=VOCAB_SIZE,
        embedding_size=RNN_EMBEDDING_SIZE,
        hidden_size=RNN_HIDDEN_SIZE,
        num_layers=RNN_NUM_LAYERS,
        dropout_rate=RNN_DROPOUT_RATE,
        bidirectional=RNN_ENCODER_BIDIRECTIONAL,
    )

    # Input: batch of 1 sequence, sequence length 5
    input_data = torch.randint(0, 10000, (BATCH_SIZE, MAX_SEQ_LENGTH))

    # Forward pass
    output, hidden = encoder(input_data)

    # Output and hidden sizes
    print("Output size:", output.size())  # Output: (BATCH_SIZE, LENGTH, HIDDEN_SIZE * NUM_DIRECTIONS)
    print("Hidden size:", hidden.size())  # Hidden: (NUM_LAYERS * NUM_DIRECTIONS, BATCH_SIZE, HIDDEN_SIZE)

# [TESTING SECTION END]