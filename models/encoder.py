import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout_p=0.1, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.birectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding layer
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # Multi-layer GRU
        self.gru = nn.GRU(hidden_size, 
                          hidden_size, 
                          num_layers=num_layers, 
                          batch_first=True,
                          dropout=dropout_p,
                          bidirectional=bidirectional)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        # Pass input through embedding layer
        embedded = self.embedding(input)
        
        # Apply dropout to the embeddings
        embedded = self.dropout(embedded)
        
        # Pass through GRU
        output, hidden = self.gru(embedded)
        
        if self.birectional:
             # Concatenate the hidden states from both directions for each layer
            # The hidden state will have shape: (num_layers * 2, batch_size, hidden_size)
            hidden = hidden.view(self.num_layers, self.num_directions, -1, self.hidden_size)
            hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)
        
        return output, hidden
    
if __name__ == "__main__":
    # Test Bidirectional EncoderRNN
    input_size = 10  # Vocabulary size
    hidden_size = 10  # Hidden size of GRU
    num_layers = 2    # Number of GRU layers
    bidirectional = True  # Enable bidirectionality

    encoder = EncoderRNN(input_size, hidden_size, num_layers, bidirectional=bidirectional)

    # Input: batch of 1 sequence, sequence length 5
    input_data = torch.LongTensor([[1, 2, 3, 4, 5]])

    # Forward pass
    output, hidden = encoder(input_data)

    # Output and hidden sizes
    print("Output size:", output.size())  # (batch_size, sequence_length, hidden_size * 2) -> (1, 5, 20)
    print("Hidden size:", hidden.size())  # (num_layers, batch_size, hidden_size * 2) -> (2, 1, 20)
