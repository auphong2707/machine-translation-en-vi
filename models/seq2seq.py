import torch
import torch.nn as nn

import sys
sys.path.append('../machine-translation-en-vi')
from config import *
from models.encoder import EncoderGRU
from models.decoder import DecoderGRU

class Seq2SeqGRU(nn.Module):
    def __init__(self,
                 input_size=VOCAB_SIZE,
                 output_size=VOCAB_SIZE,
                 embedding_size=EMBEDDING_SIZE,
                 hidden_size=HIDDEN_SIZE,
                 batch_size=BATCH_SIZE,
                 dropout_rate=DROPOUT_RATE,
                 num_layers=NUM_LAYERS,
                 encoder_bidirectional=ENCODER_BIDIRECTIONAL,
                 teacher_forcing_ratio=TEACHER_FORCING_RATIO):
        
        super(Seq2SeqGRU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.encoder_bidirectional = encoder_bidirectional

        self.encoder = EncoderGRU(input_size, embedding_size, hidden_size, num_layers, dropout_rate, encoder_bidirectional)
        
        decoder_hidden_size = hidden_size * 2 if encoder_bidirectional else hidden_size
        self.decoder = DecoderGRU(embedding_size, decoder_hidden_size, output_size, dropout_rate, num_layers, teacher_forcing_ratio)

    def forward(self, input, target=None):
        encoder_outputs, encoder_hidden = self.encoder(input)

        # For bidirectional GRU, we need to combine the last hidden states
        if self.encoder_bidirectional:
            # Get the last two hidden states: forward and backward
            #hidden_forward = encoder_hidden[-2]  # Last hidden state from the forward layer
            #hidden_backward = encoder_hidden[-1]  # Last hidden state from the backward layer
            # Take all the odd layer
            hidden_forward = encoder_hidden[0::2]
            hidden_backward = encoder_hidden[1::2]
            
            decoder_hidden = torch.cat((hidden_forward, hidden_backward), dim=-1)
        else:
            decoder_hidden = encoder_hidden  # Use the hidden state directly

        decoder_outputs, _, _ = self.decoder(encoder_outputs, decoder_hidden, target)
        return decoder_outputs

    
if __name__ == "__main__":
    # Example test for Seq2Seq class
    src = torch.randint(0, 10, (BATCH_SIZE, MAX_SEQ_LENGTH)).to(DEVICE)  # (BATCH_SIZE, MAX_SEQ_LENGTH)
    trg = torch.randint(0, 10, (BATCH_SIZE, MAX_SEQ_LENGTH)).to(DEVICE)  # (BATCH_SIZE, MAX_SEQ_LENGTH)

    model = Seq2SeqGRU().to(DEVICE)
    outputs = model(src, trg)

    print(outputs.shape)  # Expected shape: (SEQ_LENGTH, BATCH_SIZE, VOCAB_SIZE)