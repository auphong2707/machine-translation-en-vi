import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import EncoderRNN
from decoder import DecoderRNN

import sys
sys.path.append('../machine-translation-en-vi')
from config import *
from utils import teacher_forcing


import torch.nn as nn
import torch

import torch
import torch.nn as nn

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
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
        
        super(Seq2Seq, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.encoder_bidirectional = encoder_bidirectional

        self.encoder = EncoderRNN(input_size, embedding_size, hidden_size, num_layers, dropout_rate, encoder_bidirectional)
        
        # Adjust number of layers for the decoder if encoder is bidirectional
        decoder_num_layers = 2 if encoder_bidirectional else 1
        self.decoder = DecoderRNN(embedding_size, hidden_size, output_size, dropout_rate, decoder_num_layers, teacher_forcing_ratio)

    def forward(self, input, target=None):
        encoder_outputs, encoder_hidden = self.encoder(input)

        # For bidirectional GRU, we need to combine the last hidden states
        if self.encoder_bidirectional:
            # Get the last two hidden states: forward and backward
            hidden_forward = encoder_hidden[-2]  # Last hidden state from the forward layer
            hidden_backward = encoder_hidden[-1]  # Last hidden state from the backward layer
            
            # Combine them to create the decoder's hidden state
            decoder_hidden = torch.cat((hidden_forward.unsqueeze(0), hidden_backward.unsqueeze(0)), dim=0)  # Shape: (NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE)
        else:
            decoder_hidden = encoder_hidden  # Use the hidden state directly

        decoder_outputs, _, _ = self.decoder(encoder_outputs, decoder_hidden, target)
        return decoder_outputs

    
if __name__ == "__main__":
    # Example test for Seq2Seq class
    src = torch.randint(0, 10, (BATCH_SIZE, MAX_SEQ_LENGTH)).to(DEVICE)  # (BATCH_SIZE, MAX_SEQ_LENGTH)
    trg = torch.randint(0, 10, (BATCH_SIZE, MAX_SEQ_LENGTH)).to(DEVICE)  # (BATCH_SIZE, MAX_SEQ_LENGTH)

    model = Seq2Seq().to(DEVICE)
    outputs = model(src, trg)

    print(outputs.shape)  # Expected shape: (sequence_length, batch_size, trg_vocab_size)