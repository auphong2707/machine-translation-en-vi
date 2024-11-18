import torch
import torch.nn as nn

import sys
sys.path.append('../machine-translation-en-vi')
from config import *
from models.encoder import EncoderGRU
from models.decoder import DecoderGRU, DecoderAttnRNN

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
                 teacher_forcing_ratio=TEACHER_FORCING_RATIO,
                 max_seq_length = MAX_SEQ_LENGTH,
                 sos_token=SOS_TOKEN,
                 device = DEVICE):
        super(Seq2SeqGRU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.encoder_bidirectional = encoder_bidirectional
        
        self.device = device
        self.max_seq_length = max_seq_length

        self.encoder = EncoderGRU(input_size=input_size,
                                  embedding_size=embedding_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  dropout_rate=dropout_rate,
                                  bidirectional=encoder_bidirectional)
        
        decoder_hidden_size = hidden_size * 2 if encoder_bidirectional else hidden_size
        self.decoder = DecoderGRU(batch_size=batch_size,
                                  max_seq_length=max_seq_length,
                                  num_layers=num_layers,
                                  embedding_size=embedding_size,
                                  hidden_size=decoder_hidden_size,
                                  output_size=output_size,
                                  dropout_rate=dropout_rate,
                                  teacher_forcing_ratio=teacher_forcing_ratio,
                                  sos_token=sos_token,
                                  device=device)
        
        self.to(device)

    def forward(self, input, target=None):
        encoder_outputs, encoder_hidden = self.encoder(input)

        # For bidirectional GRU, we need to combine the last hidden states
        if self.encoder_bidirectional:
            # Concatenate the final forward and backward hidden states
            hidden_forward = encoder_hidden[0::2]
            hidden_backward = encoder_hidden[1::2]
            
            decoder_hidden = torch.cat((hidden_forward, hidden_backward), dim=-1)
        else:
            decoder_hidden = encoder_hidden  # Use the hidden state directly

        decoder_outputs, _, _ = self.decoder(encoder_outputs, decoder_hidden, target)
        return decoder_outputs
    
class Seq2SeqAttn(nn.Module):
    def __init__(self,
                 input_size=VOCAB_SIZE,
                 output_size=VOCAB_SIZE,
                 embedding_size=EMBEDDING_SIZE,
                 hidden_size=HIDDEN_SIZE,
                 batch_size=BATCH_SIZE,
                 dropout_rate=DROPOUT_RATE,
                 num_layers=NUM_LAYERS,
                 encoder_bidirectional=ENCODER_BIDIRECTIONAL,
                 teacher_forcing_ratio=TEACHER_FORCING_RATIO,
                 max_seq_length = MAX_SEQ_LENGTH,
                 device = DEVICE):
        
        super(Seq2SeqAttn, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.encoder_bidirectional = encoder_bidirectional
        
        self.device = device
        self.max_seq_length = max_seq_length

        self.encoder = EncoderGRU(input_size=input_size,
                                  embedding_size=embedding_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  dropout_rate=dropout_rate,
                                  bidirectional=encoder_bidirectional)
        
        decoder_hidden_size = hidden_size * 2 if encoder_bidirectional else hidden_size
        self.decoder = DecoderAttnRNN(batch_size=batch_size,
                                      max_seq_length=max_seq_length,
                                      num_layers=num_layers,
                                      embedding_size=embedding_size,
                                      hidden_size=decoder_hidden_size,
                                      output_size=output_size,
                                      dropout_rate=dropout_rate,
                                      teacher_forcing_ratio=teacher_forcing_ratio,
                                      sos_token=SOS_TOKEN,
                                      device=device)
        
        self.to(device)
        
    def forward(self, input, target=None, attention_return=False):
        encoder_outputs, encoder_hidden = self.encoder(input)
        
        # For bidirectional GRU, we need to combine the last hidden states
        if self.encoder_bidirectional:
            # Concatenate the final forward and backward hidden states
            hidden_forward = encoder_hidden[0::2]
            hidden_backward = encoder_hidden[1::2]
            
            decoder_hidden = torch.cat((hidden_forward, hidden_backward), dim=-1)
        else:
            decoder_hidden = encoder_hidden  # Use the hidden state directly
        
        decoder_outputs, _, attentions = self.decoder(encoder_outputs, decoder_hidden, target)
        return decoder_outputs
    
if __name__ == "__main__":
    # Example test for Seq2Seq class
    src = torch.randint(0, 10, (BATCH_SIZE, MAX_SEQ_LENGTH)).to(DEVICE)  # (BATCH_SIZE, MAX_SEQ_LENGTH)
    trg = torch.randint(0, 10, (BATCH_SIZE, MAX_SEQ_LENGTH)).to(DEVICE)  # (BATCH_SIZE, MAX_SEQ_LENGTH)

    model = Seq2SeqGRU().to(DEVICE)
    outputs = model(src, trg)

    print(outputs.shape)  # Expected shape: (BATCH_SIZE, MAX_SEQ_LENGTH, VOCAB_SIZE)
    
    model = Seq2SeqAttn().to(DEVICE)
    outputs = model(src, trg)

    print(outputs.shape)  # Expected shape: (BATCH_SIZE, MAX_SEQ_LENGTH, VOCAB_SIZE)