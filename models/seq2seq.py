import torch
import torch.nn as nn

import sys
sys.path.append('../machine-translation-en-vi')
from config import *
from models.encoder import EncoderGRU
from models.decoder import DecoderGRU, DecoderAttnRNN

class Seq2SeqRNN(nn.Module):
    def __init__(self,
                 batch_size=BATCH_SIZE,
                 max_seq_length = MAX_SEQ_LENGTH,
                 num_layers=RNN_NUM_LAYERS,
                 input_size=VOCAB_SIZE,
                 output_size=VOCAB_SIZE,
                 embedding_size=RNN_EMBEDDING_SIZE,
                 hidden_size=RNN_HIDDEN_SIZE,
                 dropout_rate=RNN_DROPOUT_RATE,
                 encoder_bidirectional=RNN_ENCODER_BIDIRECTIONAL,
                 teacher_forcing_ratio=TEACHER_FORCING_RATIO,
                 sos_token=SOS_TOKEN,
                 device = DEVICE):
        super(Seq2SeqRNN, self).__init__()
        
        # [SAVE PARAMETERS]
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        
        self.dropout_rate = dropout_rate
        self.encoder_bidirectional = encoder_bidirectional
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.sos_token = sos_token
        
        self.device = device

        # [CREATE LAYERS]
        # Encoder
        self.encoder = EncoderGRU(input_size=input_size,
                                  embedding_size=embedding_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  dropout_rate=dropout_rate,
                                  bidirectional=encoder_bidirectional)
        
        # Decoder
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
        # Input: (BATCH_SIZE, MAX_SEQ_LENGTH)
        
        # Encode the input sequence
        encoder_outputs, encoder_hidden = self.encoder(input)

        # For bidirectional GRU, we need to combine the last hidden states
        if self.encoder_bidirectional:
            # Concatenate the final forward and backward hidden states
            hidden_forward = encoder_hidden[0::2]
            hidden_backward = encoder_hidden[1::2]
            
            decoder_hidden = torch.cat((hidden_forward, hidden_backward), dim=-1)
        else:
            decoder_hidden = encoder_hidden  # Use the hidden state directly

        # Decode the output sequence
        decoder_outputs, _, _ = self.decoder(encoder_outputs, decoder_hidden, target)
        
        # Output: (BATCH_SIZE, MAX_SEQ_LENGTH, VOCAB_SIZE)
        return decoder_outputs
    
class Seq2SeqRNNAttn(nn.Module):
    def __init__(self,
                 batch_size=BATCH_SIZE,
                 max_seq_length = MAX_SEQ_LENGTH,
                 num_layers=RNN_ATTN_NUM_LAYERS,
                 input_size=VOCAB_SIZE,
                 output_size=VOCAB_SIZE,
                 embedding_size=RNN_ATTN_EMBEDDING_SIZE,
                 hidden_size=RNN_ATTN_HIDDEN_SIZE,
                 dropout_rate=RNN_ATTN_DROPOUT_RATE,
                 encoder_bidirectional=RNN_ATTN_ENCODER_BIDIRECTIONAL,
                 teacher_forcing_ratio=TEACHER_FORCING_RATIO,
                 sos_token=SOS_TOKEN,
                 device = DEVICE):
        
        super(Seq2SeqRNNAttn, self).__init__()
        
        # [SAVE PARAMETERS]
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        
        self.dropout_rate = dropout_rate
        self.encoder_bidirectional = encoder_bidirectional
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.sos_token = sos_token
        
        self.device = device

        # [CREATE LAYERS]
        # Encoder
        self.encoder = EncoderGRU(input_size=input_size,
                                  embedding_size=embedding_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  dropout_rate=dropout_rate,
                                  bidirectional=encoder_bidirectional)
        
        # Decoder
        decoder_hidden_size = hidden_size * 2 if encoder_bidirectional else hidden_size
        self.decoder = DecoderAttnRNN(batch_size=batch_size,
                                      max_seq_length=max_seq_length,
                                      num_layers=num_layers,
                                      embedding_size=embedding_size,
                                      hidden_size=decoder_hidden_size,
                                      output_size=output_size,
                                      dropout_rate=dropout_rate,
                                      teacher_forcing_ratio=teacher_forcing_ratio,
                                      sos_token=self.sos_token,
                                      device=device)
        
        self.to(device)
        
    def forward(self, input, target=None, attention_return=False):
        # Input: (BATCH_SIZE, MAX_SEQ_LENGTH)
        
        # Encode the input sequence
        encoder_outputs, encoder_hidden = self.encoder(input)
        
        # For bidirectional GRU, we need to combine the last hidden states
        if self.encoder_bidirectional:
            # Concatenate the final forward and backward hidden states
            hidden_forward = encoder_hidden[0::2]
            hidden_backward = encoder_hidden[1::2]
            
            decoder_hidden = torch.cat((hidden_forward, hidden_backward), dim=-1)
        else:
            decoder_hidden = encoder_hidden  # Use the hidden state directly
        
        # Decode the output sequence
        decoder_outputs, _, attentions = self.decoder(encoder_outputs, decoder_hidden, target)
        
        # Output: (BATCH_SIZE, MAX_SEQ_LENGTH, VOCAB_SIZE)
        return decoder_outputs
    
    
# [TESTING SECTION START]
if __name__ == "__main__":
    # Example test for Seq2Seq class
    src = torch.randint(0, 10, (BATCH_SIZE, MAX_SEQ_LENGTH)).to(DEVICE)  # (BATCH_SIZE, MAX_SEQ_LENGTH)
    trg = torch.randint(0, 10, (BATCH_SIZE, MAX_SEQ_LENGTH)).to(DEVICE)  # (BATCH_SIZE, MAX_SEQ_LENGTH)

    model = Seq2SeqRNN().to(DEVICE)
    outputs = model(src, trg)

    print(outputs.shape)  # Expected shape: (BATCH_SIZE, MAX_SEQ_LENGTH, VOCAB_SIZE)
    
    model = Seq2SeqRNNAttn().to(DEVICE)
    outputs = model(src, trg)

    print(outputs.shape)  # Expected shape: (BATCH_SIZE, MAX_SEQ_LENGTH, VOCAB_SIZE)
# [TESTING SECTION END]