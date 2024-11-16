import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('../machine-translation-en-vi')
from config import *

class DecoderGRU(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, dropout_rate, num_layers, teacher_forcing_ratio,
                 batch_size, max_seq_length, device, sos_token=SOS_TOKEN):
        super(DecoderGRU, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.teacher_forcing_ratio = teacher_forcing_ratio  # Store teacher forcing ratio
        
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.device = device
        
        self.sos_token = sos_token

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        decoder_input = torch.empty(self.batch_size, 1, dtype=torch.long, device=self.device).fill_(self.sos_token)
        decoder_hidden = encoder_hidden

        decoder_outputs = []
        
        for i in range(self.max_seq_length):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            # Decide whether to use teacher forcing
            if target_tensor is not None and random.random() < self.teacher_forcing_ratio:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Use the actual target
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None  # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.dropout(self.embedding(input))
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)
        
    def forward(self, query, keys):
        # keys = keys.permute(1, 0, 2)
        
        # query : (B, number of direction * number of layers, D)
        # keys : (B, Seq, D)
        query = query[:, :1, :].repeat(1, keys.size(1), 1)
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        
        return context, weights
    
class DecoderAttnRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, dropout_rate, num_layers, teacher_forcing_ratio,
                 batch_size, max_seq_length, device, sos_token=SOS_TOKEN):
        super(DecoderAttnRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.teacher_forcing_ratio = teacher_forcing_ratio  # Store teacher forcing ratio
        
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2*embedding_size, hidden_size, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.device = device
        
        self.sos_token = sos_token
        
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        decoder_input = torch.empty(self.batch_size, 1, dtype=torch.long, 
                                    device=self.device).fill_(self.sos_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []
        
        for i in range(self.max_seq_length):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)
            
            if target_tensor is not None and random.random() < self.teacher_forcing_ratio:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()    # detach from history as input
        
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(self.out(decoder_outputs), dim=-1)
        attentions = torch.cat(attentions, dim=1)
        
        return decoder_outputs, decoder_hidden, attentions
    
    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        print(f"Context shape: {context.shape}")
        print(f"Embedded shape: {embedded.shape}")
        input_gru = torch.cat([embedded, context], dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)
        
        return output, hidden, attn_weights
    
if __name__ == "__main__":

    # Test DecoderAttnRNN
    num_layers = 4
    embedding_size = 256
    hidden_size = 256
    output_size = 100
    dropout_rate = 0.1
    teacher_forcing_ratio = 0.5
    batch_size = 32
    max_seq_length = 10
    
    decoder_attn = DecoderAttnRNN(embedding_size, hidden_size, output_size, dropout_rate, num_layers,
                                  teacher_forcing_ratio, batch_size, max_seq_length, DEVICE)
    encoder_outputs = torch.randn(32, 10, 256)
    encoder_hidden = torch.randn(num_layers, 32, 256)
    target_tensor = torch.randint(0, 100, (batch_size, max_seq_length), dtype=torch.long)
    
    decoder_outputs, decoder_hidden, attentions = decoder_attn(encoder_outputs, encoder_hidden, target_tensor)
    print("Decoder outputs shape:", decoder_outputs.shape)
    print("Decoder hidden shape:", decoder_hidden.shape)
   
    print("Attention weights shape:", attentions.shape)
    print("Attention weights sum:", attentions.sum(dim=-1))
    
    # Test BahdanauAttention
    # hidden_size = 256
    # batch_size = 2
    # seq_len = 10

    # attention = BahdanauAttention(hidden_size)
    # query = torch.randn(batch_size, seq_len, hidden_size)
    # keys = torch.randn(seq_len, batch_size, hidden_size)

    # context, weights = attention(query, keys)

    # print("Query shape:", query.shape)
    # print("Keys shape:", keys.shape)
    # print("Context shape:", context.shape)
    # print("Attention weights shape:", weights.shape)
     