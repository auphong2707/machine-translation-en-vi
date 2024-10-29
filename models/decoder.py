import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('../machine-translation-en-vi')
from config import *

class DecoderGRU(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, dropout_rate, num_layers, teacher_forcing_ratio):
        super(DecoderGRU, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.teacher_forcing_ratio = teacher_forcing_ratio  # Store teacher forcing ratio
        
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        decoder_input = torch.empty(BATCH_SIZE, 1, dtype=torch.long, device=DEVICE).fill_(SOS_TOKEN)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_SEQ_LENGTH):
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
        output = self.embedding(input)
        output = F.relu(output)
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
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        
        return context, weights
    
class DecoderAttnRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(DecoderAttnRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(1)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, 
                                    device=DEVICE).fill_(SOS_TOKEN)
        decoder_input = encoder_hidden
        decoder_outputs = []
        attentions = []
        
        for i in range(MAX_SEQ_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, encoder_outputs, encoder_hidden
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)
            
            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(1).detach()    # detach from history as input
        
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(self.out(decoder_outputs), dim=-1)
        attentions = torch.cat(attentions, dim=1)
        
        return decoder_outputs, decoder_hidden, attentions
    
    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat([embedded, context], dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)
        
        return output, hidden, attn_weights
    
if __name__ == "__main__":

    decoder = DecoderGRU(EMBEDDING_SIZE, HIDDEN_SIZE, VOCAB_SIZE, DROPOUT_RATE, NUM_LAYERS * 2).to(DEVICE)
    
    encoder_outputs = torch.randn(MAX_SEQ_LENGTH, BATCH_SIZE, HIDDEN_SIZE * 2).to(DEVICE)
    encoder_hidden = torch.randn(NUM_LAYERS * 2, BATCH_SIZE, HIDDEN_SIZE).to(DEVICE)
    target_tensor = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LENGTH)).to(DEVICE)

    decoder_outputs, decoder_hidden, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

    assert decoder_outputs.size() == (BATCH_SIZE, MAX_SEQ_LENGTH, VOCAB_SIZE)
    assert decoder_hidden.size() == (NUM_LAYERS * 2, BATCH_SIZE, HIDDEN_SIZE)
    print("DecoderRNN test passed.")