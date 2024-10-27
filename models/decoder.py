import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('../machine-translation-en-vi')
from config import *

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(output_size, hidden_size).to(DEVICE)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True).to(DEVICE)
        self.out = nn.Linear(self.hidden_size, output_size).to(DEVICE)
        
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=DEVICE).fill_(SOS_TOKEN)

        decoder_hidden = encoder_hidden[:self.num_layers]
            
        decoder_outputs = []
        
        for i in range(MAX_SEQ_LENGTH):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)
            
            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()
                
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None 
    
    def forward_step(self, input, hidden):
        output = self.embedding(input).to(DEVICE)
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
        batch_size = encoder_outputs.size(0)
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
    hidden_size = 256
    output_size = 10
    num_layers = 2

    decoder = DecoderRNN(hidden_size, output_size, num_layers)

    encoder_outputs = torch.randn(5, 10, hidden_size * 2).to(DEVICE)  # batch_size x seq_len x hidden_size*2
    encoder_hidden = torch.randn(num_layers * 2, 5, hidden_size).to(DEVICE)  # num_layers*2 x batch_size x hidden_size

    decoder_outputs, decoder_hidden, _ = decoder(encoder_outputs, encoder_hidden)

    print("Decoder outputs shape:", decoder_outputs.shape)
    print("Decoder hidden shape:", decoder_hidden.shape)