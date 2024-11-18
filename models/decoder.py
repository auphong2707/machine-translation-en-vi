import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderGRU(nn.Module):
    def __init__(self, batch_size, max_seq_length, num_layers,
                 embedding_size, hidden_size, output_size, 
                 dropout_rate, teacher_forcing_ratio, sos_token,
                 device):
        super(DecoderGRU, self).__init__()
        
        # [SAVE PARAMETERS]
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.dropout_rate = dropout_rate
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.sos_token = sos_token
        
        self.device = device
        
        # [CREATE LAYERS]
        # Embedding layer
        self.embedding = nn.Embedding(output_size, embedding_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # GRU layer
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
       
        # Output layer
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        # Initialize the decoder input with SOS_token
        decoder_input = torch.empty(self.batch_size, 1, dtype=torch.long, device=self.device).fill_(self.sos_token)
        
        # Use the hidden state of the encoder as the initial hidden state of the decoder
        decoder_hidden = encoder_hidden
        
        # Store decoder outputs
        decoder_outputs = []
        
        # Iterate over the maximum sequence length
        for i in range(self.max_seq_length):
            # Forward step
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

        # Concatenate the decoder outputs
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        
        # Apply log softmax to the outputs
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        
        return decoder_outputs, decoder_hidden, None

    def forward_step(self, input, hidden):
        # Input: (BATCH_SIZE, 1)
        
        output = self.dropout(self.embedding(input))
        # Output: (BATCH_SIZE, 1, EMBEDDING_SIZE)
        
        output, hidden = self.gru(output, hidden)
        # Output: (BATCH_SIZE, 1, HIDDEN_SIZE)
        
        output = self.out(output)
        # Output: (BATCH_SIZE, 1, OUTPUT_SIZE)
        
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        
        # [CREATE LAYERS]
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)
        
    def forward(self, query, keys):
        # Query: (BATCH_SIZE, NUM_LAYERS * NUM_DIRECTIONS, HIDDEN_SIZE)
        # Keys: (BATCH_SIZE, LENGTH, HIDDEN_SIZE)
        
        query = query[:, :1, :].repeat(1, keys.size(1), 1)
        # Query: (BATCH_SIZE, LENGTH, HIDDEN_SIZE)
        
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        # Scores: (BATCH_SIZE, 1, LENGTH)
        
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        # Weights: (BATCH_SIZE, 1, LENGTH)
        # Context: (BATCH_SIZE, 1, HIDDEN_SIZE)
        
        return context, weights
    
class DecoderAttnRNN(nn.Module):
    def __init__(self, batch_size, max_seq_length, num_layers,
                 embedding_size, hidden_size, output_size, 
                 dropout_rate, teacher_forcing_ratio, sos_token,
                 device):
        super(DecoderAttnRNN, self).__init__()
        
        # [SAVE PARAMETERS]
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.dropout_rate = dropout_rate
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.sos_token = sos_token
        
        self.device = device
        
        # [CREATE LAYERS]
        # Embedding layer
        self.embedding = nn.Embedding(output_size, embedding_size)
        
        # Attention layer
        self.attention = BahdanauAttention(hidden_size)
        
        # GRU layer
        self.gru = nn.GRU(embedding_size + hidden_size, hidden_size, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        
        # Output layer
        self.out = nn.Linear(hidden_size, output_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        # Initialize the decoder input with SOS_token
        decoder_input = torch.empty(self.batch_size, 1, dtype=torch.long, device=self.device).fill_(self.sos_token)
        
        # Use the hidden state of the encoder as the initial hidden state of the decoder
        decoder_hidden = encoder_hidden
        
        # Store decoder outputs and attention weights
        decoder_outputs = []
        attentions = []
        
        # Iterate over the maximum sequence length
        for i in range(self.max_seq_length):
            # Forward step
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
        
        # Concatenate the decoder outputs
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        
        # Apply log softmax to the outputs
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        
        # Concatenate the attention weights
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions
    
    def forward_step(self, input, hidden, encoder_outputs):
        # Input: (BATCH_SIZE, 1)
        
        embedded = self.dropout(self.embedding(input))
        # Embedded: (BATCH_SIZE, 1, EMBEDDING_SIZE)
        
        query = hidden.permute(1, 0, 2)
        # Query: (BATCH_SIZE, NUM_LAYERS, HIDDEN_SIZE)
        
        context, attn_weights = self.attention(query, encoder_outputs)
        # Context: (BATCH_SIZE, 1, HIDDEN_SIZE)
        
        input_gru = torch.cat([embedded, context], dim=2)
        # Input GRU: (BATCH_SIZE, 1, EMBEDDING_SIZE + HIDDEN_SIZE)

        output, hidden = self.gru(input_gru, hidden)
        # Output: (BATCH_SIZE, 1, HIDDEN_SIZE)
        
        output = self.out(output)
        # Output: (BATCH_SIZE, 1, OUTPUT_SIZE)
        
        return output, hidden, attn_weights
    

# [TESTING SECTION START]
if __name__ == "__main__":
    import sys
    sys.path.append('../machine-translation-en-vi')
    from config import *

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

# [TESTING SECTION END]