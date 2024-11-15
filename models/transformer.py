import math
import torch.nn as nn
from torch import Tensor

import sys
sys.path.append('../machine-translation-en-vi')
from config import *
from models.positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self,
                 input_size=VOCAB_SIZE,
                 output_size=VOCAB_SIZE,
                 embedding_size=TFM_EMBEDDING_SIZE,
                 batch_size=BATCH_SIZE,
                 dropout_rate=TFM_DROPOUT_RATE,
                 layers=TFM_NUM_LAYERS,
                 heads=TFM_NUM_HEADS,
                 feed_forward=TFM_DIM_FEED_FORWARD,
                 device=DEVICE):
        super(Transformer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.layers = layers
        self.heads = heads
        self.feed_forward = feed_forward
        self.device = device
        
        self.input_embedding = nn.Embedding(input_size, embedding_size)
        self.target_embedding = nn.Embedding(output_size, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size, dropout_rate)
        
        self.transformer = nn.Transformer(d_model=embedding_size,
                                          nhead=heads,
                                          num_encoder_layers=layers,
                                          num_decoder_layers=layers,
                                          dim_feedforward=feed_forward,
                                          dropout=dropout_rate,
                                          batch_first=True)
        
        self.linear = nn.Linear(embedding_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
        self.to(self.device)
    
    def create_masks(self, src, tgt):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]
        
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len, device=self.device).type(torch.bool)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(torch.bool)
        
        src_padding_mask = (src == PAD_TOKEN).type(torch.bool)
        tgt_padding_mask = (tgt == PAD_TOKEN).type(torch.bool)
        
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
        
    def forward(self, input: Tensor,
                target: Tensor,
                input_mask: Tensor = None,
                target_mask: Tensor = None,
                input_padding_mask: Tensor = None,
                target_padding_mask: Tensor = None,
                memory_key_padding_mask: Tensor = None):
        
        input_embedding = self.positional_encoding(self.input_embedding(input) * math.sqrt(self.embedding_size))
        target_embedding = self.positional_encoding(self.target_embedding(target) * math.sqrt(self.embedding_size))
        outputs = self.transformer(input_embedding,
                                   target_embedding,
                                   input_mask,
                                   target_mask,
                                   None,
                                   input_padding_mask,
                                   target_padding_mask,
                                   memory_key_padding_mask)

        outputs = self.log_softmax(self.linear(outputs))
        
        return outputs
    
    def translate(self, input):
        self.eval()
        
        y_input = torch.full((self.batch_size, 1), SOS_TOKEN, device=self.device)
        for _ in range(MAX_SEQ_LENGTH):
            # Create mask
            input_mask, target_mask, input_padding_mask, target_padding_mask = self.create_masks(input, y_input)
            
            # Forward pass
            output = self(input,
                          target=y_input,
                          input_mask=input_mask, 
                          target_mask=target_mask, 
                          input_padding_mask=input_padding_mask, 
                          target_padding_mask=target_padding_mask)
            
            # Take the highest prediction of each batch
            output = output[:, -1].argmax(dim=1)
            
            # Add the last output to y_input
            y_input = torch.cat([y_input, output.unsqueeze(1)], dim=1)
            
        return y_input

if __name__ == '__main__':
    # Create a random input tensor
    input_tensor = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LENGTH)).to(DEVICE)
    target_tensor = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LENGTH)).to(DEVICE)

    # Initialize the Transformer model
    model = Transformer()

    # Set the model to training mode
    model.train()

    # Forward pass
    output = model(input_tensor, target_tensor)

    # Print the output shape
    print("Output shape:", output.shape)

    # Test the translate function
    translated_output = model.translate(input_tensor)
    print("Translated output shape:", translated_output.shape)
    
    # print(model.transformer.generate_square_subsequent_mask(10, device=DEVICE).type(torch.bool))