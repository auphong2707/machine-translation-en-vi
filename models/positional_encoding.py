import torch
import torch.nn as nn
from torch import Tensor
import math
import seaborn as sns


class PositionalEncoding(nn.Module):
    def __init__(self,
                 embedding_size: int,
                 dropout_rate: float = 0.1,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        den = torch.exp(- torch.arange(0, embedding_size, 2)* math.log(10000) / embedding_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, embedding_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)
        
        self.register_buffer('pos_embedding', pos_embedding)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])
    
if __name__ == "__main__":
    # Plot positional encoding
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(5, 15))
    # pe = PositionalEncoding(embedding_size=20, dropout_rate=0, maxlen=70)
    # y = pe.pos_embedding[0, :, :].numpy()
    # sns.heatmap(y, cmap='Blues')
    # plt.xlabel('Embedding Dimensions')
    # plt.ylabel('Position')
    # plt.title('Positional Encoding Heatmap')
    # plt.savefig('positional_encoding_heatmap.png')
    
    # Test positional encoding
    pos_encoder = PositionalEncoding(embedding_size=512, dropout_rate=0.1, maxlen=1000)
    sample_embedding = torch.randn(32, 100, 512)  # (batch_size=32, seq_len=100, embedding_size=512)
    output = pos_encoder(sample_embedding)  # Output will also have shape (32, 100, 512)
    print(output.shape)
