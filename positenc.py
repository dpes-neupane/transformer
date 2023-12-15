import torch
import numpy as np
from torch import nn, Tensor

class PositionalEncoding(object):
    def __init__(self, max_length, d_model, scalar=10000) -> None:
        self.pos = np.arange(max_length)[:, np.newaxis]
        self.d_model = d_model
        self.scalar = scalar
        self.pe = np.zeros((max_length, self.d_model))

    def encode(self, emb):
        division_term = np.exp(np.arange(0, self.d_model, 2) * (- np.log(10000.0) /self.d_model))
        self.pe[:, 0::2] = np.sin(self.pos * division_term)
        self.pe[:, 1::2] = np.cos(self.pos * division_term)
        return self.pe + emb

class PositionalEncodingTorch(nn.Module):
    '''
    Implementation of Positional Encoding according to the "Attention is all you need paper". 
    
    PE(pos, 2i) = sin(pos / 10000^(2i/dmodel))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/dmodel))
    
    Args:
        max_length : length of the sequence
        d_model: embedding dimension
        scalar: divisor in the denominator of the function
    
    Shape:
        - Input: (seq_length, emb_dimension)
        - Output: (seq_length, emb_dimension)
    
    
    '''
    def __init__(self, max_length: int, d_model: int, scalar: int = 10000) -> None:
        super().__init__()
        # self.pos = torch.arange(0, max_length)[:, None]
        self.d_model = d_model
        self.scalar = scalar
        # self.pe = torch.zeros((max_length, self.d_model))
        
    def forward(self, emb: Tensor) -> Tensor:
        self.pos = torch.arange(0, emb.shape[1])[:, None]
    
        self.pe = torch.zeros((emb.shape[1], self.d_model))
        divs = np.exp(torch.arange(0, self.d_model, 2) * (-np.log(10000.0) / self.d_model))
        self.pe[:, 0::2] = torch.sin(self.pos * divs)
        self.pe[:, 1::2] = torch.cos(self.pos * divs)
        # print("PEis\n")
        # print(self.pe.shape)
        return self.pe + emb
