from torch import nn, Tensor
from typing import Union
from .encoder import Encoders
from .decoder import Decoders


class Transformer(nn.Module):
    '''
    Class for Transformer. 
    
    Args:
        layers: no of encoder and decoder layers
        input_dim: the dimension of the input embeddings, here it is equal to the emb_dim
        emb_dim: the dimension of the transformer embeddings
        heads: no of heads for multihead attention
        linear_dim: dimension of the FCNN layer
        dropout: dropout probability.

    Shape:
        - Input: (seq_length, input_dim), (seq_length, input_dim)
        - Output: (seq_length, emb_dim)

    '''
    def __init__(self, layers: int, input_dim: int, emb_dim: int, heads: int, linear_dim: int,  dropout: float=0.0) -> None:
        super().__init__()
        self.encoder = Encoders(layers, input_dim=input_dim, emb_dim=emb_dim, heads=heads, linear_dim=linear_dim, dropout=dropout)
        self.decoder = Decoders(layers, input_dim=input_dim, emb_dim=emb_dim, heads=heads, linear_dim=linear_dim, dropout=dropout)
        
    def forward(self, x: Tensor, y: Tensor, mask1: Union[Tensor, None]=None, mask2: Union[Tensor, None]=None, return_att: bool=False) -> Tensor:
        softmax, x = self.encoder.forward(x, mask1, return_att=return_att)
        z = self.decoder.forward(x, y, mask2=mask2, return_att=return_att)
        if return_att:
            return softmax, z
        return None, z
