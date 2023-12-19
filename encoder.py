from torch import nn, Tensor
from typing import Any, Union
from multihead import Multihead

class Encoder(nn.Module):
    '''
    Class for the encoder layer of the Transformer.
    
    Args:
        input_dim: size of the input dimension 
        emb_dim: dimension of the embeddings inside the transformer
        heads: no of heads for multihead selfAttention calculation
        linear_dim: size of the FCNN above each multihead sub-layer
        dropout: dropout probability.
    
    Shape:
        - Input: (seq_length, input_dim)
        - Output: (seq_length, emb_dim)
        
    '''
    def __init__(self, input_dim: int, emb_dim: int, heads: int, linear_dim: int, dropout: float=0.0) -> None:
        super().__init__()
        self.mulatt = Multihead(input_dim, emb_dim, heads)
        self.linear_layer = nn.Sequential(nn.Linear(input_dim, linear_dim),
                                          nn.Dropout(dropout),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(linear_dim, input_dim))
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:Tensor, mask:Union[Tensor, None]=None, return_att: bool=False)  -> Tensor:
        selfatt = self.mulatt(x, mask=mask, ret_att=return_att)
        x = x + self.dropout(selfatt)
        x = self.ln1(x)
        linout = self.linear_layer(x)
        x = x + self.dropout(linout)
        x = self.ln2(x)
        return x

class Encoders(nn.Module):
    '''
    Class for n Encoders 
    
    Args: 
        layers: number of encoders
        encoder_args: all the encoder arguments
    
    Shape:
        - Input: (seq_length, input_dim)
        - Output: (seq_length, emb_dim)
    '''
    def __init__(self, layers: int, **encoder_args: Any) -> None:
        super().__init__()
        self.layers = layers
        self.encoders = nn.ModuleList([Encoder(**encoder_args) for _ in range(layers)])

    def forward(self, x: Tensor, mask:Union[None, Tensor]=None, return_att: bool=False) -> Tensor:
        for l in self.encoders:
            x = l(x, mask)
        return x
