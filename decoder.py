from torch import nn, Tensor
from typing import Any, Union
from multihead import DecoderMultihead, Multihead
class Decoder(nn.Module):
    ''' 
    Class for the decoder layer of the Transformer.
    
    Args:
        input_dim: size of the input dimension 
        emb_dim: dimension of the embeddings inside the transformer
        heads: no of heads for multihead selfAttention calculation
        linear_dim: size of the FCNN above each multihead sub-layer
        dropout: dropout probability.

    Shape:
        - Input: (seq_length, input_dim) and the output of the encoder (seq_length, emb_dim).
        - Output: (seq_length, emb_dim).
    '''
    def __init__(self, input_dim: int, emb_dim: int, heads: int, linear_dim: int, dropout: float=0.0) -> None:
        super().__init__()
        self.multihead= DecoderMultihead(input_dim, emb_dim, heads)
        self.maskedMultihead= Multihead(input_dim, emb_dim, heads)
        self.linear_layer = nn.Sequential(nn.Linear(input_dim, linear_dim),
                                          nn.Dropout(dropout),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(linear_dim, input_dim))
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.ln3 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, y: Tensor, mask1:Union[Tensor, None]=None, mask2:Union[Tensor, None]=None, return_att: bool=False) -> Tensor: 
        maskedselfatt = self.maskedMultihead(y, mask=mask2, ret_att=return_att)
        y = y + self.dropout(maskedselfatt)
        y = self.ln1(y)
        selfatt = self.multihead(x, y, mask=mask1, ret_att=return_att)
        y = y + self.dropout(selfatt)
        y = self.ln2(y)
        linear = self.linear_layer(y)
        x = x + self.dropout(linear)
        x = self.ln3(x)
        return x

class Decoders(nn.Module):
    '''
    Class for n Decoders 
    
    Args: 
        layers: number of encoders
        decoder_args: all the encoder arguments
    
    Shape:
        - Input: (seq_length, input_dim) and the output of the encoder (seq_length, emb_dim).
        - Output: (seq_length, emb_dim). 
    '''
    def __init__(self, layers: int, **decoder_args: Any) -> None:
        super().__init__()
        self.layers = layers
        self.decoders = nn.ModuleList([Decoder(**decoder_args) for _ in range(layers)])

    def forward(self, x: Tensor, y: Tensor, mask1:Union[Tensor, None]=None, mask2:Union[Tensor, None]=None, return_att: bool=False) -> Tensor:
        for l in self.decoders:
            y = l(x, y, mask1, mask2)
        return y