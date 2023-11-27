import torch
from torch import nn
from multihead import DecoderMultihead, Multihead
class Decoder(nn.Module):
    def __init__(self, input_dim, emb_dim, heads, linear_dim, dropout=0.0):
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

    def forward(self, x, y, mask1=None, mask2=None): #mask needs to be added!!!
        maskedselfatt = self.maskedMultihead(y, mask=mask1)
        y = y + self.dropout(maskedselfatt)
        y = self.ln1(y)
        y = y.detach()
        z = torch.hstack((y, x, x))
        selfatt = self.multihead(z, mask2) #mask!1
        y = y + self.dropout(selfatt)
        y = self.ln2(y)
        linear = self.linear_layer(y)
        x = x + self.dropout(linear)
        x = self.ln3(x)
        return x

class Decoders(nn.Module):
    def __init__(self, layers, **decoder_args):
        super().__init__()
        self.layers = layers
        self.decoders = nn.ModuleList([Decoder(**decoder_args) for _ in range(layers)])

    def forward(self, x, y, mask1=None, mask2=None):
        for l in self.decoders:
            x = l(x, y, mask1, mask2)
        return x