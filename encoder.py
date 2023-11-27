from torch import nn

from multihead import Multihead


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, heads, linear_dim, dropout=0.0) :
        super().__init__()
        self.mulatt = Multihead(input_dim, emb_dim, heads)
        self.linear_layer = nn.Sequential(nn.Linear(input_dim, linear_dim),
                                          nn.Dropout(dropout),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(linear_dim, input_dim))
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        selfatt = self.mulatt(x, mask=mask, ret_att=False)
        x = x + self.dropout(selfatt)
        x = self.ln1(x)
        linout = self.linear_layer(x)
        x = x + self.dropout(linout)
        x = self.ln2(x)
        return x

class Encoders(nn.Module):
    def __init__(self, layers, **encoder_args):
        super().__init__()
        self.layers = layers
        self.encoders = nn.ModuleList([Encoder(**encoder_args) for _ in range(layers)])

    def forward(self, x):
        for l in self.encoders:
            x = l(x)
        return x
