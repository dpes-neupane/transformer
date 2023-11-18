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

    def forward(self, x):
        selfatt = self.mulatt(x, att=False)
        x = x + self.dropout(selfatt)
        x = self.ln1(x)
        linout = self.linear_layer(x)
        x = x + self.dropout(linout)
        x = self.ln2(x)
        return x

