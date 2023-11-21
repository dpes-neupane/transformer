from torch import nn
from multihead import Multihead
class Decoder(nn.Module):
    def __init__(self,input_dim, emb_dim, heads, linear_dim, dropout=0.0):
        super().__init__()
        self.mmulatt = Multihead(input_dim, emb_dim, heads, decoder=True)
        self.mulatt = Multihead(
        _, selfatt= self.mmulatt.selfAttention(x, k, v)
        self.linear_layer = nn.Sequential(nn.Linear(input_dim, linear_dim), 
                                          nn.Dropout(dropout),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(linear_dim, input_dim))
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.ln3 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, k, v):
        mselfatt = self.mmulatt(x, ret_att=False)
        x = x + self.dropout(mselfatt)
        x = self.ln1(x)
        x = x + self.dropout(selfatt)
        x = self.ln2(x)
        lin_out = self.linear_layer(x)
        x = x + self.dropout(lin_out)
        x = self.ln3(x)
        return x

        





