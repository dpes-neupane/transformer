from torch import nn
from encoder import Encoder
class Encoders(nn.Module):
    def __init__(self, layers, **encoder_args):
        super().__init__()
        self.layers = layers
        self.encoders = nn.ModuleList([Encoder(**encoder_args) for _ in range(layers)])

    def forward(self, x):
        for l in self.encoders:
            x = l(x)
        return x
