import numpy as np

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


